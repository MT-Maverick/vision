"""Vision Snapshot: point a phone or laptop camera at something and a local
vision LLM (served by Ollama) names it or describes what is happening."""
import asyncio
import base64
import binascii
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:  # optional: server-side downscaling. The browser already resizes frames.
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("vision")

BASE_DIR = Path(__file__).resolve().parent
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# qwen3:8b is text-only, so it cannot read images. qwen2.5vl:3b and gemma3 can.
MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
MAX_DIM = int(os.getenv("MAX_DIM", "512"))
MAX_FRAMES = 6
MAX_FRAME_BYTES = 4_000_000
TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300"))

http: httpx.AsyncClient
gpu_lock = asyncio.Lock()  # one inference at a time, extra phones queue up
geo_cache: dict[tuple[float, float], str] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http
    http = httpx.AsyncClient()
    warm = asyncio.create_task(pre_warm_model())
    yield
    warm.cancel()
    await http.aclose()


app = FastAPI(title="Vision Snapshot", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


async def model_state() -> dict:
    """Reports whether Ollama is reachable and the model is installed/loaded."""
    try:
        tags = (await http.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)).json()
        installed = [m.get("name", "") for m in tags.get("models", [])]
        running = (await http.get(f"{OLLAMA_URL}/api/ps", timeout=3.0)).json()
        loaded = [m.get("name", "") for m in running.get("models", [])]
        return {
            "model": MODEL,
            "ollama": True,
            "installed": any(n.startswith(MODEL) for n in installed),
            "loaded": any(n.startswith(MODEL) for n in loaded),
            "available": installed,
        }
    except Exception:
        return {"model": MODEL, "ollama": False, "installed": False, "loaded": False}


async def pre_warm_model():
    state = await model_state()
    if not state["ollama"]:
        log.warning("Ollama is not reachable at %s. Start it with `ollama serve`.", OLLAMA_URL)
        return
    if not state["installed"]:
        log.warning("Model '%s' is not installed. Run `ollama pull %s`. Installed: %s",
                    MODEL, MODEL, ", ".join(state["available"]) or "none")
        return
    if state["loaded"]:
        log.info("Model '%s' is already in memory.", MODEL)
        return
    log.info("Loading '%s' into memory...", MODEL)
    try:
        await http.post(f"{OLLAMA_URL}/api/chat", timeout=180.0, json={
            "model": MODEL, "messages": [{"role": "user", "content": "hi"}],
            "stream": False, "keep_alive": "1h", "options": {"num_predict": 1},
        })
        log.info("Model ready.")
    except Exception as e:
        log.warning("Warm-up failed: %s", e)


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/status")
async def status():
    return await model_state()


async def reverse_geocode(lat: float, lon: float) -> str:
    """Turns coordinates into 'suburb, city, country' via Nominatim."""
    key = (round(lat, 3), round(lon, 3))
    if key in geo_cache:
        return geo_cache[key]
    try:
        r = await http.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 16},
            headers={"User-Agent": "SnapshotVisionEngine/2.0"},
            timeout=5.0,
        )
        if r.status_code == 200:
            a = r.json().get("address", {})
            parts = [
                a.get("suburb") or a.get("neighbourhood") or a.get("village") or "",
                a.get("city") or a.get("town") or a.get("municipality") or "",
                a.get("country", ""),
            ]
            name = ", ".join(p for p in parts if p)
            if name:
                geo_cache[key] = name
                return name
    except Exception as e:
        log.info("Reverse geocoding unavailable: %s", e)
    return f"{lat:.4f}, {lon:.4f}"


def prepare_frames(images: list) -> list[str]:
    """Strips data-URL prefixes, validates, and downsizes frames if OpenCV is present."""
    out = []
    for raw in images[:MAX_FRAMES]:
        if not isinstance(raw, str):
            continue
        if "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            continue
        if not data or len(data) > MAX_FRAME_BYTES:
            continue
        if cv2 is not None:
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if max(h, w) > MAX_DIM:
                s = MAX_DIM / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            data = enc.tobytes()
        out.append(base64.b64encode(data).decode())
    return out


def build_prompt(mode: str, n_frames: int, place: str | None) -> str:
    source = (f"These {n_frames} images are frames from one short camera clip, in order."
              if n_frames > 1 else "This is a single photo from a camera.")
    where = f" It was taken near {place}." if place else ""
    if mode == "describe":
        task = ("Describe what is happening in one or two short sentences. "
                "Mention the main subject, what it is doing, and the setting. "
                "Only state what is visible.")
    else:
        task = ("Identify the main object, person, animal, landmark, or activity. "
                "Reply with only its name in 1 to 5 words, with no punctuation.")
    return f"{source}{where} {task}"


async def ask_model(prompt: str, frames: list[str], mode: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": frames}],
        "stream": False,
        "keep_alive": "1h",
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
            "num_predict": 160 if mode == "describe" else 40,
        },
    }
    r = await http.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "").strip() or "Nothing recognised"


def ollama_error(e: httpx.HTTPStatusError) -> str:
    try:
        return e.response.json().get("error", e.response.text)
    except Exception:
        return e.response.text


def parse_location(loc) -> tuple[float, float] | None:
    try:
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except (TypeError, KeyError, ValueError):
        pass
    return None


@app.websocket("/ws/vision")
async def vision_websocket(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                continue
            rid = msg.get("id")

            async def reply(status: str, label: str = "", **extra):
                await ws.send_json({"id": rid, "status": status, "label": label, **extra})

            images = msg.get("images") or ([msg["image"]] if msg.get("image") else [])
            frames = prepare_frames(images)
            if not frames:
                await reply("error", "No usable image received")
                continue

            mode = "describe" if msg.get("mode") == "describe" else "identify"
            coords = parse_location(msg.get("location"))
            place = await reverse_geocode(*coords) if coords else None

            await reply("working")
            log.info("%s: %d frame(s)%s", mode, len(frames), f" near {place}" if place else "")
            try:
                async with gpu_lock:
                    label = await ask_model(build_prompt(mode, len(frames), place), frames, mode)
                log.info("Result: %s", label)
                await reply("success", label, location_name=place,
                            coords={"latitude": coords[0], "longitude": coords[1]} if coords else None)
            except httpx.ConnectError:
                await reply("error", f"Cannot reach Ollama at {OLLAMA_URL}. Is it running?")
            except httpx.HTTPStatusError as e:
                detail = ollama_error(e)
                log.warning("Ollama error: %s", detail)
                await reply("error", f"Ollama: {detail}")
            except httpx.TimeoutException:
                await reply("error", "The model took too long to answer")
            except Exception as e:
                log.exception("Inference failed")
                await reply("error", f"Unexpected error: {e}")
    except WebSocketDisconnect:
        log.info("Client disconnected")
