"""Vision Snapshot: point a phone or laptop camera at something and a local
vision LLM (served by Ollama) names it or describes what is happening."""
import asyncio
import base64
import binascii
import ipaddress
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:  # optional: server-side downscaling. The browser already resizes frames.
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("vision")

BASE_DIR = Path(__file__).resolve().parent
_ollama = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = _ollama if _ollama.startswith(("http://", "https://")) else f"http://{_ollama}"
# Must be a vision model (qwen3:8b is text-only). qwen3-vl:2b is light enough for a CPU.
MODEL = os.getenv("VISION_MODEL", "qwen3-vl:2b")
MAX_DIM = int(os.getenv("MAX_DIM", "768"))
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))
MAX_FRAMES = 6
MAX_FRAME_BYTES = 4_000_000
TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300"))

http: httpx.AsyncClient
gpu_lock = asyncio.Lock()  # one inference at a time, extra phones queue up
geo_cache: dict[tuple[float, float], str] = {}
ip_geo_cache: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http
    http = httpx.AsyncClient()
    warm = asyncio.create_task(pre_warm_model())
    yield
    warm.cancel()
    await http.aclose()


app = FastAPI(title="Vision Snapshot", version="2.0.0", lifespan=lifespan)
class NoCacheStatic(StaticFiles):
    """Always revalidate, so edits to the CSS/JS show up on a normal refresh."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", NoCacheStatic(directory=BASE_DIR / "static"), name="static")


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
            "stream": False, "keep_alive": "24h", "think": False,
            "options": {"num_predict": 1, "num_ctx": NUM_CTX},
        })
        log.info("Model ready.")
    except Exception as e:
        log.warning("Warm-up failed: %s", e)


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
async def status():
    return await model_state()


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def visitor_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")


@app.get("/api/ip-location")
async def ip_location(request: Request):
    """City-level location from the visitor's public IP, used when the browser
    cannot get a GPS or Wi-Fi fix. On a LAN the visitor's IP is private, so we
    look up this server's own public IP instead (same network, same city)."""
    ip = visitor_ip(request)
    try:
        target = ip if ipaddress.ip_address(ip).is_global else ""
    except ValueError:
        target = ""
    hit = ip_geo_cache.get(target)
    if hit and time.time() - hit["t"] < 3600:
        return hit["v"]
    try:
        d = (await http.get(f"https://ipwho.is/{target}", timeout=5.0)).json()
        if d.get("success"):
            v = {"latitude": d["latitude"], "longitude": d["longitude"]}
            ip_geo_cache[target] = {"v": v, "t": time.time()}
            return v
    except Exception as e:
        log.info("IP location unavailable: %s", e)
    return JSONResponse({"error": "unavailable"}, status_code=503)


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
        "think": False,  # skip the reasoning phase: it is slow on CPU and would eat the short answer budget
        "keep_alive": "24h",
        "options": {
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
            "num_predict": {"describe": 160, "scout": 6}.get(mode, 40),
        },
    }
    r = await http.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    text = r.json().get("message", {}).get("content", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # older Ollama may inline reasoning
    return text or "Nothing recognised"


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


SCOUT_TARGETS = {
    "person": "a person (face or body)",
    "landmark": "a well-known landmark, monument, or notable building",
}


async def run_scout(ws: WebSocket, msg: dict):
    """Cheap one-word check that Watch mode uses to decide whether a full analysis is worth running."""
    rid = msg.get("id")
    targets = [t for t in msg.get("targets", []) if t in SCOUT_TARGETS]
    frames = prepare_frames(msg.get("images") or [])[:1]

    async def out(status: str, hit=None):
        await ws.send_json({"id": rid, "type": "scout", "status": status, "hit": hit})

    if not frames or not targets or gpu_lock.locked():
        return await out("skipped")
    options = [t.upper() for t in targets] + ["NONE"]
    prompt = (
        "Look at this image. Is any of these clearly visible: "
        + "; ".join(f"{t.upper()} = {SCOUT_TARGETS[t]}" for t in targets)
        + f"? If several apply, pick the most prominent. Reply with exactly one word: {', '.join(options)}."
    )
    try:
        async with gpu_lock:
            text = (await ask_model(prompt, frames, "scout")).upper()
        found = [(text.find(o), o) for o in options if o in text]
        hit = min(found)[1].lower() if found else None
        hit = hit if hit in targets else None
        log.info("Scout: %s", hit or "nothing")
        await out("success", hit)
    except Exception as e:
        log.info("Scout failed: %s", e)
        await out("error")


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
            if msg.get("type") == "scout":
                await run_scout(ws, msg)
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
