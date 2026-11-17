"""Vision Snapshot: point a phone or laptop camera at something and a local
vision model (SmolVLM, run by llama.cpp inside the same container) names it or describes what is happening."""
import asyncio
import base64
import binascii
import json
import logging
import os
import re
import time
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

BASE_DIR = Path(__file__).resolve().parent.parent
# The model runs inside this container: llama.cpp's server, which speaks an OpenAI-compatible API.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
HEALTH_URL = os.getenv("LLM_HEALTH_URL", "http://127.0.0.1:8080/health")
MODEL = os.getenv("LLM_MODEL", "SmolVLM-500M-Instruct")  # label only; the server uses the model it loaded
MAX_DIM = int(os.getenv("MAX_DIM", "512"))
MAX_FRAMES = 6
MAX_FRAME_BYTES = 4_000_000
TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "1"))  # a CPU handles one image at a time

http: httpx.AsyncClient
api_slots = asyncio.Semaphore(MAX_CONCURRENT)
geo_cache: dict[tuple[float, float], str] = {}


class ModelError(Exception):
    """An error message that is safe and useful to show to the person using the app."""


async def probe_model() -> tuple[bool, str]:
    """(ready, reason) for the llama.cpp server."""
    try:
        r = await http.get(HEALTH_URL, timeout=2.0)
        return r.status_code == 200, f"health check answered HTTP {r.status_code}"
    except httpx.ConnectError:
        return False, "nothing is listening on that port, so the model server isn't running"
    except Exception as e:
        return False, type(e).__name__


async def model_ready() -> bool:
    return (await probe_model())[0]


async def watch_startup():
    """Logs when the model server is ready, and why not while it isn't."""
    why = ""
    for i in range(90):
        ok, why = await probe_model()
        if ok:
            log.info("Model '%s' is ready.", MODEL)
            return
        if i % 10 == 0:
            log.info("Waiting for the model server: %s", why)
        await asyncio.sleep(2)
    log.warning("The model server did not become ready at %s (%s).", HEALTH_URL, why)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http
    http = httpx.AsyncClient()
    warm = asyncio.create_task(watch_startup())
    yield
    warm.cancel()
    await http.aclose()


app = FastAPI(title="Vision Snapshot", version="3.0.0", lifespan=lifespan)


class NoCacheStatic(StaticFiles):
    """Always revalidate, so edits to the CSS/JS show up on a normal refresh."""
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", NoCacheStatic(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/status")
async def status():
    return {"model": MODEL, "ready": await model_ready()}


WARD = re.compile(r"\bward\s*\d+\b", re.I)


def place_name(a: dict) -> str:
    """Builds 'suburb, town, country' from a Nominatim address.
    South African data tags election wards as 'suburb' and municipalities as 'city',
    so we skip ward labels and prefer the town or village when there is one."""
    suburb = next((a[k] for k in ("suburb", "neighbourhood", "quarter")
                   if a.get(k) and not WARD.search(a[k])), "")
    town = a.get("town") or a.get("village") or a.get("city") or a.get("municipality") or ""
    if suburb == town:
        suburb = ""
    return ", ".join(p for p in (suburb, town, a.get("country", "")) if p)


async def reverse_geocode(lat: float, lon: float) -> str:
    """Turns coordinates into 'suburb, city, country' via Nominatim."""
    key = (round(lat, 3), round(lon, 3))
    if key in geo_cache:
        return geo_cache[key]
    try:
        r = await http.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "json", "lat": lat, "lon": lon, "zoom": 16},
            headers={"User-Agent": "SnapshotVisionEngine/3.0"},
            timeout=5.0,
        )
        if r.status_code == 200:
            name = place_name(r.json().get("address", {}))
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


STYLE_TASKS = {
    "identify": ("Identify the main object, person, animal, landmark, or activity. "
                 "Reply with only its name in 1 to 5 words, with no punctuation."),
    "describe": ("Describe this in three or four sentences. Cover who or what is in the frame, what they are doing, "
                 "their clothing, expression and posture if visible, the objects around them, the setting, "
                 "and the lighting. Only state what is visible; do not guess."),
    "narrate": ("Narrate this like a calm nature-documentary presenter, in two short sentences. "
                "Only mention things that are visible."),
    "story": ("Write a tiny three-sentence story inspired by what you see, in the style of a film-noir "
              "detective. Only use things that are visible."),
    "funny": ("Make one light-hearted, affectionate joke about the scene in a single sentence. "
              "Never insult anyone's appearance."),
    "funfact": ("Name the main object, animal or landmark, then give one short fun fact about it. "
                "If you are not sure what it is, say what it looks like instead of guessing."),
}
CREATIVE = {"narrate", "story", "funny", "funfact"}
MODES = set(STYLE_TASKS) | {"ask"}
MAX_TOKENS = {"ask": 120, "identify": 40, "describe": 220, "narrate": 150, "story": 170, "funny": 70, "funfact": 110, "scout": 6}


def build_prompt(mode: str, n_frames: int, place: str | None, question: str = "") -> str:
    source = (f"These {n_frames} images are frames from one short camera clip, in order."
              if n_frames > 1 else "This is a single photo from a camera.")
    where = f" It was taken near {place}." if place else ""
    if mode == "ask":
        noun = "images" if n_frames > 1 else "image"
        task = (f"Answer this question about the {noun}: {question} "
                "Reply briefly and only use what is visible. If the answer cannot be seen, say so.")
    else:
        task = STYLE_TASKS.get(mode, STYLE_TASKS["identify"])
    return f"{source}{where} {task}"


def log_timings(data: dict, elapsed: float) -> None:
    """Logs where the time went, using the llama.cpp server's own timing figures."""
    t = data.get("timings") or {}
    try:
        log.info("Took %.1fs. Reading photo and prompt: %.1fs (%.0f tokens/s). Writing the answer: %.1fs (%.1f tokens/s, %d tokens).",
                 elapsed, t.get("prompt_ms", 0) / 1000, t.get("prompt_per_second", 0),
                 t.get("predicted_ms", 0) / 1000, t.get("predicted_per_second", 0), int(t.get("predicted_n", 0)))
    except Exception:
        log.info("Took %.1fs.", elapsed)


async def ask_model(prompt: str, frames: list[str], mode: str) -> str:
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in frames
    ]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS.get(mode, 40),
        "temperature": 0.7 if mode in CREATIVE else 0.1,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        r = await http.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT)
    except httpx.ConnectError:
        log.warning("Can't connect to the model server at %s", LLM_BASE_URL)
        raise ModelError("The model isn't ready yet. Give it a minute and try again.")
    if r.status_code >= 400:
        try:
            body = r.json()
            err = body.get("error", body)
            detail = err.get("message", "") if isinstance(err, dict) else str(err)
        except Exception:
            detail = r.text[:200]
        log.warning("Model server returned %s: %s", r.status_code, detail)
        if r.status_code == 503:
            raise ModelError("The model is still loading. Give it a minute and try again.")
        raise ModelError(f"Model error: {detail or r.status_code}")
    data = r.json()
    log_timings(data, time.perf_counter() - started)
    msg = (data.get("choices") or [{}])[0].get("message", {})
    text = msg.get("content") or ""
    if isinstance(text, list):  # some servers return a list of content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text or "Nothing recognised"


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

    if not frames or not targets or api_slots.locked():
        return await out("skipped")
    options = [t.upper() for t in targets] + ["NONE"]
    prompt = (
        "Look at this image. Is any of these clearly visible: "
        + "; ".join(f"{t.upper()} = {SCOUT_TARGETS[t]}" for t in targets)
        + f"? If several apply, pick the most prominent. Reply with exactly one word: {', '.join(options)}."
    )
    try:
        async with api_slots:
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

            mode = msg.get("mode") if msg.get("mode") in MODES else "identify"
            question = re.sub(r"[\x00-\x1f]", " ", str(msg.get("question") or "")).strip()[:300]
            if mode == "ask" and not question:
                await reply("error", "Type a question first")
                continue
            coords = parse_location(msg.get("location"))
            typed = re.sub(r"[\x00-\x1f]", " ", str(msg.get("place") or "")).strip()[:80]
            place = typed or (await reverse_geocode(*coords) if coords else None)

            await reply("working")
            log.info("%s: %d frame(s)%s", mode, len(frames), f" near {place}" if place else "")
            try:
                async with api_slots:
                    label = await ask_model(build_prompt(mode, len(frames), place, question), frames, mode)
                log.info("Result: %s", label)
                await reply("success", label, location_name=place,
                            coords={"latitude": coords[0], "longitude": coords[1]} if coords else None)
            except ModelError as e:
                await reply("error", str(e))
            except httpx.TimeoutException:
                await reply("error", "The model took too long to answer")
            except httpx.RequestError:
                await reply("error", "Couldn't reach the model service.")
            except Exception as e:
                log.exception("Inference failed")
                await reply("error", f"Unexpected error: {e}")
    except WebSocketDisconnect:
        log.info("Client disconnected")