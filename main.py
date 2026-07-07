import base64
import io
import json
import os
import re
import time
import asyncio
import logging

import httpx
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dispatch-api")

# ---- config ----
PROVIDER        = os.getenv("PROVIDER", "ollama")          # "ollama" | "openai"
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME      = os.getenv("MODEL_NAME", "gemma4:12b")
API_KEY         = os.getenv("API_KEY", "")
MAX_IMAGE_DIM   = int(os.getenv("MAX_IMAGE_DIM", "1600"))
JPEG_QUALITY    = int(os.getenv("JPEG_QUALITY", "85"))
NUM_CTX         = int(os.getenv("NUM_CTX", "6144"))
NUM_PREDICT     = int(os.getenv("NUM_PREDICT", "350"))
NUM_THREAD      = int(os.getenv("NUM_THREAD", "8"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/tiff"}
ALLOWED_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff"}

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    PROMPT_TEXT = f.read()

app = FastAPI(title="AMU Dispatch AI API")

_inference_lock = asyncio.Lock()


def check_api_key(x_api_key: str | None):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def validate_file(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File extension not allowed: {ext}")
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Content type not allowed: {file.content_type}")


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]", "_", name or "unknown")


def preprocess_image(raw_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(raw_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:300]}")
    return json.loads(text[start:end + 1])


async def call_ollama(image_b64: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEXT,
        "images": [image_b64],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repeat_penalty": 1.0,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "num_thread": NUM_THREAD,
        },
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        resp.raise_for_status()
    return extract_json(resp.json().get("response", ""))


async def call_openai_compatible(image_b64: str) -> dict:
    """Works with OpenAI, Groq, Together, LM Studio, or any OpenAI-compatible endpoint."""
    headers = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": NUM_PREDICT,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return extract_json(content)


async def call_model(image_b64: str) -> dict:
    if PROVIDER == "openai":
        return await call_openai_compatible(image_b64)
    return await call_ollama(image_b64)


@app.get("/health")
async def health():
    return {"status": "ok", "provider": PROVIDER, "model": MODEL_NAME}


@app.post("/v1/dispatch/extract")
async def extract_dispatch_fields(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    check_api_key(x_api_key)
    validate_file(file)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image_b64 = preprocess_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    safe_name = sanitize_filename(file.filename)
    start = time.time()
    async with _inference_lock:
        try:
            result = await call_model(image_b64)
        except httpx.HTTPStatusError as e:
            log.error("Model HTTP error: %s", e)
            raise HTTPException(status_code=502, detail="Model returned an error")
        except (ValueError, json.JSONDecodeError) as e:
            log.error("Bad JSON from model: %s", e)
            raise HTTPException(status_code=502, detail="Model did not return valid JSON")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Model request timed out")

    elapsed = round(time.time() - start, 2)
    log.info("Processed %s in %.2fs", safe_name, elapsed)

    return JSONResponse(content={"result": result, "processing_seconds": elapsed})
