import os
import json
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple, Union

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.wsgi import WSGIMiddleware
from flask import Flask, render_template

from pydantic import BaseModel, Field

try:
    from gpt4all import GPT4All
except Exception as e:  # pragma: no cover - install-time error surface
    GPT4All = None  # type: ignore

try:
    # Optional dependency for embeddings
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore


# -----------------------
# Configuration
# -----------------------
MODELS_DIR = os.getenv("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))
MODEL_NAME = os.getenv("MODEL_NAME")  # can be a gguf filename or a known model name
MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "1"))
REQUIRE_API_KEY = (os.getenv("REQUIRE_API_KEY", "true").lower() in ("1", "true", "yes"))
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "sk-local-default")
_api_keys_env = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
API_KEYS = {k.strip() for k in _api_keys_env.split(",") if k.strip()}
if DEFAULT_API_KEY:
    API_KEYS.add(DEFAULT_API_KEY)
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# Where to persist ingested training examples (JSONL files)
BASE_DIR = os.path.dirname(__file__)
TRAINING_DATA_DIR = os.getenv("TRAINING_DATA_DIR", os.path.join(BASE_DIR, "data", "training"))

# Server start time
START_TIME = time.time()


# If MODEL_NAME not provided, try to detect a single gguf in models/
def _auto_detect_model() -> Optional[str]:
    try:
        files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".gguf")]
        if len(files) == 1:
            return files[0]
    except FileNotFoundError:
        return None
    return None


if not MODEL_NAME:
    MODEL_NAME = _auto_detect_model()

if not MODEL_NAME:
    # Prefer default model if present
    try:
        default_candidate = "mistral-7b.gguf"
        if default_candidate in [f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".gguf")]:
            MODEL_NAME = default_candidate
            print(f"[INFO] Defaulting MODEL_NAME to '{MODEL_NAME}'")
    except FileNotFoundError:
        pass

if not MODEL_NAME:
    # We keep going; requests will raise a 500 until a model is available.
    print("[WARN] No model detected. Set MODEL_NAME env var or add a single .gguf to models/ directory.")


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="OpenAI-compatible GPT4All Server",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# Model lifecycle
# -----------------------
_model_cache: Dict[str, GPT4All] = {}
_model_lru: List[str] = []  # names in MRU order (end is most recently used)
_model_lock = asyncio.Lock()
_emb_model: Optional[SentenceTransformer] = None
_emb_lock = asyncio.Lock()
_training_lock = asyncio.Lock()

# -----------------------
# Metrics
# -----------------------
_metrics_lock = asyncio.Lock()
_metrics: Dict[str, Any] = {
    "start_time": START_TIME,
    "total_requests": 0,
    "chat_requests": 0,
    "completion_requests": 0,
    "streaming_requests": 0,
    "errors": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "per_model": {},  # model_name -> counters dict
}


async def _bump_metrics(
    *,
    kind: str,
    model_name: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    streaming: bool = False,
):
    # kind: "chat" | "completion" | "error"
    async with _metrics_lock:
        _metrics["total_requests"] = int(_metrics.get("total_requests", 0)) + 1
        if kind == "chat":
            _metrics["chat_requests"] = int(_metrics.get("chat_requests", 0)) + 1
        elif kind == "completion":
            _metrics["completion_requests"] = int(_metrics.get("completion_requests", 0)) + 1
        elif kind == "error":
            _metrics["errors"] = int(_metrics.get("errors", 0)) + 1

        if streaming:
            _metrics["streaming_requests"] = int(_metrics.get("streaming_requests", 0)) + 1

        _metrics["prompt_tokens"] = int(_metrics.get("prompt_tokens", 0)) + int(prompt_tokens)
        _metrics["completion_tokens"] = int(_metrics.get("completion_tokens", 0)) + int(completion_tokens)

        per_model: Dict[str, Any] = _metrics.setdefault("per_model", {})  # type: ignore
        mm = per_model.setdefault(model_name, {
            "requests": 0,
            "chat": 0,
            "completion": 0,
            "streaming": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        })
        mm["requests"] += 1
        if kind == "chat":
            mm["chat"] += 1
        elif kind == "completion":
            mm["completion"] += 1
        if streaming:
            mm["streaming"] += 1
        mm["prompt_tokens"] += int(prompt_tokens)
        mm["completion_tokens"] += int(completion_tokens)


# -----------------------
# Auth
# -----------------------
async def verify_api_key(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    api_key: Optional[str] = Header(default=None),  # some clients send 'api-key'
):
    if not REQUIRE_API_KEY:
        return

    def _extract_bearer_token(header_val: Optional[str]) -> Optional[str]:
        if not header_val:
            return None
        parts = header_val.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
        return None

    token = _extract_bearer_token(authorization) or (x_api_key.strip() if x_api_key else None) or (api_key.strip() if api_key else None)

    if not API_KEYS:
        # Server requires API key but none configured
        raise HTTPException(status_code=401, detail="API key required. Configure API_KEYS env var.")

    if not token or token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _gguf_files() -> List[str]:
    try:
        return sorted([f for f in os.listdir(MODELS_DIR) if f.lower().endswith(".gguf")])
    except FileNotFoundError:
        return []


async def get_model(model_name: Optional[str] = None) -> Tuple[str, GPT4All]:
    """Return (model_name, model_instance) ensuring requested model is loaded.

    If model_name is None, uses configured MODEL_NAME or auto-detected single model.
    Maintains a small in-memory cache of loaded models.
    """
    global _model_cache, _model_lru
    if GPT4All is None:
        raise HTTPException(status_code=500, detail="gpt4all package not installed")

    # Resolve the model name
    resolved_name = model_name or MODEL_NAME or _auto_detect_model()
    if not resolved_name:
        raise HTTPException(status_code=500, detail="No model configured. Set MODEL_NAME or place exactly one .gguf in models/.")

    # Validate the requested model exists in MODELS_DIR
    available = set(_gguf_files())
    if resolved_name not in available:
        raise HTTPException(status_code=400, detail=f"Requested model '{resolved_name}' not found in {MODELS_DIR}")

    async with _model_lock:
        if resolved_name in _model_cache:
            # Update LRU order
            if resolved_name in _model_lru:
                _model_lru.remove(resolved_name)
            _model_lru.append(resolved_name)
            return resolved_name, _model_cache[resolved_name]

        # Load new model
        print(f"[INFO] Loading GPT4All model '{resolved_name}' from {MODELS_DIR} ...")
        model = GPT4All(resolved_name, model_path=MODELS_DIR)
        _model_cache[resolved_name] = model
        _model_lru.append(resolved_name)
        print("[INFO] Model loaded.")

        # Evict if cache too large
        while len(_model_cache) > max(1, MODEL_CACHE_SIZE):
            evict_name = _model_lru.pop(0)
            if evict_name == resolved_name:
                # Shouldn't happen, but guard
                continue
            old = _model_cache.pop(evict_name, None)
            try:
                if old and hasattr(old, "close"):
                    old.close()
                    print(f"[INFO] Unloaded model '{evict_name}'")
            except Exception:
                pass

        return resolved_name, model


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "OpenAI-compatible GPT4All server", "default_model": MODEL_NAME, "models_dir": MODELS_DIR}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# -----------------------
# Mount Web UI (Flask) at /chat
# -----------------------
def _build_flask_ui() -> Flask:
    base_dir = os.path.dirname(__file__)
    template_dir = os.path.join(base_dir, "webui", "templates")
    static_dir = os.path.join(base_dir, "webui", "static")

    # static_url_path must match mount prefix to avoid clashing at root
    flask_app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
        static_url_path="/chat/static",
    )

    @flask_app.context_processor
    def inject_defaults():
        # DEFAULT_API_BASE empty => same-origin
        return {
            "DEFAULT_API_BASE": os.getenv("API_BASE_URL", ""),
            "DEFAULT_MODEL": os.getenv("UI_DEFAULT_MODEL", "mistral-7b.gguf"),
            "DEFAULT_API_KEY": DEFAULT_API_KEY,
        }

    @flask_app.route("/")
    def chat_index():
        return render_template("index.html")

    return flask_app


# Mount under /chat (public)
try:
    _flask_ui = _build_flask_ui()
    app.mount("/chat", WSGIMiddleware(_flask_ui))
except Exception as e:
    print(f"[WARN] Failed to mount Flask UI at /chat: {e}")


@app.get("/models")
async def list_local_models(dep: None = Depends(verify_api_key)) -> Dict[str, Any]:
    files = _gguf_files()
    data = []
    for name in files:
        full = os.path.join(MODELS_DIR, name)
        try:
            st = os.stat(full)
            size = st.st_size
            mtime = int(st.st_mtime)
        except FileNotFoundError:
            size = None
            mtime = None
        data.append({
            "id": name,
            "filename": name,
            "size": size,
            "modified": mtime,
            "is_default": (name == MODEL_NAME),
            "is_loaded": (name in _model_cache),
        })
    return {"models_dir": MODELS_DIR, "count": len(data), "data": data}


# -----------------------
# Embeddings (OpenAI-compatible)
# -----------------------
class EmbeddingsRequest(BaseModel):
    model: Optional[str] = None  # ignored for now; uses EMBEDDING_MODEL_ID
    input: Union[str, List[str]]
    user: Optional[str] = None


async def get_embeddings_model() -> SentenceTransformer:
    global _emb_model
    if SentenceTransformer is None:
        raise HTTPException(status_code=500, detail="sentence-transformers not installed; install to use embeddings")
    async with _emb_lock:
        if _emb_model is None:
            print(f"[INFO] Loading embeddings model '{EMBEDDING_MODEL_ID}' on device '{EMBEDDING_DEVICE}' ...")
            _emb_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=EMBEDDING_DEVICE)
            print("[INFO] Embeddings model loaded.")
    assert _emb_model is not None
    return _emb_model


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingsRequest, dep: None = Depends(verify_api_key)) -> Dict[str, Any]:
    emb_model = await get_embeddings_model()
    inputs = req.input if isinstance(req.input, list) else [req.input]
    # SentenceTransformers encodes to list of lists (floats)
    vectors = emb_model.encode(inputs, normalize_embeddings=True)
    data = []
    for i, vec in enumerate(vectors):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": vec.tolist(),
        })
    total_tokens = sum(estimate_tokens(x) for x in inputs)
    return {
        "object": "list",
        "data": data,
        "model": req.model or EMBEDDING_MODEL_ID,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


# -----------------------
# Pydantic schemas (subset of OpenAI)
# -----------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[List[str], str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


class CompletionsRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[List[str], str]] = None
    user: Optional[str] = None


# -----------------------
# Training ingestion schemas
# -----------------------
class TrainingExample(BaseModel):
    prompt: str
    response: str
    meta: Optional[Dict[str, Any]] = None


class TrainingIngestRequest(BaseModel):
    dataset: str = Field(..., description="Name for the dataset (used as filename)")
    examples: List[TrainingExample] = Field(..., min_length=1, description="List of prompt/response examples")
    upsert: Optional[bool] = Field(False, description="Avoid duplicating exact prompt+response pairs")
    store_embeddings: Optional[bool] = Field(False, description="If true, compute and store embeddings for each example")


# -----------------------
# Helpers
# -----------------------
def build_chat_prompt(messages: List[ChatMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.role.lower()
        if role == "system":
            parts.append(f"### System\n{m.content}\n")
        elif role == "user":
            parts.append(f"### User\n{m.content}\n")
        elif role == "assistant":
            parts.append(f"### Assistant\n{m.content}\n")
        else:
            parts.append(f"### {m.role.capitalize()}\n{m.content}\n")
    parts.append("### Assistant\n")
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    # Very rough heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _dataset_path(name: str) -> str:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".")) or "default"
    if not safe.endswith(".jsonl"):
        safe += ".jsonl"
    return os.path.join(TRAINING_DATA_DIR, safe)


def make_chat_completion_response(
    *,
    request_id: str,
    created: int,
    model_name: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_completion_response(
    *,
    request_id: str,
    created: int,
    model_name: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def sse_event_lines_chat(
    *,
    request_id: str,
    created: int,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[Union[List[str], str]],
) -> AsyncGenerator[bytes, None]:
    model = await get_model()
    first_chunk = True
    generated_tokens = 0
    content_accum: List[str] = []

    def _should_stop(token: str) -> bool:
        if stop is None:
            return False
        if isinstance(stop, str):
            return stop in token or any("".join(content_accum).endswith(stop) for _ in [0])
        for s in stop:
            if s and (s in token or "".join(content_accum).endswith(s)):
                return True
        return False

    for token in model.generate(
        prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        streaming=True,
    ):
        content_accum.append(token)
        generated_tokens += 1

        chunk: Dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": ({"role": "assistant", "content": token} if first_chunk else {"content": token}),
                    "finish_reason": None,
                }
            ],
        }
        first_chunk = False
        yield (f"data: {JSONResponse(content=chunk).body.decode()}\n\n").encode("utf-8")

        if _should_stop(token):
            break

    # final stop chunk
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield (f"data: {JSONResponse(content=final_chunk).body.decode()}\n\n").encode("utf-8")
    yield b"data: [DONE]\n\n"
    # After finishing stream, bump metrics
    try:
        await _bump_metrics(
            kind="chat",
            model_name=model_name,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=generated_tokens,
            streaming=True,
        )
    except Exception:
        pass


async def sse_event_lines_completion(
    *,
    request_id: str,
    created: int,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[Union[List[str], str]],
) -> AsyncGenerator[bytes, None]:
    model = await get_model()
    content_accum: List[str] = []
    generated_tokens = 0

    def _should_stop(token: str) -> bool:
        if stop is None:
            return False
        if isinstance(stop, str):
            return stop in token or "".join(content_accum).endswith(stop)
        for s in stop:
            if s and (s in token or "".join(content_accum).endswith(s)):
                return True
        return False

    for token in model.generate(
        prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        streaming=True,
    ):
        content_accum.append(token)
        generated_tokens += 1
        chunk = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "text": token,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        yield (f"data: {JSONResponse(content=chunk).body.decode()}\n\n").encode("utf-8")
        if _should_stop(token):
            break

    final_chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": "",
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }
    yield (f"data: {JSONResponse(content=final_chunk).body.decode()}\n\n").encode("utf-8")
    yield b"data: [DONE]\n\n"
    # After finishing stream, bump metrics
    try:
        await _bump_metrics(
            kind="completion",
            model_name=model_name,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=generated_tokens,
            streaming=True,
        )
    except Exception:
        pass


# -----------------------
# OpenAI-compatible endpoints
# -----------------------
@app.get("/v1/models")
async def list_models(dep: None = Depends(verify_api_key)) -> Dict[str, Any]:
    files = _gguf_files()
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "gpt4all-local",
            }
            for name in files
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest, request: Request, dep: None = Depends(verify_api_key)):
    model_name, model = await get_model(req.model)
    model_name = req.model or model_name or "gpt4all"
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    prompt = build_chat_prompt(req.messages)
    stopping = req.stop

    if req.stream:
        async def event_gen():
            async for line in sse_event_lines_chat(
                request_id=request_id,
                created=created,
                model_name=model_name,
                prompt=prompt,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
                max_tokens=req.max_tokens or 256,
                stop=stopping,
            ):
                yield line

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # Non-streaming path
    text = model.generate(
        prompt,
        max_tokens=req.max_tokens or 256,
        temp=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
        streaming=False,
    )
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(text)
    resp = make_chat_completion_response(
        request_id=request_id,
        created=created,
        model_name=model_name,
        content=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    try:
        await _bump_metrics(
            kind="chat",
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            streaming=False,
        )
    except Exception:
        pass
    return JSONResponse(content=resp)


@app.post("/v1/completions")
async def completions(req: CompletionsRequest, request: Request, dep: None = Depends(verify_api_key)):
    model_name, model = await get_model(req.model)
    model_name = req.model or model_name or "gpt4all"
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4().hex}"

    prompt = req.prompt
    stopping = req.stop

    if req.stream:
        async def event_gen():
            async for line in sse_event_lines_completion(
                request_id=request_id,
                created=created,
                model_name=model_name,
                prompt=prompt,
                temperature=req.temperature or 0.7,
                top_p=req.top_p or 0.9,
                max_tokens=req.max_tokens or 256,
                stop=stopping,
            ):
                yield line

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    # Non-streaming path
    text = model.generate(
        prompt,
        max_tokens=req.max_tokens or 256,
        temp=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
        streaming=False,
    )
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(text)
    resp = make_completion_response(
        request_id=request_id,
        created=created,
        model_name=model_name,
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    try:
        await _bump_metrics(
            kind="completion",
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            streaming=False,
        )
    except Exception:
        pass
    return JSONResponse(content=resp)


# -----------------------
# Training ingestion endpoints
# -----------------------
@app.post("/v1/training/ingest")
async def ingest_training(req: TrainingIngestRequest, dep: None = Depends(verify_api_key)) -> Dict[str, Any]:
    _ensure_dir(TRAINING_DATA_DIR)
    path = _dataset_path(req.dataset)

    # Optional embeddings model
    emb_model: Optional[SentenceTransformer] = None
    if req.store_embeddings:
        try:
            emb_model = await get_embeddings_model()
        except HTTPException:
            emb_model = None

    # Build rows and handle upsert
    rows: List[Dict[str, Any]] = []
    existing: set = set()
    if req.upsert and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        k = (obj.get("prompt", ""), obj.get("response", ""))
                        existing.add(k)
                    except Exception:
                        continue
        except Exception:
            pass

    # Prepare new examples
    for ex in req.examples:
        key = (ex.prompt, ex.response)
        if req.upsert and key in existing:
            continue
        rec: Dict[str, Any] = {
            "prompt": ex.prompt,
            "response": ex.response,
            "meta": ex.meta or {},
        }
        if emb_model is not None:
            try:
                # Store embeddings for prompt and response (normalized)
                pv = emb_model.encode([ex.prompt], normalize_embeddings=True)[0].tolist()
                rv = emb_model.encode([ex.response], normalize_embeddings=True)[0].tolist()
                rec["embedding_prompt"] = pv
                rec["embedding_response"] = rv
            except Exception:
                # Do not fail the whole batch on embedding error
                rec["embedding_error"] = True
        rows.append(rec)

    # Persist to JSONL (append)
    if rows:
        async with _training_lock:
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "dataset": os.path.basename(path),
        "saved": len(rows),
        "skipped": (len(req.examples) - len(rows)) if req.upsert else 0,
        "path": path,
        "embeddings": bool(req.store_embeddings),
    }


@app.get("/v1/training/datasets")
async def list_training_datasets(dep: None = Depends(verify_api_key)) -> Dict[str, Any]:
    _ensure_dir(TRAINING_DATA_DIR)
    files: List[Dict[str, Any]] = []
    try:
        for name in sorted(os.listdir(TRAINING_DATA_DIR)):
            if not name.endswith(".jsonl"):
                continue
            full = os.path.join(TRAINING_DATA_DIR, name)
            try:
                st = os.stat(full)
                files.append({
                    "name": name,
                    "size": st.st_size,
                    "modified": int(st.st_mtime),
                    "path": full,
                })
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        pass
    return {"count": len(files), "data": files}


# -----------------------
# Metrics endpoints
# -----------------------
def _process_stats() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "uptime_sec": int(time.time() - START_TIME),
        "cpu_percent": None,
        "memory_rss_bytes": None,
        "threads": None,
    }
    if psutil is None:
        return info
    try:
        p = psutil.Process(os.getpid())
        info["cpu_percent"] = p.cpu_percent(interval=0.0)
        info["memory_rss_bytes"] = p.memory_info().rss
        info["threads"] = p.num_threads()
    except Exception:
        pass
    return info


@app.get("/metrics")
async def get_metrics(dep: None = Depends(verify_api_key)) -> Dict[str, Any]:
    async with _metrics_lock:
        snapshot = dict(_metrics)
    snapshot["process"] = _process_stats()
    snapshot["server_time"] = int(time.time())
    return snapshot


@app.get("/status")
async def status_page() -> HTMLResponse:
        # Self-contained status dashboard (no external app required)
        default_key = DEFAULT_API_KEY or ""
        default_key_js = json.dumps(default_key)
        html = """
        <!doctype html>
        <html>
            <head>
                <meta charset='utf-8'/>
                <meta name='viewport' content='width=device-width, initial-scale=1'/>
                <title>Server Status</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    :root { --bg:#ffffff; --fg:#1f2328; --muted:#6a737d; --accent:#2f81f7; --card:#f6f8fa; }
                    html, body { height:100%; margin:0; padding:0; background:var(--bg); color:var(--fg); font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
                    .container { max-width: 1200px; margin: 0 auto; padding: 16px; }
                    header { display:flex; align-items:center; justify-content:space-between; margin-bottom: 16px; }
                    header h1 { margin:0; font-size: 20px; }
                    .card { background: var(--card); border:1px solid #e1e4e8; border-radius: 8px; padding: 12px; }
                    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
                    .metrics { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px; }
                    .metric .label { color: var(--muted); font-size: 12px; }
                    .metric .value { font-size: 20px; font-weight: 600; }
                    table { width: 100%; border-collapse: collapse; }
                    th, td { border-bottom: 1px solid #e1e4e8; padding: 8px; text-align:left; font-size: 13px; }
                    th { background: #f0f3f6; }
                    .controls { display:flex; gap:8px; align-items:center; }
                    input[type="password"], input[type="text"] { padding:6px 8px; border:1px solid #d0d7de; border-radius:6px; width:320px; }
                    button { border:1px solid #d0d7de; background:#fff; border-radius:6px; padding:6px 10px; cursor:pointer; }
                    .muted { color: var(--muted); font-size: 12px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <header>
                        <h1>GPT4All Server Status</h1>
                        <div class="controls">
                            <label for="apiKey" class="muted">API Key</label>
                            <input id="apiKey" type="password" placeholder="sk-..." />
                            <button id="toggleKey">Show</button>
                            <label for="interval" class="muted">Refresh</label>
                            <input id="interval" type="text" value="2" style="width:48px" />
                            <span class="muted">sec</span>
                        </div>
                    </header>

                    <div class="metrics">
                        <div class="card metric"><div class="label">Total requests</div><div class="value" id="mRequests">0</div></div>
                        <div class="card metric"><div class="label">Streaming requests</div><div class="value" id="mStreaming">0</div></div>
                        <div class="card metric"><div class="label">Errors</div><div class="value" id="mErrors">0</div></div>
                        <div class="card metric"><div class="label">Uptime</div><div class="value" id="mUptime">0s</div></div>
                    </div>

                    <div class="grid">
                        <div class="card"><canvas id="cpuChart" height="140"></canvas></div>
                        <div class="card"><canvas id="rssChart" height="140"></canvas></div>
                    </div>

                    <div class="grid" style="margin-top:12px;">
                        <div class="card"><canvas id="reqChart" height="140"></canvas></div>
                        <div class="card"><canvas id="tokChart" height="140"></canvas></div>
                    </div>

                    <div class="card" style="margin-top:12px;">
                        <h3 style="margin:0 0 8px 0; font-size:16px;">Per-model usage</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Requests</th>
                                    <th>Chat</th>
                                    <th>Completion</th>
                                    <th>Streaming</th>
                                    <th>Prompt tokens</th>
                                    <th>Completion tokens</th>
                                </tr>
                            </thead>
                            <tbody id="perModelBody"></tbody>
                        </table>
                    </div>

                    <p class="muted" style="margin-top:8px;">Charts update by polling <code>/metrics</code>. API key is prefilled from server default.</p>
                </div>

                <script>
                    const DEFAULT_API_KEY = __DEFAULT_API_KEY__;
                    const apiKeyInput = document.getElementById('apiKey');
                    const toggleKeyBtn = document.getElementById('toggleKey');
                    const intervalInput = document.getElementById('interval');
                    apiKeyInput.value = DEFAULT_API_KEY || '';
                    toggleKeyBtn.addEventListener('click', () => {
                        apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
                        toggleKeyBtn.textContent = apiKeyInput.type === 'password' ? 'Show' : 'Hide';
                    });

                    const mRequests = document.getElementById('mRequests');
                    const mStreaming = document.getElementById('mStreaming');
                    const mErrors = document.getElementById('mErrors');
                    const mUptime = document.getElementById('mUptime');
                    const perModelBody = document.getElementById('perModelBody');

                    const cpuChart = new Chart(document.getElementById('cpuChart').getContext('2d'), {
                        type: 'line', data: { labels: [], datasets: [{ label:'CPU %', data: [], borderColor:'#2f81f7', tension:0.2, pointRadius:0 }] },
                        options: { responsive:true, maintainAspectRatio:false, animation:false, scales:{ y:{ suggestedMin:0, suggestedMax:100 } } }
                    });
                    const rssChart = new Chart(document.getElementById('rssChart').getContext('2d'), {
                        type: 'line', data: { labels: [], datasets: [{ label:'RSS (MB)', data: [], borderColor:'#8250df', tension:0.2, pointRadius:0 }] },
                        options: { responsive:true, maintainAspectRatio:false, animation:false }
                    });
                    const reqChart = new Chart(document.getElementById('reqChart').getContext('2d'), {
                        type: 'line', data: { labels: [], datasets: [{ label:'Total Requests', data: [], borderColor:'#3fb950', tension:0.2, pointRadius:0 }] },
                        options: { responsive:true, maintainAspectRatio:false, animation:false }
                    });
                    const tokChart = new Chart(document.getElementById('tokChart').getContext('2d'), {
                        type: 'line', data: { labels: [], datasets: [
                            { label:'Prompt tokens', data: [], borderColor:'#d29922', tension:0.2, pointRadius:0 },
                            { label:'Completion tokens', data: [], borderColor:'#f85149', tension:0.2, pointRadius:0 },
                        ] }, options: { responsive:true, maintainAspectRatio:false, animation:false }
                    });

                    const MAX_POINTS = 300;
                    function pushPoint(chart, x, y, dsIndex=0) {
                        chart.data.labels.push(x);
                        chart.data.datasets[dsIndex].data.push(y);
                        if (chart.data.labels.length > MAX_POINTS) {
                            chart.data.labels.shift();
                            chart.data.datasets.forEach(d => d.data.shift());
                        }
                        chart.update('none');
                    }

                    function fmtUptime(sec) {
                        sec = Math.max(0, parseInt(sec||0,10));
                        const h = Math.floor(sec/3600); const m = Math.floor((sec%3600)/60); const s = sec%60;
                        return `${h}h ${m}m ${s}s`;
                    }

                    async function fetchMetrics() {
                        const headers = {};
                        const key = apiKeyInput.value.trim();
                        if (key) headers['Authorization'] = `Bearer ${key}`;
                        const res = await fetch('/metrics', { headers });
                        if (!res.ok) throw new Error(`HTTP ${res.status}`);
                        return await res.json();
                    }

                    async function tick() {
                        try {
                            const data = await fetchMetrics();
                            const t = new Date((data.server_time||Math.floor(Date.now()/1000))*1000).toLocaleTimeString();
                            const p = data.process||{};
                            const cpu = (p.cpu_percent==null)? null : p.cpu_percent;
                            const rssMB = (p.memory_rss_bytes==null)? null : (p.memory_rss_bytes/(1024*1024));

                            if (cpu!=null) pushPoint(cpuChart, t, cpu, 0);
                            if (rssMB!=null) pushPoint(rssChart, t, rssMB, 0);
                            pushPoint(reqChart, t, data.total_requests||0, 0);
                            pushPoint(tokChart, t, data.prompt_tokens||0, 0);
                            pushPoint(tokChart, t, data.completion_tokens||0, 1);

                            mRequests.textContent = data.total_requests||0;
                            mStreaming.textContent = data.streaming_requests||0;
                            mErrors.textContent = data.errors||0;
                            mUptime.textContent = fmtUptime((p.uptime_sec)||0);

                            const per = data.per_model||{};
                            let rows = '';
                            Object.keys(per).sort().forEach(k => {
                                const v = per[k]||{};
                                rows += `<tr><td>${k}</td><td>${v.requests||0}</td><td>${v.chat||0}</td><td>${v.completion||0}</td><td>${v.streaming||0}</td><td>${v.prompt_tokens||0}</td><td>${v.completion_tokens||0}</td></tr>`;
                            });
                            perModelBody.innerHTML = rows || '<tr><td colspan="7" class="muted">No data</td></tr>';
                        } catch (e) {
                            console.warn('metrics fetch failed', e);
                        } finally {
                            const ms = Math.max(500, parseInt(intervalInput.value||'2',10)*1000);
                            setTimeout(tick, ms);
                        }
                    }

                    tick();
                </script>
            </body>
        </html>
        """
        html = html.replace("__DEFAULT_API_KEY__", default_key_js)
        return HTMLResponse(content=html)


# -------------
# Run: uvicorn main:app --reload
# -------------
if __name__ == "__main__":
    # Allow running as: python3 main.py
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=os.getenv("RELOAD", "false").lower() == "true")
