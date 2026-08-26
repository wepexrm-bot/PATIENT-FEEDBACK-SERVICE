import time
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import HTTP_TIMEOUT_SECONDS, PII_SANITIZER_URL
from app.schemas import FeedbackIn, FeedbackOut
from shared.logging_config import configure_logging

logger = configure_logging("ingestion-api")

app = FastAPI(title="Ingestion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 60


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/feedback":
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < 60]
        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        _rate_limit_store[client_ip].append(now)
    return await call_next(request)


@app.post("/feedback", response_model=FeedbackOut)
async def submit_feedback(payload: FeedbackIn):
    # Forward raw text to sanitizer immediately. Never persist raw text here.
    logger.info("forwarding feedback to sanitizer", extra={"patient_ref": _mask(payload.patient_ref)})
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{PII_SANITIZER_URL}/sanitize", json=payload.model_dump())
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error("sanitizer returned error status", extra={"status_code": exc.response.status_code})
        raise HTTPException(status_code=exc.response.status_code, detail="sanitizer upstream error")
    except httpx.RequestError:
        logger.error("sanitizer unreachable")
        raise HTTPException(status_code=503, detail="sanitizer service unavailable")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _mask(ref: str) -> str:
    """Log only a short prefix of the patient reference, never the full token."""
    return ref[:8] + "..." if ref else ""