import httpx
from fastapi import FastAPI, HTTPException

from app.config import HTTP_TIMEOUT_SECONDS, PII_SANITIZER_URL
from app.schemas import FeedbackIn, FeedbackOut
from shared.logging_config import configure_logging

logger = configure_logging("ingestion-api")

app = FastAPI(title="Ingestion API")


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