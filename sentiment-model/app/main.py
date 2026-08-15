import httpx
from fastapi import FastAPI

from app.config import GOVERNANCE_URL, HTTP_TIMEOUT_SECONDS
from app.inference import predict
from shared.logging_config import configure_logging

logger = configure_logging("sentiment-model")

app = FastAPI(title="Sentiment Model Service")


@app.post("/predict")
async def predict_sentiment(payload: dict):
    result = predict(payload["text"])
    result["model_version"] = result["model_version"]

    governance_payload = {
        "patient_ref": payload["patient_ref"],
        "redacted_text": payload.get("redacted_text", payload["text"]),
        "manifest_hash": payload.get("manifest_hash"),
        "input_hash": result.pop("input_hash"),
        "label": result["label"],
        "confidence": result["confidence"],
        "model_version": result["model_version"],
        "latency_ms": result["latency_ms"],
    }

    flagged_for_review = None
    review_reason = None
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{GOVERNANCE_URL}/log-prediction", json=governance_payload)
            resp.raise_for_status()
            gov = resp.json()
            flagged_for_review = gov.get("flagged_for_review")
            review_reason = gov.get("review_reason")
    except (httpx.RequestError, httpx.HTTPStatusError):
        logger.warning("governance logging failed; returning prediction without log")

    result["flagged_for_review"] = flagged_for_review
    result["review_reason"] = review_reason
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}