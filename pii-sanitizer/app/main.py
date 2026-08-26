from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import SENTIMENT_MODEL_URL
from app.manifest import manifest_hash, store_manifest
from app.ner_recognizer import apply_ner_redaction, ner_available, warmup
from app.rules import apply_rule_based_redaction
from shared.logging_config import configure_logging

logger = configure_logging("pii-sanitizer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ner_available():
        logger.info("warming up NER engine (Presidio)")
        try:
            warmup()
        except Exception:  # noqa: BLE001 - NER must never break startup
            logger.warning("NER warmup failed; continuing without NER")
    yield


app = FastAPI(title="PII Sanitizer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sanitize")
async def sanitize(payload: dict):
    text = payload["text"]
    patient_ref = payload["patient_ref"]

    # Pass 1: deterministic rules
    text, rule_manifest = apply_rule_based_redaction(text)
    # Pass 2: NER-based detection on remaining free text (optional at runtime)
    ner_manifest: list[dict] = []
    if ner_available():
        try:
            text, ner_manifest = apply_ner_redaction(text)
        except Exception:  # noqa: BLE001 - NER must never break sanitization
            logger.warning("NER pass failed; continuing with rule-based redaction only")
    else:
        logger.warning("Presidio unavailable; continuing with rule-based redaction only")

    manifests = rule_manifest + ner_manifest
    await store_manifest(patient_ref, manifests)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SENTIMENT_MODEL_URL}/predict",
            json={
                "text": text,
                "patient_ref": patient_ref,
                "redacted_text": text,
                "manifest_hash": manifest_hash(manifests),
            },
        )
        resp.raise_for_status()
        result = resp.json()

    return {**result, "redacted_text": text}


@app.get("/health")
async def health():
    return {"status": "ok"}