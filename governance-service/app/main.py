import json

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import OOV_RATIO_THRESHOLD, SAMPLE_REVIEW_EVERY_N
from app.db import SessionLocal, engine
from app.models import Base, Prediction, RedactionAudit, ReviewQueueItem
from app.review_api import router as review_router
from app.threshold import needs_review
from shared.logging_config import configure_logging
from shared.security import get_manifest_key, hash_patient_ref, require_key

logger = configure_logging("governance-service")


def _ensure_schema():
    """Create the restricted 'redaction' schema (Postgres); no-op on SQLite."""
    if engine.dialect.name == "sqlite":
        return
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS redaction"))
        conn.commit()


_ensure_schema()
Base.metadata.create_all(bind=engine)
if engine.dialect.name != "sqlite":
    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS oov_score DOUBLE PRECISION")
        )

app = FastAPI(title="Governance Service")
app.include_router(review_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionLog(BaseModel):
    model_config = {"protected_namespaces": ()}
    patient_ref: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(..., min_length=1)
    input_hash: str | None = None
    latency_ms: float | None = None
    redacted_text: str | None = None
    manifest_hash: str | None = None
    oov_score: float | None = Field(default=None, ge=0.0, le=1.0)


class AuditPayload(BaseModel):
    patient_ref_hash: str = Field(..., min_length=1)
    manifest: list = Field(..., min_length=1)
    manifest_hash: str = Field(..., min_length=1)


@app.post("/log-prediction")
def log_prediction(payload: PredictionLog):
    with SessionLocal() as session:
        pred = Prediction(
            patient_ref=hash_patient_ref(payload.patient_ref),
            label=payload.label,
            confidence=payload.confidence,
            model_version=payload.model_version,
            input_hash=payload.input_hash,
            latency_ms=payload.latency_ms,
            redacted_text=payload.redacted_text,
            manifest_hash=payload.manifest_hash,
            oov_score=payload.oov_score,
        )
        session.add(pred)
        session.commit()
        session.refresh(pred)

        flagged = needs_review(pred.label, pred.confidence)
        sampled = SAMPLE_REVIEW_EVERY_N > 0 and pred.id % SAMPLE_REVIEW_EVERY_N == 0
        domain_shift = OOV_RATIO_THRESHOLD > 0 and (pred.oov_score or 0.0) >= OOV_RATIO_THRESHOLD
        reasons = []
        if flagged:
            reasons.append("low-confidence")
        if sampled:
            reasons.append("sampled")
        if domain_shift:
            reasons.append("domain-shift")
        queued = bool(reasons)
        reason = "+".join(reasons) if reasons else None
        if queued:
            session.add(ReviewQueueItem(prediction_id=pred.id, reason=reason))
            session.commit()

        logger.info(
            "prediction logged",
            extra={
                "model_version": payload.model_version,
                "label": payload.label,
                "confidence": payload.confidence,
            },
        )
        return {"status": "logged", "flagged_for_review": queued, "review_reason": reason}


@app.post("/redaction-audit")
def redaction_audit(
    payload: AuditPayload,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
):
    if not require_key([get_manifest_key()], x_api_key):
        raise HTTPException(status_code=401, detail="invalid manifest API key")
    with SessionLocal() as session:
        audit = RedactionAudit(
            patient_ref_hash=payload.patient_ref_hash,
            manifest_hash=payload.manifest_hash,
            manifest=json.dumps(payload.manifest),
        )
        session.add(audit)
        session.commit()
        return {"status": "stored"}


@app.get("/health")
async def health():
    return {"status": "ok"}