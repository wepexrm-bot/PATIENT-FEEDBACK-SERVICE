import datetime
import os

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# The PHI-adjacent audit table lives in its own schema on Postgres for tighter
# access control; SQLite (tests/dev) has no schema support, so it's unqualified.
if os.getenv("DATABASE_URL", "").startswith("sqlite"):
    _RESTRICTED_SCHEMA = None
else:
    _RESTRICTED_SCHEMA = "redaction"


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    patient_ref = Column(String, index=True)          # hashed, never raw
    label = Column(String)
    confidence = Column(Float)
    model_version = Column(String, index=True)
    input_hash = Column(String, index=True)           # sha256 of sanitized text
    latency_ms = Column(Float, nullable=True)
    redacted_text = Column(Text, nullable=True)       # sanitized text shown to reviewers
    manifest_hash = Column(String, nullable=True)     # links to redaction audit
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


class ReviewQueueItem(Base):
    __tablename__ = "review_queue"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    status = Column(String, default="pending")  # pending | reviewed
    reason = Column(String, nullable=True)      # low-confidence | sampled | both
    corrected_label = Column(String, nullable=True)
    reviewer = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    prediction = relationship("Prediction")


class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    id = Column(Integer, primary_key=True)
    model_version = Column(String, index=True)
    rolling_accuracy = Column(Float)
    label_psi = Column(Float, nullable=True)
    degraded = Column(String, nullable=True)   # e.g. "accuracy" | "psi" | "none"
    window_size = Column(Integer)
    computed_at = Column(DateTime, default=datetime.datetime.utcnow)
    note = Column(Text, nullable=True)


class RedactionAudit(Base):
    """Access-restricted audit record for PII redaction.

    Intended to live in its own schema/DB with tighter access control than the
    operational tables. Only hashed patient references are stored — never raw.
    """
    __tablename__ = "redaction_audit"
    __table_args__ = ({"schema": _RESTRICTED_SCHEMA} if _RESTRICTED_SCHEMA else {})
    id = Column(Integer, primary_key=True)
    patient_ref_hash = Column(String, index=True)
    manifest_hash = Column(String, index=True)
    manifest = Column(Text)  # JSON: what was removed, offsets, types
    created_at = Column(DateTime, default=datetime.datetime.utcnow)