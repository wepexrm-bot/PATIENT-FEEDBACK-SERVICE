import datetime
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import drift as drift_engine
from app.auth import require_role
from app.config import DRIFT_WINDOW_SIZE
from app.db import SessionLocal
from app.models import DriftMetric, Prediction, ReviewQueueItem

router = APIRouter()


class ReviewSubmit(BaseModel):
    corrected_label: str = Field(..., min_length=1)
    reviewer: str = Field(..., min_length=1)


@router.get("/review-queue")
def list_pending_reviews():
    with SessionLocal() as session:
        items = (
            session.query(ReviewQueueItem)
            .join(Prediction, ReviewQueueItem.prediction_id == Prediction.id)
            .filter(ReviewQueueItem.status == "pending")
            .all()
        )
        reviewed = [
            {
                "id": i.id,
                "prediction_id": i.prediction_id,
                "reason": i.reason,
                "redacted_text": i.prediction.redacted_text,
                "label": i.prediction.label,
                "confidence": i.prediction.confidence,
                "model_version": i.prediction.model_version,
                "oov_score": i.prediction.oov_score,
            }
            for i in items
        ]
    return {"items": reviewed}


@router.post("/review-queue/{item_id}", dependencies=[Depends(require_role("reviewer"))])
def submit_review(item_id: int, body: ReviewSubmit):
    with SessionLocal() as session:
        item = session.get(ReviewQueueItem, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="review item not found")
        if item.status != "pending":
            raise HTTPException(status_code=409, detail="review item already reviewed")
        item.corrected_label = body.corrected_label
        item.reviewer = body.reviewer
        item.status = "reviewed"
        item.reviewed_at = datetime.datetime.now(datetime.timezone.utc)
        session.commit()
        return {"status": "ok", "id": item.id}


@router.post("/drift/{model_version}", dependencies=[Depends(require_role("admin"))])
def compute_drift(model_version: str):
    with SessionLocal() as session:
        rows = (
            session.query(Prediction, ReviewQueueItem)
            .join(ReviewQueueItem, ReviewQueueItem.prediction_id == Prediction.id)
            .filter(
                Prediction.model_version == model_version,
                ReviewQueueItem.status == "reviewed",
            )
            .order_by(ReviewQueueItem.reviewed_at.desc())
            .limit(DRIFT_WINDOW_SIZE)
            .all()
        )
        reviewed = [
            {"label": p.label, "corrected_label": r.corrected_label} for p, r in rows
        ]
        # reference distribution = the model's own auto labels for this version
        reference = {}
        for p, _ in rows:
            reference[p.label] = reference.get(p.label, 0) + 1

        summary = drift_engine.summarize(reviewed, reference)
        metric = DriftMetric(
            model_version=model_version,
            rolling_accuracy=summary["rolling_accuracy"],
            label_psi=summary["label_psi"],
            degraded=summary["degraded"],
            window_size=summary["window_size"],
            note=json.dumps({"current_distribution": summary["current_distribution"]}),
        )
        session.add(metric)
        session.commit()
        return summary


@router.get("/drift/{model_version}")
def get_drift(model_version: str):
    with SessionLocal() as session:
        latest = (
            session.query(DriftMetric)
            .filter(DriftMetric.model_version == model_version)
            .order_by(DriftMetric.computed_at.desc())
            .first()
        )
        if latest is None:
            raise HTTPException(status_code=404, detail=f"no drift metrics for {model_version}")
        return {
            "model_version": latest.model_version,
            "rolling_accuracy": latest.rolling_accuracy,
            "label_psi": latest.label_psi,
            "degraded": latest.degraded,
            "window_size": latest.window_size,
            "computed_at": str(latest.computed_at),
        }