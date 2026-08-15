from app.config import CONFIDENCE_THRESHOLD, LABEL_CONFIDENCE_THRESHOLDS


def needs_review(label: str, confidence: float) -> bool:
    """True when the model's confidence for `label` is below its acceptance bar.

    Uses the per-label threshold when configured, otherwise the global
    CONFIDENCE_THRESHOLD.
    """
    threshold = LABEL_CONFIDENCE_THRESHOLDS.get(label, CONFIDENCE_THRESHOLD)
    return confidence < threshold
