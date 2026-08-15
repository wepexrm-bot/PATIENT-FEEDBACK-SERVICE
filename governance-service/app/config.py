import json
import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
# Per-label acceptance thresholds; labels not listed fall back to
# CONFIDENCE_THRESHOLD. Example:
#   LABEL_CONFIDENCE_THRESHOLDS={"negative":0.7,"neutral":0.8,"positive":0.7}
LABEL_CONFIDENCE_THRESHOLDS: dict[str, float] = json.loads(
    os.getenv("LABEL_CONFIDENCE_THRESHOLDS", "{}")
)
# Sample-based review: queue every Nth prediction regardless of confidence
# (0 disables sampling). Catches confident-but-wrong predictions and gives the
# drift window an unbiased sample.
SAMPLE_REVIEW_EVERY_N = int(os.getenv("SAMPLE_REVIEW_EVERY_N", "0"))
DRIFT_WINDOW_SIZE = int(os.getenv("DRIFT_WINDOW_SIZE", "500"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://{}:{}@{}:{}/{}".format(
        os.getenv("POSTGRES_USER", "governance"),
        os.getenv("POSTGRES_PASSWORD", "changeme"),
        os.getenv("POSTGRES_HOST", "postgres"),
        os.getenv("POSTGRES_PORT", "5432"),
        os.getenv("POSTGRES_DB", "patient_feedback"),
    ),
)