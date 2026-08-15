import os

MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
GOVERNANCE_URL = os.getenv("GOVERNANCE_URL", "http://governance-service:8010")
HTTP_TIMEOUT_SECONDS = float(os.getenv("SENTIMENT_HTTP_TIMEOUT", "10"))