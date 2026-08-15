import os

SENTIMENT_MODEL_URL = os.getenv("SENTIMENT_MODEL_URL", "http://sentiment-model:8002")
GOVERNANCE_URL = os.getenv("GOVERNANCE_URL", "http://governance-service:8010")
MANIFEST_TIMEOUT = float(os.getenv("MANIFEST_TIMEOUT", "5"))