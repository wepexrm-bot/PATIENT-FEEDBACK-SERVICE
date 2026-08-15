import os

PII_SANITIZER_URL = os.getenv("PII_SANITIZER_URL", "http://pii-sanitizer:8001")
HTTP_TIMEOUT_SECONDS = float(os.getenv("INGESTION_HTTP_TIMEOUT", "10"))