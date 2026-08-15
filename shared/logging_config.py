"""Shared structured logging setup for all services."""
import json
import logging
import os
import sys

_LOGGER = ""


def configure_logging(name: str = "app", level: str | None = None) -> logging.Logger:
    """Structured JSON logging to stdout, with PII-safety as a default posture."""
    global _LOGGER
    _LOGGER = name
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel((level or os.getenv("LOG_LEVEL", "INFO")).upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "service": _LOGGER,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("patient_ref", "model_version", "label", "confidence", "latency_ms", "input_hash"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)