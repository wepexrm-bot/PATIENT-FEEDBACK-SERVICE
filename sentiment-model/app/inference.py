import hashlib
import time

from app.model_loader import get_version, load_predictor, oov_score


def predict(text: str) -> dict:
    predictor = load_predictor()
    start = time.perf_counter()
    label, confidence = predictor(text)
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "label": label,
        "confidence": confidence,
        "model_version": get_version(),
        "latency_ms": round(latency_ms, 2),
        "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "oov_score": oov_score(text),
    }