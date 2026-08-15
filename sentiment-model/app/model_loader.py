"""Loads the sentiment model once at startup.

Supports two backends, auto-detected from the weights directory:
  - `model.onnx`   -> ONNX runtime (sklearn pipelines exported via skl2onnx)
  - `model.joblib` -> pickled sklearn pipeline produced by the SACR converter

The classifier is exposed as a plain callable:  run(text) -> (label, confidence).
"""
import json
from pathlib import Path

from app import sacr_compat
from app.config import MODEL_PATH, MODEL_VERSION

_META = None


def load_meta() -> dict:
    global _META
    if _META is None:
        meta_path = Path(MODEL_PATH) / "meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            _META = json.load(f)
    return _META


def get_version() -> str:
    meta = load_meta()
    return str(meta.get("model_version") or MODEL_VERSION)


def get_labels() -> list[str]:
    meta = load_meta()
    labels = list(meta.get("labels", []))
    if not labels:
        raise RuntimeError("meta.json must declare the ordered class labels")
    return labels


def backend_name() -> str:
    weights = Path(MODEL_PATH)
    if (weights / "model.onnx").exists():
        return "onnx"
    if (weights / "model.joblib").exists():
        return "joblib"
    raise FileNotFoundError(
        f"no model found in {MODEL_PATH!s}: expected model.onnx or model.joblib"
    )


def _load_joblib_predictor():
    import joblib

    model = joblib.load(str(Path(MODEL_PATH) / "model.joblib"))
    labels = get_labels()
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("converted model must expose predict_proba")

    def run(text: str) -> tuple[str, float]:
        cleaned = sacr_compat.data_cleaning(text)
        probs = model.predict_proba([cleaned])[0]
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])

    return run


def _load_onnx_predictor():
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(Path(MODEL_PATH) / "model.onnx"))
    labels = get_labels()
    name = session.get_inputs()[0].name
    prob_names = [o.name for o in session.get_outputs()]

    def run(text: str) -> tuple[str, float]:
        results = session.run(None, {name: np.array([text])})
        # take the probability tensor (prefer one named with 'prob' or the last output)
        probs = None
        for r, oname in zip(results, prob_names):
            arr = np.asarray(r)
            if arr.ndim >= 2 and arr.shape[-1] == len(labels) and "lab" not in oname.lower():
                probs = arr
        if probs is None:
            probs = np.asarray(results[-1])
        probs = np.asarray(probs)
        if probs.ndim != 1:
            probs = probs[0] if probs.shape[0] == 1 else np.max(probs, axis=0)
        probs = np.asarray(probs, dtype=float).ravel()
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])

    return run


_PREDICTOR = None


def load_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        backend = backend_name()
        if backend == "joblib":
            _PREDICTOR = _load_joblib_predictor()
        else:
            _PREDICTOR = _load_onnx_predictor()
    return _PREDICTOR