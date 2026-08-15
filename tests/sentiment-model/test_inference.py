import json
import sys
from pathlib import Path

import pytest
from app import inference, model_loader


@pytest.fixture()
def joblib_weights(tmp_path, monkeypatch):
    """Build a tiny 3-class sklearn pipeline and stage it as model.joblib + meta.json."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    corpus = [
        "nurse was kind and helpful", "terrible wait times", "average care overall",
        "staff rushed me through", "clean facility great doctor", "ok not great not bad",
    ]
    y = ["positive", "negative", "neutral", "negative", "positive", "neutral"]

    vect = TfidfVectorizer()
    clf = LogisticRegression()
    clf.fit(vect.fit_transform(corpus), y)

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("vect", vect), ("clf", clf)])

    weights = tmp_path / "weights"
    weights.mkdir()
    import joblib
    joblib.dump(pipe, weights / "model.joblib")
    (weights / "meta.json").write_text(
        json.dumps({"model_version": "v9.9.9", "labels": ["negative", "neutral", "positive"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(model_loader, "MODEL_PATH", str(weights))
    monkeypatch.setattr(model_loader, "_META", None)
    monkeypatch.setattr(model_loader, "_PREDICTOR", None)
    # keep the unit fast + deterministic: bypass NLTK cleaning
    monkeypatch.setattr(model_loader.sacr_compat, "data_cleaning", lambda t: t)
    yield weights


def test_joblib_backend_predicts(joblib_weights):
    result = inference.predict("nurse was kind and helpful")
    assert result["label"] == "positive"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] == "v9.9.9"
    assert result["latency_ms"] >= 0.0


def test_predict_returns_input_hash(joblib_weights):
    text = "average care overall"
    result = inference.predict(text)
    import hashlib
    assert result["input_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_model_version_defaults_when_missing(joblib_weights, monkeypatch):
    weights_dir = joblib_weights
    meta = json.loads((weights_dir / "meta.json").read_text(encoding="utf-8"))
    meta.pop("model_version")
    (weights_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(model_loader, "_META", None)
    monkeypatch.setattr(model_loader, "MODEL_VERSION", "v-fallback")
    assert model_loader.get_version() == "v-fallback"


def test_backend_name_detects_joblib(joblib_weights):
    assert model_loader.backend_name() == "joblib"


def test_backend_detection_prefers_onnx_when_present(joblib_weights, monkeypatch):
    weights = Path(joblib_weights)
    (weights / "model.onnx").touch(exist_ok=True)
    model_loader._PREDICTOR = None
    assert model_loader.backend_name() == "onnx"


def test_load_meta_labels(joblib_weights):
    assert model_loader.get_labels() == ["negative", "neutral", "positive"]


def test_onnx_predictor_with_fake_session(monkeypatch, tmp_path):
    """Inject a fake onnxruntime module to exercise the ONNX load path end-to-end."""
    import types

    import numpy as np

    class _FakeInput:
        name = "X"

    class _FakeOutput:
        name = "output_probability"

    class _FakeSession:
        def get_inputs(self):
            return [_FakeInput()]

        def get_outputs(self):
            return [_FakeOutput()]

        def run(self, *_):
            probs = np.array([[0.1, 0.2, 0.7]])
            return [np.array(["positive"], dtype="<U8"), probs]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda path: _FakeSession()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.onnx").touch(exist_ok=True)
    (weights / "meta.json").write_text(
        json.dumps({"model_version": "v1.0.0", "labels": ["negative", "neutral", "positive"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(weights))
    monkeypatch.setattr(model_loader, "_META", None)
    monkeypatch.setattr(model_loader, "_PREDICTOR", None)
    monkeypatch.setattr(model_loader.sacr_compat, "data_cleaning", lambda t: t)

    predictor = model_loader.load_predictor()
    label, conf = predictor("great care")
    assert label == "positive"
    assert conf == pytest.approx(0.7)