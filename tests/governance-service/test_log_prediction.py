from app import main
from app.main import app
from fastapi.testclient import TestClient


def _log(client, *, confidence, label="positive", oov_score=None):
    payload = {
        "patient_ref": "tok-1",
        "label": label,
        "confidence": confidence,
        "model_version": "v1.0.0",
        "redacted_text": "some text",
    }
    if oov_score is not None:
        payload["oov_score"] = oov_score
    return client.post("/log-prediction", json=payload)


def test_low_confidence_flagged_with_reason():
    with TestClient(app) as c:
        resp = _log(c, confidence=0.5)
    assert resp.status_code == 200
    body = resp.json()
    assert body["flagged_for_review"] is True
    assert body["review_reason"] == "low-confidence"


def test_high_confidence_not_flagged():
    with TestClient(app) as c:
        resp = _log(c, confidence=0.97)
    body = resp.json()
    assert body["flagged_for_review"] is False
    assert body["review_reason"] is None


def test_sampled_reason(monkeypatch):
    monkeypatch.setattr(main, "SAMPLE_REVIEW_EVERY_N", 1)
    with TestClient(app) as c:
        resp = _log(c, confidence=0.97)
    body = resp.json()
    assert body["flagged_for_review"] is True
    assert body["review_reason"] == "sampled"


def test_domain_shift_flagged(monkeypatch):
    monkeypatch.setattr(main, "OOV_RATIO_THRESHOLD", 0.3)
    with TestClient(app) as c:
        resp = _log(c, confidence=0.97, oov_score=0.5)
    body = resp.json()
    assert body["flagged_for_review"] is True
    assert body["review_reason"] == "domain-shift"


def test_domain_shift_high_confidence_still_flagged(monkeypatch):
    monkeypatch.setattr(main, "OOV_RATIO_THRESHOLD", 0.3)
    with TestClient(app) as c:
        resp = _log(c, confidence=0.99, oov_score=0.4)
    body = resp.json()
    assert body["flagged_for_review"] is True
    assert "domain-shift" in body["review_reason"]