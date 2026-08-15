import app.auth as app_auth
import app.main as main_module
import pytest
from app.db import SessionLocal
from app.main import app
from app.models import Prediction, ReviewQueueItem
from fastapi.testclient import TestClient

REVIEWER_TOKEN = "review-token-abc"
ADMIN_TOKEN = "admin-token-xyz"


@pytest.fixture()
def client(monkeypatch):
    # configure auth tokens for the test session
    monkeypatch.setattr(app_auth, "get_review_keys", lambda: [REVIEWER_TOKEN, ADMIN_TOKEN])
    monkeypatch.setattr(main_module, "get_manifest_key", lambda: "manifest-key")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def seed_predictions():
    """Insert one low-confidence (flagged + queued) and one high-confidence prediction."""
    with SessionLocal() as session:
        low = Prediction(
            patient_ref="hash-low",
            label="positive",
            confidence=0.4,
            model_version="v1.0.0",
            input_hash="abc",
            redacted_text="The [PATIENT_NAME] was rude",
        )
        high = Prediction(
            patient_ref="hash-high",
            label="positive",
            confidence=0.95,
            model_version="v1.0.0",
            input_hash="def",
            redacted_text="The staff were kind",
        )
        session.add_all([low, high])
        session.flush()
        queue_item = ReviewQueueItem(prediction_id=low.id)
        session.add(queue_item)
        session.commit()
        session.refresh(queue_item)
        yield {"queue_id": queue_item.id, "prediction_id": low.id}


def _auth_headers(role="reviewer", token=REVIEWER_TOKEN):
    return {"Authorization": f"Bearer {token}", "X-Reviewer-Role": role}


def test_log_prediction_flags_low_confidence(client):
    resp = client.post(
        "/log-prediction",
        json={
            "patient_ref": "tok-123",
            "label": "positive",
            "confidence": 0.4,
            "model_version": "v1.0.0",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["flagged_for_review"] is True


def test_log_prediction_confident_not_flagged(client):
    resp = client.post(
        "/log-prediction",
        json={
            "patient_ref": "tok-456",
            "label": "negative",
            "confidence": 0.9,
            "model_version": "v1.0.0",
        },
    )
    assert resp.json()["flagged_for_review"] is False


def test_log_prediction_sampled_high_confidence(client, monkeypatch):
    monkeypatch.setattr(main_module, "SAMPLE_REVIEW_EVERY_N", 1)
    resp = client.post(
        "/log-prediction",
        json={
            "patient_ref": "tok-smp",
            "label": "positive",
            "confidence": 0.9,
            "model_version": "v1.0.0",
        },
    )
    assert resp.json()["flagged_for_review"] is True
    items = client.get("/review-queue").json()["items"]
    assert items and items[0]["reason"] == "sampled"


def test_log_prediction_sampled_and_low_confidence(client, monkeypatch):
    monkeypatch.setattr(main_module, "SAMPLE_REVIEW_EVERY_N", 1)
    resp = client.post(
        "/log-prediction",
        json={
            "patient_ref": "tok-both",
            "label": "neutral",
            "confidence": 0.4,
            "model_version": "v1.0.0",
        },
    )
    assert resp.json()["flagged_for_review"] is True
    items = client.get("/review-queue").json()["items"]
    assert items and items[0]["reason"] == "both"


def test_queue_reason_low_confidence(client):
    client.post(
        "/log-prediction",
        json={
            "patient_ref": "tok-low",
            "label": "negative",
            "confidence": 0.4,
            "model_version": "v1.0.0",
        },
    )
    items = client.get("/review-queue").json()["items"]
    assert items and items[0]["reason"] == "low-confidence"


def test_queue_reason_none_without_sampling_or_flag(client):
    client.post(
        "/log-prediction",
        json={
            "patient_ref": "tok-clean",
            "label": "positive",
            "confidence": 0.95,
            "model_version": "v1.0.0",
        },
    )
    assert client.get("/review-queue").json()["items"] == []


def test_log_prediction_hashes_patient_ref(client):
    client.post(
        "/log-prediction",
        json={"patient_ref": "secret-ref", "label": "neutral", "confidence": 0.9,
              "model_version": "v1.0.0", "redacted_text": "ok", "input_hash": "x"},
    )
    with SessionLocal() as session:
        rows = session.query(Prediction).all()
        assert all("secret-ref" not in (r.patient_ref or "") for r in rows)


def test_review_queue_returns_context(client, seed_predictions):
    resp = client.get("/review-queue")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    item = items[0]
    assert item["redacted_text"] == "The [PATIENT_NAME] was rude"
    assert item["label"] == "positive"
    assert item["confidence"] == 0.4


def test_submit_review_requires_auth(client, seed_predictions):
    resp = client.post(f"/review-queue/{seed_predictions['queue_id']}", json={"corrected_label": "negative", "reviewer": "r"})
    assert resp.status_code in (401, 422)


def test_submit_review_bad_role_rejected(client, seed_predictions):
    headers = _auth_headers(role="reporter")
    resp = client.post(
        f"/review-queue/{seed_predictions['queue_id']}",
        json={"corrected_label": "negative", "reviewer": "alice"},
        headers=headers,
    )
    assert resp.status_code == 403


def test_submit_review_success(client, seed_predictions):
    resp = client.post(
        f"/review-queue/{seed_predictions['queue_id']}",
        json={"corrected_label": "negative", "reviewer": "alice"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    # queue should now be empty of pending items
    assert client.get("/review-queue").json()["items"] == []


def test_submit_review_double_submit_rejected(client, seed_predictions):
    headers = _auth_headers()
    client.post(f"/review-queue/{seed_predictions['queue_id']}", json={"corrected_label": "negative", "reviewer": "a"}, headers=headers)
    resp = client.post(f"/review-queue/{seed_predictions['queue_id']}", json={"corrected_label": "positive", "reviewer": "a"}, headers=headers)
    assert resp.status_code == 409


def test_redaction_audit_requires_api_key(client):
    payload = {"patient_ref_hash": "h", "manifest": [{"type": "SSN", "start": 3, "end": 8}], "manifest_hash": "mh"}
    assert client.post("/redaction-audit", json=payload).status_code == 401
    ok = client.post("/redaction-audit", json=payload, headers={"X-Api-Key": "manifest-key"})
    assert ok.status_code == 200


def test_drift_endpoint_requires_admin(client, seed_predictions):
    resp = client.post("/drift/v1.0.0", headers=_auth_headers(role="reviewer"))
    assert resp.status_code == 403


def test_drift_compute_and_read(client, seed_predictions):
    # review the flagged item so human ground truth exists
    headers = _auth_headers()
    item_id = client.get("/review-queue").json()["items"][0]["id"]
    client.post(f"/review-queue/{item_id}", json={"corrected_label": "negative", "reviewer": "alice"}, headers=headers)

    admin = _auth_headers(role="admin")
    resp = client.post("/drift/v1.0.0", headers=admin)
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_size"] == 1
    assert body["rolling_accuracy"] == 0.0  # predicted positive, corrected negative

    read = client.get("/drift/v1.0.0")
    assert read.status_code == 200
    assert read.json()["rolling_accuracy"] == 0.0
    assert read.json()["degraded"] != "none"
