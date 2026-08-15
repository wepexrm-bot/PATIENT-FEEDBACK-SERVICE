from unittest.mock import AsyncMock, MagicMock, patch

import app.main as main_module
import pytest
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_sanitizer():
    """Mock the upstream PII sanitizer HTTP call."""
    payload = {
        "label": "positive",
        "confidence": 0.92,
        "model_version": "v1.0.0",
        "redacted_text": "The staff were [PATIENT_NAME] great",
    }
    with patch.object(main_module, "PII_SANITIZER_URL", "http://sanitizer:8001"), patch(
        "httpx.AsyncClient"
    ) as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_client_cls.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
        yield payload


def test_feedback_forwards_and_returns_sentiment(client, mock_sanitizer):
    resp = client.post(
        "/feedback",
        json={"patient_ref": "tok-123", "text": "Great care, thank you", "source": "portal"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert body["model_version"] == "v1.0.0"


def test_feedback_rejects_bad_source(client):
    resp = client.post(
        "/feedback",
        json={"patient_ref": "tok-1", "text": "hello", "source": "carrier-pigeon"},
    )
    assert resp.status_code == 422


def test_feedback_rejects_blank_text(client):
    resp = client.post(
        "/feedback",
        json={"patient_ref": "tok-1", "text": "   ", "source": "portal"},
    )
    assert resp.status_code == 422


def test_feedback_sanitizer_unavailable(client, monkeypatch):
    monkeypatch.setattr(main_module, "PII_SANITIZER_URL", "http://sanitizer:8001")
    monkeypatch.setattr(main_module.httpx, "AsyncClient", lambda *a, **k: AsyncMockWrap(_boom))
    resp = client.post(
        "/feedback",
        json={"patient_ref": "tok-1", "text": "hello world", "source": "sms"},
    )
    assert resp.status_code == 503


async def _boom(*args, **kwargs):
    raise __import__("httpx").RequestError("boom", request=None)


class AsyncMockWrap:
    def __init__(self, side_effect):
        self._side_effect = side_effect

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *args, **kwargs):
        await self._side_effect(*args, **kwargs)