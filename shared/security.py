"""Shared security utilities: token generation/verification and role checks.

Review endpoints must be authenticated with role-based access (reviewer vs admin).
Keep this dependency-free so it can run in any service.
"""
import hashlib
import hmac
import os
import secrets
from collections.abc import Sequence


def _normalise_keys(raw: str) -> list[str]:
    """Parse a comma-separated REVIEW_API_KEYS env var into tokens."""
    return [t.strip() for t in raw.split(",") if t.strip()]


def get_review_keys() -> list[str]:
    return _normalise_keys(os.getenv("REVIEW_API_KEYS", ""))


def get_manifest_key() -> str:
    return os.getenv("MANIFEST_API_KEY", "").strip()


def require_key(raw_keys: Sequence[str], provided: str | None) -> bool:
    """Constant-time check that `provided` matches one of the acceptable keys, if any are configured.

    If no keys are configured the check is skipped (local/dev convenience) —
    production deployments should always set REVIEW_API_KEYS / MANIFEST_API_KEY.
    """
    if not raw_keys:
        return True
    if not provided:
        return False
    return any(hmac.compare_digest(provided, k) for k in raw_keys)


def hash_patient_ref(patient_ref: str) -> str:
    """Hash a patient reference before persistence; never store the raw token."""
    return hashlib.sha256(patient_ref.encode("utf-8")).hexdigest()


def generate_token() -> str:
    """Generate a random bearer token for a review user."""
    return secrets.token_urlsafe(32)