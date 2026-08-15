"""Stores redaction manifests separately from redacted text for audit purposes.

Manifests are forwarded to the governance service's access-restricted
`redaction_audit` store. Only a hash of the patient reference is persisted —
never the raw token — and raw identifiers never leave the sanitizer.
"""
import hashlib
import json

import httpx

from app.config import GOVERNANCE_URL, MANIFEST_TIMEOUT
from shared.logging_config import configure_logging
from shared.security import get_manifest_key, hash_patient_ref

logger = configure_logging("pii-sanitizer")


def manifest_hash(manifest: list[dict]) -> str:
    """Deterministic hash of a manifest, used to link audit and prediction records."""
    serialized = json.dumps(manifest, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


async def store_manifest(patient_ref: str, manifest: list[dict]) -> bool:
    """Forward a redaction manifest to the restricted audit store.

    Audit persistence must not break the critical path — failures are logged
    and swallowed so a sanitized prediction still proceeds.
    """
    if not manifest:
        return False
    payload = {
        "patient_ref_hash": hash_patient_ref(patient_ref),
        "manifest": manifest,
        "manifest_hash": manifest_hash(manifest),
    }
    headers = {}
    api_key = get_manifest_key()
    if api_key:
        headers["X-Api-Key"] = api_key
    try:
        async with httpx.AsyncClient(timeout=MANIFEST_TIMEOUT) as client:
            resp = await client.post(
                f"{GOVERNANCE_URL}/redaction-audit", json=payload, headers=headers
            )
            resp.raise_for_status()
        return True
    except (httpx.RequestError, httpx.HTTPStatusError):
        logger.warning("failed to store redaction manifest for audit")
        return False