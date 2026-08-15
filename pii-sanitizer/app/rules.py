"""Deterministic regex-based PII detectors: MRN, SSN, phone, email, DOB, ZIP+4, insurance ID."""
import re

PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "MRN": re.compile(r"\bMRN\s*:?\s*\d{6,10}\b", re.IGNORECASE),
    "ZIP4": re.compile(r"\b\d{5}-\d{4}\b"),
    "DOB": re.compile(
        r"\b(?:DOB|DATE ?OF ?BIRTH|BIRTH ?DATE|BORN)[:\s-]*"
        r"(\d{1,3}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        re.IGNORECASE,
    ),
    "INSURANCE_ID": re.compile(
        r"\b(?:INS(?:URANCE)?|MEMBER|POLICY)[\s-]*(?:ID|NO|NUMBER)"
        r"[\s:#-]*[A-Z0-9]{4,14}\b",
        re.IGNORECASE,
    ),
}


def apply_rule_based_redaction(text: str) -> tuple[str, list[dict]]:
    """Redact deterministic patterns and return a manifest of what was removed.

    The manifest records offsets in the *original* text for audit purposes;
    redaction is applied sequentially on the progressively redacted text.
    """
    manifest = []
    redacted = text
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            manifest.append({"type": label, "start": match.start(), "end": match.end()})
        redacted = pattern.sub(f"[{label}]", redacted)
    return redacted, manifest