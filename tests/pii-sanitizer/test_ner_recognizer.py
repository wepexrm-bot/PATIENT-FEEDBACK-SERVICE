import pytest

presidio = pytest.importorskip("presidio_analyzer", reason="presidio not installed")

from app.ner_recognizer import _get_engines, apply_ner_redaction, ner_available


def test_ner_available():
    assert ner_available() is True


def test_apply_ner_redaction_returns_text_and_manifest():
    redacted, manifest = apply_ner_redaction("Dr. Alice Johnson saw me at Central Hospital.")
    assert isinstance(redacted, str)
    assert isinstance(manifest, list)
    # NER tags whatever PERSON/ORG/LOCATION entities it detects
    assert all({"type", "start", "end"}.issubset(m) for m in manifest)


def test_get_engines_initialises_analyzer_and_anonymizer():
    analyzer, anonymizer = _get_engines()
    assert analyzer is not None
    assert anonymizer is not None