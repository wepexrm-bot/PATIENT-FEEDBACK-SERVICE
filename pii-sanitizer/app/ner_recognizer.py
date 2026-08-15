"""NER-based PII detection via Microsoft Presidio (PERSON, LOCATION, ORG, DATE, etc.).

Presidio is loaded lazily so that services/tests that don't need it (e.g. pure
rule-based tests, local dev without the heavy dependency) can still import this
module. If Presidio is unavailable at call time, apply_ner_redaction raises
ImportError and callers should treat the NER pass as skipped.
"""
from __future__ import annotations

_ENGINES = None


def _get_engines():
    global _ENGINES
    if _ENGINES is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "presidio is not installed; the NER pass is unavailable"
            ) from exc
        _ENGINES = AnalyzerEngine(), AnonymizerEngine()
    return _ENGINES


def warmup() -> None:
    if ner_available():
        apply_ner_redaction("")


def apply_ner_redaction(text: str) -> tuple[str, list[dict]]:
    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(text=text, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    manifest = [
        {"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
        for r in results
    ]
    return anonymized.text, manifest


def ner_available() -> bool:
    try:
        _get_engines()
        return True
    except ImportError:
        return False