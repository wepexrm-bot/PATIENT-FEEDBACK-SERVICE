import app.threshold as threshold_mod
import pytest
from app.config import CONFIDENCE_THRESHOLD
from app.threshold import needs_review

EMPTY_LABEL_THRESHOLDS = {}


@pytest.fixture()
def no_label_thresholds(monkeypatch):
    monkeypatch.setattr(threshold_mod, "LABEL_CONFIDENCE_THRESHOLDS", EMPTY_LABEL_THRESHOLDS)


@pytest.fixture()
def per_label_thresholds(monkeypatch):
    monkeypatch.setattr(
        threshold_mod,
        "LABEL_CONFIDENCE_THRESHOLDS",
        {"negative": 0.7, "neutral": 0.8, "positive": 0.7},
    )


def test_needs_review_below_global_threshold(no_label_thresholds):
    assert needs_review("positive", CONFIDENCE_THRESHOLD - 0.01) is True


def test_no_review_at_or_above_global_threshold(no_label_thresholds):
    assert needs_review("negative", CONFIDENCE_THRESHOLD) is False
    assert needs_review("positive", 1.0) is False


def test_extreme_low_confidence_flagged(no_label_thresholds):
    assert needs_review("neutral", 0.0) is True


def test_unlisted_label_falls_back_to_global(no_label_thresholds):
    assert needs_review("positive", 0.76) is False  # global bar is 0.75
    assert needs_review("positive", 0.74) is True


def test_label_specific_threshold_overrides_global(per_label_thresholds):
    # neutral's bar is raised to 0.8, so 0.75 is now flagged for review
    assert needs_review("neutral", 0.75) is True
    assert needs_review("neutral", 0.8) is False
    # negative's bar is lowered to 0.7, so 0.74 is accepted
    assert needs_review("negative", 0.74) is False
    assert needs_review("negative", 0.69) is True


def test_thresholds_are_additive_across_labels(per_label_thresholds):
    # same confidence, different bars
    assert needs_review("neutral", 0.75) is True
    assert needs_review("positive", 0.75) is False