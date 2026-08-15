from app import drift as d


def test_empty_sample_has_perfect_accuracy():
    assert d.compute_rolling_accuracy([]) == 1.0


def test_rolling_accuracy_matches_human_review():
    reviewed = [
        {"label": "positive", "corrected_label": "positive"},
        {"label": "negative", "corrected_label": "positive"},
        {"label": "neutral", "corrected_label": "neutral"},
        {"label": "positive", "corrected_label": "negative"},
    ]
    assert d.compute_rolling_accuracy(reviewed) == 0.5


def test_degradation_floor():
    assert d.check_for_degradation(0.79) is True
    assert d.check_for_degradation(0.80) is False


def test_psi_identical_distribution_is_zero():
    dist = {"negative": 10, "neutral": 20, "positive": 30}
    assert d.compute_label_psi(dist, dist) == 0.0


def test_psi_shift_exceeds_threshold():
    reference = {"negative": 100, "neutral": 0, "positive": 0}
    current = {"negative": 0, "neutral": 0, "positive": 100}
    assert d.compute_label_psi(reference, current) > d.PSI_THRESHOLD


def test_psi_missing_label_tolerated():
    reference = {"negative": 5, "positive": 5}
    current = {"negative": 3}
    assert d.compute_label_psi(reference, current) >= 0.0


def test_summarize_flags_both_signals():
    reviewed = [
        {"label": "positive", "corrected_label": "negative"},
        {"label": "positive", "corrected_label": "negative"},
        {"label": "positive", "corrected_label": "negative"},
        {"label": "positive", "corrected_label": "negative"},
    ]
    summary = d.summarize(reviewed, reference={"positive": 400, "negative": 0})
    assert summary["degraded"] != "none"
    assert "accuracy" in summary["degraded"]


def test_summarize_healthy():
    reviewed = [
        {"label": "positive", "corrected_label": "positive"},
        {"label": "negative", "corrected_label": "negative"},
    ]
    summary = d.summarize(reviewed, reference={"negative": 1, "positive": 1})
    assert summary["rolling_accuracy"] == 1.0
    assert summary["degraded"] == "none"