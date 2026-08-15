"""Rolling accuracy + label-distribution shift tracking against human-reviewed
ground truth. Flags a model_version for retraining review when the accuracy
floor is breached or the label distribution drifts (population stability index).
"""
import math
from collections import Counter

ACCURACY_FLOOR = 0.80
PSI_THRESHOLD = 0.25
_EPSILON = 1e-6


def compute_rolling_accuracy(reviewed_predictions: list[dict]) -> float:
    """Fraction of human-reviewed predictions where label == corrected_label."""
    if not reviewed_predictions:
        return 1.0
    correct = sum(1 for p in reviewed_predictions if p["label"] == p["corrected_label"])
    return correct / len(reviewed_predictions)


def check_for_degradation(rolling_accuracy: float) -> bool:
    return rolling_accuracy < ACCURACY_FLOOR


def compute_label_psi(reference: dict, current: dict) -> float:
    """Population stability index between two label distributions.

    Both inputs are {label: count}. Uses every observed label, with a small
    epsilon so zero buckets stay finite.
    """
    labels = sorted(set(reference) | set(current))
    ref_total = float(sum(reference.values())) or 1.0
    cur_total = float(sum(current.values())) or 1.0
    psi = 0.0
    for label in labels:
        expected = (reference.get(label, 0) / ref_total) or _EPSILON
        actual = (current.get(label, 0) / cur_total) or _EPSILON
        psi += (actual - expected) * math.log(actual / expected)
    return psi


def check_for_psi_drift(psi: float) -> bool:
    return psi > PSI_THRESHOLD


def summarize(reviewed_predictions: list[dict], reference: dict | None = None) -> dict:
    """Assess a reviewed sample: accuracy, label shift, and degradation flags."""
    accuracy = compute_rolling_accuracy(reviewed_predictions)
    corrected_dist = dict(Counter(p["corrected_label"] for p in reviewed_predictions))
    ref_dist = reference if reference is not None else corrected_dist
    psi = compute_label_psi(ref_dist, corrected_dist)

    degraded = []
    if check_for_degradation(accuracy):
        degraded.append("accuracy")
    if check_for_psi_drift(psi):
        degraded.append("psi")

    return {
        "rolling_accuracy": accuracy,
        "label_psi": psi,
        "window_size": len(reviewed_predictions),
        "degraded": ",".join(degraded) if degraded else "none",
        "current_distribution": corrected_dist,
    }