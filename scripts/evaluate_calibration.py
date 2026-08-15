"""Evaluate calibration of the served sentiment model on a true held-out set.

Loads the baked weights model (`sentiment-model/weights/model.joblib`) and scores
it against the UCI Drugs.com TEST split, reporting:

  - accuracy
  - Brier score (multiclass)
  - ECE (expected calibration error, top-label, 10 buckets)
  - reliability table (predicted-confidence bucket vs observed accuracy)

Ratings are binned exactly as in training: 1-4 -> negative, 5-6 -> neutral,
7-10 -> positive. No files are written.

Usage:
  python scripts/evaluate_calibration.py
  python scripts/evaluate_calibration.py --test-zip "path/to/drugsComTest_raw.csv.zip"
"""
import argparse
import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parent.parent

RATING_BINS = {
    (1, 4): "negative",
    (5, 6): "neutral",
    (7, 10): "positive",
}


def _load_app_sacr_compat():
    """Register sentiment-model's `app` + `app.sacr_compat` (for unpickling)."""
    app_dir = REPO / "sentiment-model" / "app"
    for name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "app", app_dir / "__init__.py", submodule_search_locations=[str(app_dir)]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["app"] = pkg
    spec.loader.exec_module(pkg)
    compat_spec = importlib.util.spec_from_file_location(
        "app.sacr_compat", app_dir / "sacr_compat.py"
    )
    compat = importlib.util.module_from_spec(compat_spec)
    sys.modules["app.sacr_compat"] = compat
    compat_spec.loader.exec_module(compat)
    return compat


def _read_test_rows(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(member) as fh:
            text = fh.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rating_col = next(
        (c for c in ("rating", "overall-ratings") if c in (reader.fieldnames or [])),
        None,
    )
    if rating_col is None:
        raise ValueError(f"no rating column found in {zip_path}")

    rows = []
    for rec in reader:
        review = (rec.get("review") or "").strip()
        rating_raw = (rec.get(rating_col) or "").strip()
        try:
            rating = int(float(rating_raw))
        except ValueError:
            continue
        if review and 1 <= rating <= 10:
            rows.append((review, rating))
    return rows


def bin_rating(rating: int) -> str:
    for (lo, hi), label in RATING_BINS.items():
        if lo <= rating <= hi:
            return label
    raise ValueError(f"rating {rating} out of range")


def reliability_bins(confidences, correct, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == n_bins - 1 else (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        if n == 0:
            table.append((lo, hi, 0, np.nan, np.nan, np.nan))
            continue
        avg_conf = float(confidences[mask].mean())
        acc = float(correct[mask].mean())
        table.append((lo, hi, n, avg_conf, acc, abs(avg_conf - acc)))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=str(REPO / "sentiment-model" / "weights" / "model.joblib"),
        help="baked model.joblib to evaluate",
    )
    parser.add_argument(
        "--test-zip",
        default="d:/research/dataset/drugsComTest_raw.csv.zip",
        help="UCI Drugs.com test split zip",
    )
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--ece-threshold", type=float, default=0.05)
    args = parser.parse_args()

    compat = _load_app_sacr_compat()
    _ = compat  # module must be importable for joblib unpickling

    meta_path = Path(args.weights).with_name("meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels = list(meta["labels"])
    print(f"model: {Path(args.weights)} | labels: {labels} | backend: {meta.get('backend')}")

    pipe = joblib.load(args.weights)

    print(f"loading test rows from {args.test_zip} ...")
    rows = _read_test_rows(Path(args.test_zip))
    print(f"parsed {len(rows)} test rows")

    texts = [r[0] for r in rows]
    y_true = np.array([bin_rating(r[1]) for r in rows])

    probs = pipe.predict_proba(texts)
    pred_idx = probs.argmax(axis=1)
    y_pred = np.array([labels[i] for i in pred_idx])
    conf = probs.max(axis=1)

    acc = float((y_pred == y_true).mean())

    y_onehot = np.zeros(probs.shape)
    for i, lab in enumerate(labels):
        y_onehot[:, i] = (y_true == lab).astype(float)
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    correct = y_pred == y_true
    table = reliability_bins(conf, correct, args.ece_bins)
    ece = float(sum(n * gap for _, _, n, _, _, gap in table if n > 0)) / len(conf)

    print(f"\nreliability (top-label, {args.ece_bins} buckets):")
    print(f"{'bucket':>14} {'n':>7} {'avg_conf':>9} {'accuracy':>9} {'gap':>7}")
    for lo, hi, n, avg_conf, acc_b, gap in table:
        if n == 0:
            print(f"[{lo:.2f},{hi:.2f}) {n:>7} {'-':>9} {'-':>9} {'-':>7}")
        else:
            print(f"[{lo:.2f},{hi:.2f}) {n:>7} {avg_conf:9.4f} {acc_b:9.4f} {gap:7.4f}")

    verdict = "PASS (already calibrated)" if ece <= args.ece_threshold else "FAIL (recalibration needed)"
    print("\n--- summary ---")
    print(f"test rows   : {len(texts)}")
    print(f"accuracy    : {acc:.4f}")
    print(f"brier       : {brier:.4f}  (perfect = 0.0)")
    print(f"ECE         : {ece:.4f}  (threshold {args.ece_threshold:.2f}) -> {verdict}")


if __name__ == "__main__":
    main()