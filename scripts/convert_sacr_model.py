"""Convert a trained SACR model directory into the sentiment-model weights bundle.

Produces `sentiment-model/weights/{model.joblib, meta.json}` so the running
service can serve the model WITHOUT re-training and without depending on the
SACR project at runtime.

How it works:
  1. Load SACR's artifacts (`best_pipeline.joblib`, `meta.json`).
  2. Rebind the vectorizer's custom `LemmaTokenizer` to the copy bundled inside
     `sentiment-model/app/sacr_compat.py` so the re-dumped pipeline unpickles
     without the SACR checkout present (only weights/vocab are reused).
  3. Write the rebased pipeline + a `meta.json` (ordered labels, model version).

Usage:
  python scripts/convert_sacr_model.py \
      --model-dir "D:\\ML web app\\sacr_model" \
      --sacr-src "D:\\ML web app" \
      --version v1.0.0
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_app_sacr_compat():
    """Load sentiment-model's `app` package + its `sacr_compat` module.

    Registers them under the real package name `app.sacr_compat` so pickled
    classes reference the same module path the running container imports
    (model_loader does `from app import sacr_compat`).
    """
    app_dir = REPO / "sentiment-model" / "app"
    for name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "app", app_dir / "__init__.py", submodule_search_locations=[str(app_dir)]
    )
    app_pkg = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app_pkg
    spec.loader.exec_module(app_pkg)
    compat_spec = importlib.util.spec_from_file_location(
        "app.sacr_compat", app_dir / "sacr_compat.py"
    )
    compat = importlib.util.module_from_spec(compat_spec)
    sys.modules["app.sacr_compat"] = compat
    compat_spec.loader.exec_module(compat)
    return compat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="SACR output dir (best_pipeline.joblib, meta.json)")
    parser.add_argument("--sacr-src", help="Path to the SACR checkout to make sacr_cli importable at load time")
    parser.add_argument("--weights-out", default=str(REPO / "sentiment-model" / "weights"))
    parser.add_argument("--version", default="v1.0.0", help="model_version written to meta.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "meta.json").exists() or not (model_dir / "best_pipeline.joblib").exists():
        sys.exit(f"[ERROR] {model_dir} is missing SACR artifacts (meta.json, best_pipeline.joblib)")

    # SACR's pickled pipeline references `sacr_cli.LemmaTokenizer`; we need the
    # module importable once, during conversion, before rebinding below.
    if args.sacr_src:
        sys.path.insert(0, str(Path(args.sacr_src).resolve()))

    import joblib

    # SACR's best_pipeline.joblib is created by running sacr_cli.py as __main__,
    # so the pickled tokenizer is referenced as __main__.LemmaTokenizer.
    # Register it so unpickling resolves before we rebind below.
    if args.sacr_src:
        import sacr_cli
        sys.modules["__main__"].LemmaTokenizer = sacr_cli.LemmaTokenizer

    compat = _load_app_sacr_compat()

    with open(model_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = list(meta.get("class_names", []))
    if not class_names:
        sys.exit("[ERROR] SACR meta.json has no class_names")

    pipe = joblib.load(model_dir / "best_pipeline.joblib")
    if not hasattr(pipe, "named_steps") or "vect" not in pipe.named_steps:
        sys.exit("[ERROR] best_pipeline.joblib is not a sklearn Pipeline with a 'vect' step")

    vect = pipe.named_steps["vect"]
    vect.tokenizer = compat.LemmaTokenizer()
    if hasattr(vect, "_validate_vocabulary"):
        pass  # vocabulary preserved; re-validation happens on transform, not here

    out_dir = Path(args.weights_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib", compress=3)

    out_meta = {
        "model_version": args.version,
        "labels": class_names,
        "backend": "joblib",
        "source": "sacr",
        "n_classes": len(class_names),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)

    print(f"Wrote {out_dir / 'model.joblib'}")
    print(f"Wrote {out_dir / 'meta.json'}")
    print(f"Labels: {class_names}")
    print("To 3-class (negative/neutral/positive): train SACR on a 3-level/rating column.")


if __name__ == "__main__":
    main()