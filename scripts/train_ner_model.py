"""Train/fine-tune a spaCy NER model for clinical PII (PERSON, ORG, LOCATION, DATE).

The PII sanitizer uses the spaCy model through Presidio by default;
this script provides the optional fine-tuning path described in the architecture:

  python scripts/train_ner_model.py \
      --data data/raw/ner_annotations.jsonl \
      --out data/models/ner_model

JSONL format per line:
  {"text": "Dr. Jane Smith saw me.", "entities": [{"start": 4, "end": 14, "label": "PERSON"}]}

Requires: spacy (>= 3.5) and its training API. Safe to skip if you rely on the
default Presidio + built-in spaCy models.
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_annotations(path: Path) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # spaCy needs entities as a list of tuples for the updater
            docs.append(
                (obj["text"], {"entities": [(e["start"], e["end"], e["label"]) for e in obj["entities"]]})
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(REPO / "data" / "raw" / "ner_annotations.jsonl"))
    parser.add_argument("--out", default=str(REPO / "data" / "models" / "ner_model"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(
            f"[ERROR] {data_path} not found. Create annotations first or skip NER "
            "fine-tuning (Presidio ships working defaults)."
        )

    import spacy

    docs = load_annotations(data_path)
    print(f"Loaded {len(docs)} annotated docs")

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    labels = set()
    for _, ann in docs:
        for _, _, label in ann["entities"]:
            labels.add(label)
    for label in sorted(labels):
        ner.add_label(label)
    print(f"Labels: {sorted(labels)}")

    optimizer = nlp.begin_training()
    rng = random.Random(42)
    for itn in range(args.epochs):
        rng.shuffle(docs)
        losses = {}
        for text, ann in docs:
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(doc, ann)
            nlp.update([example], sgd=optimizer, drop=args.dropout, losses=losses)
        print(f"Epoch {itn + 1}/{args.epochs}: {losses}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out)
    print(f"Saved NER model -> {out.resolve()}")
    print("Wire it up by pointing Presidio's NlpEngine at this model dir.")


if __name__ == "__main__":
    main()