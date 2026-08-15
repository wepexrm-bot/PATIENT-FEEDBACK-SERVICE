"""Seed the governance Postgres with sample predictions + review queue for demo.

Only hashed patient references and sanitized/redacted text are written — never
raw identifiers. Uses the same models as governance-service.

Usage:
  python scripts/seed_db.py [--url postgresql://user:pass@host:port/db]
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "governance-service"))

from app.db import SessionLocal, engine
from app.models import Base, Prediction, ReviewQueueItem

SAMPLE = [
    ("Feeling rushed but overall acceptable.", "neutral", 0.62),
    ("The nurse was extraordinary and kind.", "positive", 0.97),
    ("Billing mistakes again, very frustrated.", "negative", 0.88),
    ("It was okay, not good, not bad.", "neutral", 0.55),
    ("Doctor explained everything thoroughly.", "positive", 0.74),
    ("Long wait times ruined the visit.", "negative", 0.91),
    ("Facility was clean and quiet.", "positive", 0.7),
    ("They lost my paperwork, unacceptable.", "negative", 0.83),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="SQLAlchemy DB URL (defaults to env DATABASE_URL / postgres vars)")
    args = parser.parse_args()

    if args.url:
        import os

        os.environ["DATABASE_URL"] = args.url

    Base.metadata.create_all(bind=engine)

    rng = random.Random(11)
    queued = 0
    with SessionLocal() as session:
        for i, (text, label, conf) in enumerate(SAMPLE):
            pred = Prediction(
                patient_ref=hashlib.sha256(f"opaque-seed-{i}".encode()).hexdigest(),
                label=label,
                confidence=conf,
                model_version="v1.0.0",
                input_hash=hashlib.sha256(text.encode()).hexdigest(),
                latency_ms=round(rng.uniform(5, 40), 2),
                redacted_text=text,
                manifest_hash=None,
            )
            session.add(pred)
            session.flush()
            if conf < 0.75:
                session.add(ReviewQueueItem(prediction_id=pred.id))
                queued += 1
        session.commit()

    print(f"Seeded {len(SAMPLE)} predictions ({queued} queued for review).")


if __name__ == "__main__":
    main()