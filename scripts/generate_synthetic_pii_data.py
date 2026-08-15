"""Generate synthetic patient feedback data with embedded PII for sanitizer tests.

Writes `data/raw/synthetic_feedback.csv` with columns:
  patient_ref, source, text, rating

Texts embed deterministic patterns (names, phones, emails, SSNs, MRNs, DOBs,
ZIP+4, insurance ids) so the PII sanitizer's rule + NER passes can be validated.
Keep the PII markers strictly synthetic so the file is safe to version/commit.
"""
import argparse
import csv
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FIRST = ["Jane", "John", "Maria", "Ahmed", "Grace", "Omar", "Priya", "Leo"]
LAST = ["Doe", "Smith", "Garcia", "Ali", "Lee", "Kim", "Shah", "Berg"]
FEEDBACK = [
    ("The entire staff was fantastic and the treatment was exactly on time.", 9),
    ("I called {phone} but waited forever at reception, very disappointed.", 2),
    ("Doctor {name} listened carefully. Refer me to {email} for follow-up.", 8),
    ("Billing was a nightmare. My SSN {ssn} was on a visible form.", 1),
    ("Clean facility, nurse {name} explained everything about my MRN {mrn}.", 7),
    ("The room was cold and noisy; please fix before my next DOB {dob} visit.", 3),
    ("Reached support at {phone}; my insurance id {ins} got charged wrong.", 2),
    ("Postal code {zip} service center was closed; wasted a trip.", 3),
    ("Great experience overall, {name} and team went above and beyond.", 10),
    ("Average care; nothing wrong but nothing memorable either.", 5),
]


def _rand_phone(rng: random.Random) -> str:
    return f"{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def _rand_ssn(rng: random.Random) -> str:
    return f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


def _rand_mrn(rng: random.Random) -> str:
    return f"MRN {rng.randint(10_000_000, 99_999_999)}"


def _rand_dob(rng: random.Random) -> str:
    return f"DOB {rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(1940, 2015)}"


def _rand_ins(rng: random.Random) -> str:
    letters = "ABCDEFGHJKMNP"
    return f"Insurance ID: {rng.choice(letters)}{rng.randint(0, 9)}{rng.randint(10_000, 99_999)}"


def _rand_email(name: str) -> str:
    return f"{name.lower().replace(' ', '.')}@example.com"


def build_rows(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        text_tpl, rating = rng.choice(FEEDBACK)
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        vars_ = {
            "name": name,
            "phone": _rand_phone(rng),
            "ssn": _rand_ssn(rng),
            "mrn": _rand_mrn(rng),
            "dob": _rand_dob(rng),
            "ins": _rand_ins(rng),
            "zip": f"{rng.randint(10000, 99999)}-{rng.randint(1000, 9999)}",
            "email": _rand_email(name),
        }
        text = text_tpl.format(**vars_)
        rows.append(
            {
                "patient_ref": f"opaque-{i:06d}",
                "source": rng.choice(["portal", "sms", "kiosk"]),
                "text": text,
                "rating": rating,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(REPO / "data" / "raw" / "synthetic_feedback.csv"))
    args = parser.parse_args()

    rows = build_rows(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_ref", "source", "text", "rating"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out.resolve()}")
    print("Run the SACR tool on this file with a rating column to train a 3-class model:")
    print(f'  sacr_cli train {out} --text-col text --label-col rating --out my_model --include-neutral')


if __name__ == "__main__":
    main()