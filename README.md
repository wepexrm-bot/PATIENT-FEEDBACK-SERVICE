# Patient Feedback Sentiment Analysis Microservice

Local, containerized system that ingests patient feedback, sanitizes PII at the
edge, runs transformer-based sentiment analysis, and governs prediction quality
via a human-in-the-loop review layer.

See `docs/architecture.md` for full design.

## Services
- `ingestion-api/`      — receives feedback, forwards to sanitizer
- `pii-sanitizer/`      — rule-based + NER PII redaction
- `sentiment-model/`    — transformer sentiment inference (ONNX-served)
- `governance-service/` — prediction logging, confidence routing, HITL review, drift tracking
- `shared/`             — cross-service utilities (logging, security)
- `scripts/`            — training and data-prep scripts
- `data/`               — raw / processed / model artifacts (gitignored)
- `tests/`              — per-service test suites

## Quickstart
```bash
cp .env.example .env
docker compose up --build
```

## Model calibration gate
After any retrain or weight re-conversion, verify the served model's
probabilities are well-calibrated before rebuilding `sentiment-model`:

```bat
scripts\check_calibration.bat
```

It scores the model against the held-out UCI Drugs.com test split and exits
non-zero (FAIL) when ECE > 0.05. Direct equivalent:
`python scripts\evaluate_calibration.py --ece-threshold 0.05`

## Web UI (Streamlit)
Browser console for submitting feedback and running the HITL review loop.
Runs on the host against the Docker stack's exposed ports (8000 / 8010):

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Three tabs: **Analyze Feedback** (submit a comment, see sentiment + redaction
+ review flag), **Review Queue** (accept/correct queued predictions), and
**Drift** (compute/view model drift metrics). Review actions use the
`REVIEW_API_KEYS` token and role header set in the sidebar.
