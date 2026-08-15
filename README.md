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
