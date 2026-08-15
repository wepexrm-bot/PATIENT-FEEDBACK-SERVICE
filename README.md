# Patient Feedback Sentiment Analysis Microservice

Containerized system that ingests patient feedback, sanitizes PII at the edge,
runs sentiment analysis, and governs prediction quality through a
human-in-the-loop review layer with drift tracking. Designed so **raw PII
never leaves the sanitizer** and **no raw identifier is ever persisted**.

```
                       ┌────────────────────────────┐
                       │   Streamlit Console (:8501) │   host-run UI
                       │   Analyze / Review / Drift  │   (ui/app.py)
                       └──────────────┬──────────────┘
                                      │ HTTP
┌─────────────────┐   ┌───────────────▼──────────────┐   ┌───────────────────────────┐
│  ingestion-api  │──▶│      pii-sanitizer           │──▶│      sentiment-model       │
│    :8000        │   │      :8001  (internal)       │   │      :8002  (internal)     │
│    (public)     │   │  rule regex + Presidio NER   │   │  data_cleaning + tokenizer │
│  POST /feedback │   │  redacts before forwarding   │   │  model.joblib / model.onnx │
└─────────────────┘   └──────────────┬───────────────┘   └──────────────┬────────────┘
                                     │ manifest (audit)                  │ prediction
                                     ▼                                   ▼
                     ┌──────────────────────────────────────────────────────────────┐
                     │                  governance-service (:8010, internal)         │
                     │   log-prediction  ·  review-queue  ·  drift  ·  redaction-audit│
                     │   review routing: low-confidence / sampled / domain-shift      │
                     └──────────────────────────────┬────────────────────────────────┘
                                                    ▼
                                    ┌────────────────────────────┐
                                    │   postgres (:5432, internal)│
                                    │   predictions · review_queue│
                                    │   drift_metrics · redaction │
                                    └────────────────────────────┘
```

## Request lifecycle

A single `POST /feedback` triggers the whole chain, and the redacted text is
the only thing that reaches the model:

1. **ingestion-api** validates the payload (`patient_ref`, `text`, `source`)
   and forwards the raw text to the sanitizer. It never persists or logs the
   raw text — only a masked 8-char prefix of the patient reference.
2. **pii-sanitizer** runs two passes and returns only the redacted text:
   - **Pass 1 – deterministic rules:** `SSN`, `PHONE`, `EMAIL`, `MRN`, `ZIP4`,
     `DOB`, `INSURANCE_ID` → replaced with `[TYPE]` placeholders.
   - **Pass 2 – Presidio NER:** `PERSON`, `DATE_TIME`, `LOCATION`, `ORG`, … →
     replaced with `<TYPE>` placeholders. If Presidio is unavailable or fails,
     the pass is skipped and rule-based redaction still applies.
   - A **redaction manifest** (what was removed + offsets + scores) is forwarded
     to governance's access-restricted `redaction_audit` store, keyed by a hash
     of the patient reference. This step is **non-critical**: audit failures are
     logged and swallowed so the prediction still proceeds.
3. **sentiment-model** runs the model on the redacted text only, then logs the
   prediction to governance and returns the enriched result.
4. **governance-service** persists the prediction (hashed patient reference,
   never raw), applies review routing, and the response flows back to the
   caller.

The caller receives: `label`, `confidence`, `model_version`, `redacted_text`,
`latency_ms`, `oov_score`, `flagged_for_review`, `review_reason`.

## Services

| Service | Port | Access | Role |
|---|---|---|---|
| `ingestion-api` | 8000 | host/public | Entry point; forwards raw text to sanitizer |
| `pii-sanitizer` | 8001 | internal only | Rule + NER redaction; manifest audit |
| `sentiment-model` | 8002 | internal only | Cleaning + prediction on redacted text |
| `governance-service` | 8010 | internal only | Persistence, review routing, drift, audit |
| `postgres` | 5432 | internal only | State store (never exposed to host) |
| Streamlit console | 8501 | host | `ui/app.py` — submit, review, drift |

Internal-only services are `expose`d on the compose network, not published to
the host — the only ways in are `ingestion-api:8000` and `governance:8010`
(plus the UI on the host).

## Repository layout

```
ingestion-api/        FastAPI entry point (public)
pii-sanitizer/        redaction: rules.py, ner_recognizer.py, manifest.py
sentiment-model/      model_loader, inference, cleaning (sacr_compat.py), weights/
governance-service/   log-prediction, review-queue, drift, redaction-audit
shared/               logging (JSON, PII-safe), security (tokens, hashing)
ui/                   Streamlit console (host-run)
scripts/              convert_sacr_model, evaluate_calibration, gate, seed, train
data/                 raw / processed / model artifacts (gitignored)
tests/                per-service suites
```

## Key design decisions

- **PII at the edge.** Raw patient text exists only inside the sanitizer's
  request scope. Downstream services, storage, and logs see redacted text and
  hashes.
- **Hashed identifiers.** `patient_ref` is SHA-256 hashed before persistence
  (`hash_patient_ref`); raw tokens are never stored or logged.
- **Confidence routing (`needs_review`).** Per-label acceptance bars from
  `LABEL_CONFIDENCE_THRESHOLDS` (unlisted labels fall back to
  `CONFIDENCE_THRESHOLD`).
- **Sampled review.** Every Nth prediction (`SAMPLE_REVIEW_EVERY_N`) is queued
  regardless of confidence — this catches confident-but-wrong predictions and
  keeps the drift window unbiased.
- **Domain-shift guardrail.** `oov_score` = fraction of the model's lemmas
  missing from its vocabulary. When it reaches `OOV_RATIO_THRESHOLD`, the
  prediction is queued as `domain-shift` even at high confidence — because an
  out-of-domain model is confidently wrong on text it cannot recognize.
- **Calibration gate.** `scripts/evaluate_calibration.py` scores the served
  weights against a held-out split (ECE/Brier/accuracy). The build must PASS
  (ECE ≤ 0.05) before a model is deployed; `scripts/check_calibration.bat`
  wraps it.
- **Drift.** Reviewed predictions feed rolling accuracy (floor 0.80) and label
  PSI (threshold 0.25); `degraded` reports `accuracy` / `psi` / `none`.

## Edge cases handled

**PII / privacy**
- Multi-line and comma-bearing reviews parsed correctly (CSV parser).
- Phone/date/digit tokens are stripped during text cleaning (`isalpha`), so
  digits never leak into model features.
- Placeholder tokens from redaction (`PERSON`, `PHONE`, …) are handled by the
  text cleaning before the vectorizer.
- `patient_ref` masked to 8 chars in logs; only hashes persisted.
- Redaction audit lives in its own restricted Postgres schema (`redaction`);
  on SQLite (tests) the schema qualifier is dropped.
- Audit store guarded by `MANIFEST_API_KEY` (`X-Api-Key`, constant-time).

**Resilience / failure paths**
- Sanitizer unreachable → ingestion returns `503` and logs; raw text never
  buffered.
- Governance logging fails → sentiment still returns the prediction (warning
  logged); the review/routing layer is best-effort on the critical path.
- Redaction manifest audit fails → logged and swallowed, prediction proceeds.
- NER unavailable/fails → rule-based redaction still applies; NER warmup
  failure never breaks startup.
- NER first-call cold start (≈19 s with `en_core_web_lg`) mitigated by engine
  caching + lifespan warmup + generous `INGESTION_HTTP_TIMEOUT=60`.

**Model / prediction**
- Backend auto-detected: `model.onnx` (ONNX Runtime) or `model.joblib`
  (sklearn pipeline); missing model → clear `FileNotFoundError`.
- `predict_proba` required; labels come from `meta.json` (ordered).
- Text cleaning mirrors the SACR training pipeline exactly: contraction
  expansion, non-word stripping, stopword removal (keeping `not`), `not_`
  negation prefixing, `LemmaTokenizer` lemmatization.
- Vocabulary capped by training `max_features`; words outside it contribute no
  signal — this is the OOV signal the guardrail measures.
- High-confidence-but-out-of-domain predictions are still queued by the OOV
  guardrail or sampling (defense in depth).

**Review / governance**
- Per-label thresholds fall back to the global `CONFIDENCE_THRESHOLD`.
- Review reasons combine with `+` (e.g. `low-confidence+sampled+domain-shift`).
- Review submission is idempotency-safe: a reviewed item returns `409`.
- Drift: empty reviewed sample → accuracy 1.0; zero-count PSI buckets handled
  with epsilon; window ordered by `reviewed_at`.

**Validation**
- Blank text rejected (`422`); `source` restricted to `portal|sms|kiosk`;
  `confidence` bounded to `[0, 1]`.

**Ops / state**
- Postgres kept off the host; data in a named volume.
- Schema created idempotently (`create_all` + `ALTER TABLE ... IF NOT EXISTS`
  migration for added columns like `oov_score`).
- All services expose `/health` and run compose healthchecks with
  `depends_on: condition: service_healthy` ordering.
- Structured JSON logging with a fixed PII-safe field allowlist.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_*` | `governance`/`changeme` | Postgres credentials, host, DB |
| `CONFIDENCE_THRESHOLD` | `0.75` | Global confidence bar |
| `LABEL_CONFIDENCE_THRESHOLDS` | `{}` | Per-label bars (JSON, single line) |
| `SAMPLE_REVIEW_EVERY_N` | `0` | Queue every Nth prediction (0 = off) |
| `OOV_RATIO_THRESHOLD` | `0.2` | Domain-shift review bar (0 = off) |
| `DRIFT_WINDOW_SIZE` | `500` | Reviewed-prediction drift window |
| `MODEL_VERSION` | `v1.0.0` | Fallback version (meta.json wins) |
| `MODEL_PATH` | `/app/weights` | Weights dir in the sentiment container |
| `REVIEW_API_KEYS` | — | Comma-separated bearer tokens for review API |
| `MANIFEST_API_KEY` | — | Secret for redaction-audit writes |
| `INGESTION_HTTP_TIMEOUT` | `10` | Sanitizer call timeout (s) |

Review endpoints authenticate with `Authorization: Bearer <token>` plus an
`X-Reviewer-Role` header (`reviewer` or `admin`); tokens are checked in
constant time. If no keys are configured the token check is skipped (local
dev convenience only — **always set real keys in production**).

## Quickstart

```bash
cp .env.example .env
docker compose up --build
```

- Feedback API: `http://localhost:8000/feedback`
- Governance API: `http://localhost:8010`

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"patient_ref":"pt-001","source":"portal",
       "text":"The nurse Dr. Smith was rude. Call 555-123-4567"}'
```

## Web UI (Streamlit)

```bash
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Three tabs — **Analyze Feedback** (submit, see sentiment + redaction + review
flag + OOV score), **Review Queue** (accept/correct queued predictions with
token + role from the sidebar), **Drift** (compute/view metrics per version).

## Model pipeline

1. Train with SACR → artifacts (`best_pipeline.joblib`, `meta.json`).
2. Convert for the service (rebinds the tokenizer to the bundled compat
   module so it unpickles without the SACR checkout):
   ```
   python scripts/convert_sacr_model.py --model-dir <sacr_out> --version v1.1.0
   ```
3. Gate it — must PASS before deploying:
   ```
   scripts\check_calibration.bat
   ```
4. Rebuild the sentiment image (`docker compose up -d --build sentiment-model`).

## Testing

```bash
python -m pytest tests -q
# 58 passed, 1 skipped
ruff check .
```

## Deployment

Targets a single always-on VM running the compose stack, with Caddy for HTTPS
(Let's Encrypt) and a DuckDNS subdomain as the free front door. The OOV
guardrail ships as a mandatory layer because the model is trained on drug
reviews, not hospital feedback — out-of-domain comments are forced to human
review rather than trusted at face value.