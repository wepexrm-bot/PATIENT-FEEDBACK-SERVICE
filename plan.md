Final Plan — Patient Feedback Sentiment Microservice (3-class, Option A)
Model integration (core of your question)
- Training stays in SACR at D:\ML web app — you run sacr_cli train on a 3-class/rating-labeled CSV (SACR maps 3 levels or a 1–10 scale → negative/neutral/positive), producing best_pipeline.joblib etc.
- New scripts/convert_sacr_model.py --model-dir <sacr_out> --version v1.0.0:
1. Loads best_pipeline.joblib + vectorizer.joblib + label_encoder.joblib + meta.json
2. Rebinds the custom LemmaTokenizer/cleaning logic onto a compatibility module shipped inside this repo (sentiment-model/app/sacr_compat.py) so unpickling works without D:\ML web app being present — only trained weights/vocab are reused
3. Writes sentiment-model/weights/{model.joblib, meta.json} (class_names + model_version)
- Result: you never train a second model in this project.
- sentiment-model runtime: joblib + scikit-learn + nltk (baked into image, offline). Inference = pipeline.predict_proba() → label + confidence. Calibrated probs (SACR uses CalibratedClassifierCV) → sound input for the 0.75 threshold.
- Auto-detect backend: model.onnx → onnx, model.joblib → joblib (ONNX path stays available for later).
The rest of the build (unchanged, full architecture parity)
Phase 1 — Ingestion API: Pydantic validation (source ∈ {portal,sms,kiosk}, non-empty text), response schema, httpx timeouts/retry, structured logging via shared/. Tests (mocked httpx).
Phase 2 — PII Sanitizer: add DOB + INSURANCE_ID rules; manifest.py real → POST to governance /redaction-audit (X-Api-Key, hashed patient_ref); sanitized text + hash flow to sentiment. Tests per rule; NER via importorskip("presidio_analyzer").
Phase 3 — Sentiment Model: inference.py/model_loader.py dual-backend, labels from meta.json, latency_ms + input_hash (sha256 of sanitized text) captured and forwarded to governance; drop transformers; Dockerfile bakes weights + NLTK data (punkt_tab, stopwords, wordnet, averaged_perceptron_tagger_eng); compose removes empty bind-mount.
Phase 4 — Governance: Predictions += input_hash/latency_ms/redacted_text; ReviewQueueItem FK; new RedactionAudit in restricted redaction schema; /redaction-audit gated by API key; review API returns redacted text + prediction + confidence, requires token + role (reviewer/admin from shared/security.py); /drift/{model_version} computes rolling accuracy + PSI label/confidence shift over DRIFT_WINDOW_SIZE, writes DriftMetric, flags below ACCURACY_FLOOR. Tests: threshold, drift, review API (SQLite), tests/conftest.py.
Phase 5 — Scripts: generate_synthetic_pii_data.py → feedback CSV with embedded PII and a rating column for SACR training; convert_sacr_model.py (above); train_ner_model.py stub; seed_db.py demo rows (hash-only).
Phase 6 — Compose/env: healthchecks, only :8000/:8010 exposed, internal network, .env.example += MODEL_VERSION, REVIEW_API_KEYS, MANIFEST_API_KEY.
Verification (this machine, no Docker): pytest green locally; convert_sacr_model.py → confirm model.joblib + meta.json; you run docker compose up --build after installing Docker.
Last loose end — 3-class data source for SACR (your current sacr_model is
# Questions
Option A reuses SACR for training, but your existing model is 2-class. To get 3-class (negative/neutral/positive), what data will you feed into `sacr_cli train`?
I'll provide a real CSV
+ Thought: 1.3s
All decisions locked. Final summary:
Confirmed plan
- Training: you run SACR on your real CSV (sacr_cli train ... --label-col rating) → 3-class model output.
- scripts/convert_sacr_model.py repackages that output (rebased tokenizer, meta.json with class_names + model_version) into sentiment-model/weights/. No second model trained here.
- sentiment-model serves joblib via sklearn+NLTK (auto-detect model.onnx/model.joblib), emits label/confidence/model_version/latency_ms/input_hash.
- PII sanitizer (rules + DOB/insurance + real manifest audit POST, hashed) → governance (hash-only predictions, redacted-text review queue, token+role auth, /redaction-audit, /drift/{version} with PSI + accuracy floor) → Postgres.
- Support: generate_synthetic_pii_data.py (PII-only, for sanitizer testing), train_ner_model.py stub, seed_db.py, all 6 test suites + conftest.py, compose with healthchecks + baked weights, .env.example with API keys.
- Verify now (no Docker): pytest local; later: you run docker compose up --build after installing Docker.