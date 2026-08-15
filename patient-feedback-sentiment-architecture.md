# Patient Feedback Sentiment Analysis Microservice — Architecture Design

## 1. Overview

A fully local, containerized system that ingests free-text patient feedback, strips PII before any persistence or model inference, classifies sentiment via a transformer model, and governs prediction quality through confidence-based human-in-the-loop (HITL) review. Designed for HIPAA-sensitive environments — no raw PHI ever leaves the edge sanitization boundary.

**Core design principle:** PII redaction happens *before* logging, queuing, or model inference — not after. Nothing downstream of the sanitizer should ever see raw identifiers.

---

## 2. High-Level Flow

```
Patient Feedback (text)
        │
        ▼
[1] Ingestion API (FastAPI)
        │
        ▼
[2] PII Sanitization Service  ──► Audit store (hashed mapping only, optional)
        │  (redacted text only passes forward)
        ▼
[3] Sentiment Inference Service (HuggingFace transformer)
        │
        ▼
[4] Governance Layer
        ├─ Prediction + confidence logged
        ├─ Confidence < threshold? ──► HITL Review Queue
        └─ Rolling drift/quality metrics
        │
        ▼
[5] Storage (Postgres) + Review UI/API
```

Each numbered stage is a separate container, communicating over an internal Docker network only — no stage has external network access except [1].

---

## 3. Component Breakdown

### 3.1 Ingestion API (`ingestion-api`)
- **Stack:** FastAPI + Pydantic
- **Responsibility:** Accepts feedback via REST (`POST /feedback`) or reads from a local queue (Redis/RabbitMQ) for batch ingestion.
- **Contract:**
  ```json
  POST /feedback
  { "patient_ref": "opaque-token", "text": "raw feedback text", "source": "portal|sms|kiosk" }
  ```
- Immediately forwards to the sanitizer — **never persists raw text itself.**

### 3.2 PII Sanitization Service (`pii-sanitizer`)
- **Stack:** Microsoft Presidio (analyzer + anonymizer) or spaCy NER fine-tuned for clinical text, layered with regex rules.
- **Two-pass approach:**
  1. **Rule-based pass** — deterministic patterns: MRN, SSN, phone, email, DOB, ZIP+4, insurance ID.
  2. **NER pass** — transformer/spaCy model tags PERSON, ORG (facility names), LOCATION, DATE in free text.
- **Output:** redacted text (e.g., `[PATIENT_NAME]`, `[PHONE]`) + a redaction manifest (what was removed, offsets, type) for audit purposes — manifest stored separately from redacted text, encrypted, with restricted access.
- **Important:** This service should be swappable/versioned independently, since PII detection quality is the highest-risk failure mode in the whole system (false negatives leak PHI).

### 3.3 Sentiment Inference Service (`sentiment-model`)
- **Stack:** HuggingFace Transformers, e.g. `distilbert-base-uncased-finetuned-sst-2-english` as a baseline, or a domain-fine-tuned model on clinical/patient-satisfaction corpora for better accuracy.
- **Serving:** exported to ONNX + `onnxruntime` for fast local CPU inference (avoids GPU dependency for a local/offline deployment), wrapped in a small FastAPI inference server.
- **Output:** `{ "label": "positive|neutral|negative", "confidence": 0.0-1.0, "model_version": "v1.2.0" }`
- Only ever receives sanitized text.

### 3.4 Governance / MLOps Layer (`governance-service`)
This is the differentiator — treat it as a first-class service, not a logging afterthought.

- **Prediction logging:** every inference call logs `{input_hash, model_version, label, confidence, timestamp, latency_ms}` to Postgres. Store a hash of the sanitized text, not the text itself, unless retention policy explicitly allows it.
- **Confidence thresholding:** configurable threshold (e.g., 0.75). Below it → record inserted into `review_queue` table with status `pending`.
- **HITL review workflow:**
  - Reviewers see redacted text + model prediction + confidence via a review endpoint/UI.
  - Reviewer submits corrected label → stored as ground truth.
  - Corrected labels feed a rolling accuracy metric (`model accuracy vs. human-reviewed sample`).
- **Drift/degradation detection:**
  - Rolling window (e.g., last 500 human-reviewed predictions) tracks accuracy, confidence distribution, and label distribution shift (e.g., population stability index).
  - If accuracy drops below a set floor or confidence distribution shifts significantly → alert/flag `model_version` for retraining review. This is what prevents *silent* degradation — the system actively watches its own reliability rather than assuming a deployed model stays good forever.

### 3.5 Storage (`postgres`)
- Tables: `predictions`, `review_queue`, `redaction_audit` (access-restricted), `model_versions`, `drift_metrics`.
- PHI-adjacent tables (`redaction_audit`) should live in a separate schema/DB instance with tighter access control than the rest.

---

## 4. Deployment (Docker Compose)

```yaml
services:
  ingestion-api:      # FastAPI, exposes :8000
  pii-sanitizer:       # internal only
  sentiment-model:     # internal only
  governance-service:  # internal + review API :8010
  postgres:             # internal only, volumes for persistence
  redis:                # optional, for async queueing
```

- Only `ingestion-api` and `governance-service` (review endpoints) are exposed on the host network.
- All inter-service traffic on an internal Docker bridge network.
- No outbound internet access needed at runtime once model weights are baked into the image — good for HIPAA-sensitive, air-gapped deployments.

---

## 5. Security & Compliance Notes

- **PHI minimization:** raw text exists only transiently in `ingestion-api` memory before sanitization; never written to disk unredacted.
- **Encryption at rest** for `redaction_audit` and any patient-ref mapping tables.
- **Access control:** review UI/API should require authenticated, role-based access (reviewers vs admins).
- **Model versioning:** every prediction tagged with `model_version` so you can trace which model produced a flagged/degraded result.
- **Audit trail:** governance layer doubles as your compliance audit log — who reviewed what, when, and what changed.

---

## 6. Suggested Build Order

1. Ingestion API + PII sanitizer (get redaction correctness solid first — this is the highest-stakes component)
2. Sentiment inference service (can start with off-the-shelf model, swap in fine-tuned later)
3. Governance layer: logging → threshold routing → review queue
4. Drift/degradation metrics + alerting
5. Docker Compose wiring + network isolation
