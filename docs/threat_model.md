# Threat Model
## AI-Powered Cloud Security Threat Detection and Analytics System
### Wipro CPPE Project — Phase 5: Security Engineering

**Document status:** Project demonstration. Not a production security assessment.  
**Date:** 2026-10-13  
**Author:** Wipro CPPE Student Project

---

## 1. System Overview

The system runs locally (Windows workstation) and consists of:

| Component | Technology | Trust Boundary |
|---|---|---|
| Streamlit dashboard | Python / Streamlit | Local browser |
| ML predictor | Scikit-learn RandomForest | Local process |
| Data pipeline | Pandas + SQLite | Local filesystem |
| SQLite database | SQLite 3 | Local file |
| Saved model artefacts | Joblib .pkl files | Local filesystem |
| Synthetic dataset | CSV files | Local filesystem |
| AWS integration | **Not active — Phase 6** | N/A |

The application has **no authentication**, **no network exposure** in Phase 4,
and processes **entirely synthetic data**. All AWS references are planned only.

---

## 2. Assets and Value

| Asset | Sensitivity | Why it matters |
|---|---|---|
| `security_logs.db` | Medium | Contains all log records and incident data |
| `threat_classifier.pkl` | Medium | Trained model — could be swapped to manipulate predictions |
| `feature_engineer.pkl` | Low-Medium | Fitted encoders — must match model |
| `training_metadata.json` | Low | Config only |
| `reports/model_evaluation.txt` | Low | Evidence for CPPE report |
| Incident records (SQLite) | Medium | Contains analyst notes and triage decisions |
| Uploaded CSV files | Variable | User-controlled; must not be executed |

---

## 3. Trust Boundaries

```
┌─────────────────────────────────────────────────┐
│  LOCAL WORKSTATION (trusted)                    │
│                                                 │
│  ┌──────────────┐    ┌──────────────────────┐  │
│  │  Browser     │───▶│  Streamlit (app.py)  │  │
│  └──────────────┘    └──────┬───────────────┘  │
│                             │                  │
│              ┌──────────────┴───────┐          │
│              ▼                      ▼          │
│  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  SQLite DB       │  │  ML Predictor       │ │
│  │  (db_utils.py)   │  │  (predictor.py)     │ │
│  └──────────────────┘  └─────────────────────┘ │
│                                                 │
│  ═══ TRUST BOUNDARY ═══════════════════════════ │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  USER INPUT (untrusted)                  │  │
│  │  - CSV uploads                           │  │
│  │  - Incident form fields                  │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘

          ╔══════════════════════════╗
          ║  AWS (Phase 6 — future)  ║
          ║  NOT CONNECTED           ║
          ╚══════════════════════════╝
```

---

## 4. STRIDE Threat Analysis

### 4.1 Spoofing

| Threat | Attack scenario | Mitigation (implemented) | Mitigation (future) |
|---|---|---|---|
| S1 | Attacker replaces `threat_classifier.pkl` with a manipulated model to suppress alerts | Artefacts stored locally; OS file permissions | File hash verification on load |
| S2 | CSV upload contains records crafted to look like legitimate trusted log sources | Source IP and service validated by schema only | IP allowlist / digital signatures |

### 4.2 Tampering

| Threat | Attack scenario | Mitigation (implemented) | Mitigation (future) |
|---|---|---|---|
| T1 | Attacker modifies `security_logs.db` directly | SQLite file is local; parameterised queries prevent SQL injection | DB integrity checks; WAL mode |
| T2 | Malicious CSV injects SQL via column values | All incident writes use `?` parameterised queries; CSV values never placed in raw SQL | N/A — already mitigated |
| T3 | Uploaded CSV contains formula injection (`=cmd|'/C calc'!A0`) for spreadsheet tools | CSV values are never passed to a spreadsheet renderer in this application | Sanitise for export if Excel download added |
| T4 | Oversized CSV causes memory exhaustion | `MAX_ROWS = 5,000` hard limit enforced in `csv_validator.py` | Stream large files rather than loading all at once |

### 4.3 Repudiation

| Threat | Attack scenario | Mitigation (implemented) | Mitigation (future) |
|---|---|---|---|
| R1 | User denies creating or updating an incident | `created_at` / `updated_at` timestamps stored in SQLite | Append-only audit log; user identity per action |
| R2 | Pipeline run history lost | `pipeline_runs` table records every run with timestamp | Structured logging to file |

### 4.4 Information Disclosure

| Threat | Attack scenario | Mitigation (implemented) | Mitigation (future) |
|---|---|---|---|
| I1 | Database file read by another local user | OS filesystem permissions (not enforced by this app) | Encrypt DB at rest using SQLCipher or similar |
| I2 | Model `.pkl` file exposes training data structure | Feature list in metadata only; raw records not embedded in pkl | N/A for synthetic data; encrypt pkl for production |
| I3 | Stack traces shown in browser expose internal paths | Streamlit catches most exceptions; `try/except` in prediction path | Custom error handler; disable debug mode in production |
| I4 | Synthetic data mistaken for real AWS activity | Explicit synthetic-data banners on every dashboard page | Watermark data exports |

### 4.5 Denial of Service

| Threat | Attack scenario | Mitigation (implemented) | Mitigation (future) |
|---|---|---|---|
| D1 | Upload of a very large CSV freezes the dashboard | `MAX_ROWS = 5,000` rejects oversized files | File size check in bytes before `pd.read_csv` |
| D2 | Malformed CSV causes infinite loop in parser | `pd.read_csv` runs with default timeout behaviour; exception caught | Timeout wrapper around file parsing |
| D3 | Rapid repeat form submissions flood the incident table | Duplicate prevention via unique index on `(event_id, title)` | Rate limiting / CAPTCHA |

### 4.6 Elevation of Privilege

| Threat | Attack scenario | Mitigation (implemented) | Mitigation (future) |
|---|---|---|---|
| E1 | Dashboard user modifies `raw_logs` directly via browser | `raw_logs` has no update/delete path in the dashboard | Read-only SQLite connection for dashboard queries |
| E2 | Incident status set to an arbitrary string to bypass workflow | Status validated against strict allowlist (`VALID_STATUSES`) in `input_validator.py` | State machine enforcement with transition log |
| E3 | Incident ID manipulated to update another user's record | `validate_incident_id` ensures positive integer; parameterised UPDATE | Row-level ownership when multi-user added |

---

## 5. Attack Scenarios (Detailed)

### Scenario A: Malicious CSV Upload

**Goal:** Attacker uploads a CSV designed to inject SQL or crash the application.

**Attack steps:**
1. Craft a CSV where `event_name` column contains `'; DROP TABLE raw_logs; --`
2. Upload via the dashboard file uploader

**Result with current controls:**
- `csv_validator.validate_upload()` checks schema and value ranges.
- The value `'; DROP TABLE raw_logs; --` passes as a string (it is a valid event_name string).
- `prepare_for_prediction()` passes it to `predictor.predict()`.
- `FeatureEngineer` lowercases and encodes it — the string is treated as a category.
- **No SQL is constructed from this value** — all DB writes use `?` placeholders.
- Result: the attacker's SQL is stored as a harmless string, then predicted as low-risk.

**Residual risk:** None for SQL injection. Log entries with attacker-controlled strings are stored, which is expected behaviour.

---

### Scenario B: Oversized Upload (DoS)

**Goal:** Attacker uploads a 500,000-row CSV to exhaust memory.

**Current control:** `csv_validator.validate_upload()` checks `len(df) > MAX_ROWS` (5,000) and returns an error immediately.

**Gap:** The file is already loaded into memory by `pd.read_csv()` before the row-count check. A 500,000-row file (~50 MB) would be read entirely first.

**Recommended future fix:** Check file size in bytes before calling `pd.read_csv()`. Implemented as a test and documented; not implemented in code because this is a local demo application.

---

### Scenario C: Privilege Escalation via Incident Status

**Goal:** Attacker sets an incident's status to an arbitrary value like `"Approved"` to bypass the workflow.

**Current control:** `update_incident_status()` calls `validate_status()` which checks against `VALID_STATUSES = {"New","Investigating","Resolved","False Positive"}`. Any other value causes the function to return `False` without touching the database.

**Result:** The attacker cannot set an arbitrary status.

---

### Scenario D: Model Replacement Attack

**Goal:** Attacker swaps `models/threat_classifier.pkl` with a tampered model that always predicts `label=0` (nothing suspicious).

**Current gap:** The application loads the model with `joblib.load()` and does not verify a hash or signature.

**Recommended future fix:** Compute a SHA-256 hash of the pkl file at training time, store it in `training_metadata.json`, and verify it on every load. **Not yet implemented.**

---

## 6. Summary: Implemented vs. Planned Mitigations

| Control | Status |
|---|---|
| Parameterised SQL queries (no string interpolation) | ✅ Implemented |
| Input sanitisation (title, description, event_id) | ✅ Implemented |
| Status allowlist validation | ✅ Implemented |
| Incident ID type validation | ✅ Implemented |
| Incident duplicate prevention | ✅ Implemented |
| Row count limit on CSV uploads | ✅ Implemented |
| Schema validation on CSV uploads | ✅ Implemented |
| No code execution from uploaded content | ✅ Implemented |
| Synthetic data banners on every page | ✅ Implemented |
| No hardcoded AWS credentials | ✅ Implemented |
| Authentication / authorisation | ❌ Not implemented (single-user local app) |
| File size check before CSV parsing | ❌ Not implemented |
| Model hash verification on load | ❌ Not implemented |
| Database encryption at rest | ❌ Not implemented |
| Append-only audit log | ❌ Not implemented |
| Export sanitisation (formula injection) | ❌ Not implemented |

---

*This document is a student project demonstration. It does not constitute a professional security assessment.*
