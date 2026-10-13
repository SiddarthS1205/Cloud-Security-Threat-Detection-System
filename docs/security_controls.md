# Security Controls
## AI-Powered Cloud Security Threat Detection and Analytics System
### Wipro CPPE Project — Phase 5: Security Engineering

---

## 1. Implemented Controls

### 1.1 Input Sanitisation (`src/security/input_validator.py`)

All user-supplied text that reaches the database is processed through
`input_validator.py` before any SQL operation.

| Input | Control | Max Length |
|---|---|---|
| Incident title | `sanitise_title()` — strip + truncate, reject empty | 200 chars |
| Incident description | `sanitise_description()` — strip + truncate | 2,000 chars |
| Event ID reference | `sanitise_event_id()` — strip + truncate, default "manual" | 128 chars |
| Severity | `validate_severity()` — allowlist: Low / Medium / High / Critical | N/A |
| Status | `validate_status()` — allowlist: New / Investigating / Resolved / False Positive | N/A |
| Incident ID | `validate_incident_id()` — must be a positive integer | N/A |
| SQL LIMIT params | `clamp_limit()` — clamped to [1, 500] | N/A |

### 1.2 Parameterised SQL Queries (`dashboard/db_utils.py`)

All database write operations use `?` placeholder syntax.
No user-supplied value is ever placed directly into a SQL string via
string concatenation or f-strings.

Previously, `get_top_suspicious_users()` and `get_recent_events()`
used f-strings to interpolate `n` (an internal integer) into LIMIT
clauses. These have been changed to use `?` parameters with clamping
via `clamp_limit()`.

### 1.3 CSV Upload Validation (`dashboard/csv_validator.py`)

| Check | Action on failure |
|---|---|
| Empty file | Error — reject |
| > 5,000 rows | Error — reject |
| Missing required columns | Error — reject |
| Unparseable event_time | Error — reject |
| Non-numeric integer columns | Error — reject |
| Integer values out of range | Error — reject |
| Unknown error_code values | Warning — accept with notice |
| Empty rows | Warning — rows skipped silently |

Uploaded CSV content is **never executed**. Values are read as strings,
validated, type-cast, and passed through the existing feature engineering
pipeline as data only.

### 1.4 Incident Duplicate Prevention (`dashboard/db_utils.py`)

A unique index on `(event_id, title)` prevents duplicate incident rows
at the database level. `create_incident()` also checks for an existing
row before inserting, and returns `already_existed=True` if found.
The dashboard displays a warning to the user rather than creating a duplicate.

### 1.5 Synthetic Data Labelling

Labels in the dataset are assigned by scenario membership in the generator,
not by individual feature values. This is documented in `generator.py` and
explicitly stated in the dashboard, model evaluation report, and this document.

### 1.6 No Hardcoded Credentials

No AWS access keys, secret keys, passwords, or API tokens appear anywhere
in the codebase. The `src/cloud/` package is empty and unused in Phase 4/5.
Phase 6 will use environment variables or AWS IAM roles — never hardcoded keys.

### 1.7 Audit Trail

The `pipeline_runs` table records every data pipeline execution with a
timestamp, row counts, and notes. The `incidents` table records
`created_at` and `updated_at` for every incident.

---

## 2. Known Limitations

These are genuine gaps. They are documented here honestly.

### 2.1 No Authentication or Authorisation

The dashboard has no login system. Any user who can open the URL can
create, update, or view all incidents. This is acceptable for a local
single-user college project but is **not acceptable for a shared or
production environment**.

**Future fix:** Add Streamlit authentication (e.g. `streamlit-authenticator`
library) or deploy behind an identity-aware proxy.

### 2.2 File Size Not Checked Before Parsing

`csv_validator.validate_upload()` checks the row count **after**
`pd.read_csv()` has already loaded the file into memory. A maliciously
crafted 500 MB CSV would be fully loaded before being rejected.

**Future fix:** Check `uploaded_file.size` in bytes before calling
`pd.read_csv()`. Example threshold: 10 MB.

### 2.3 No Model Integrity Verification

`predictor.load_model()` calls `joblib.load()` without verifying a
hash or digital signature. A tampered `threat_classifier.pkl` would
be loaded and used silently.

**Future fix:** Compute SHA-256 of the pkl file at training time,
store in `training_metadata.json`, and verify on every load.

### 2.4 No Database Encryption

`security_logs.db` is a plain SQLite file. Any user with filesystem
access can read all log records and incident data.

**Future fix:** Use SQLCipher for encrypted SQLite, or store sensitive
fields encrypted at the application layer.

### 2.5 Stack Traces May Leak Internal Paths

In development mode, Streamlit can display Python stack traces in the
browser. These may reveal absolute file paths or internal module structure.

**Future fix:** Wrap all prediction and database calls in `try/except`
with user-friendly error messages and log full traces server-side only.
Already partially implemented in `app.py`'s prediction section.

### 2.6 CSV Formula Injection (Export Risk)

If the predictions download (CSV export) is ever opened in Microsoft Excel,
cell values beginning with `=`, `+`, `-`, or `@` may be interpreted as
formulas. The current export does not sanitise for this.

**Future fix:** Prefix dangerous values with a single quote or use the
`xlsxwriter` library with cell format protection if Excel export is added.

### 2.7 Synthetic Data Performance Does Not Reflect Reality

The ML model achieves ~98% accuracy on the synthetic test set. This does
not mean it will perform similarly on real AWS CloudTrail logs. The
synthetic patterns are far more regular than real-world cloud activity.

---

## 3. Security Test Coverage

Security-specific tests are in `tests/test_security.py` and cover:

| Test area | File |
|---|---|
| Input sanitisation (title, description, event_id) | `test_security.py` |
| Severity and status allowlist validation | `test_security.py` |
| Incident ID validation (integer, positive) | `test_security.py` |
| SQL meta-character detection | `test_security.py` |
| Limit clamping | `test_security.py` |
| Malformed CSV rejection | `test_security.py` |
| Oversized CSV rejection | `test_security.py` |
| CSV with embedded SQL patterns | `test_security.py` |
| Duplicate incident prevention | `tests/test_dashboard.py` |
| Parameterised query verification | `tests/test_dashboard.py` |

---

## 4. Recommended Future Improvements (Priority Order)

1. **Add file-size check before CSV parsing** (Medium effort, high value)
2. **Add model hash verification on load** (Low effort, medium value)
3. **Add authentication to the dashboard** (High effort, high value)
4. **Add a structured audit log to file** (Medium effort, medium value)
5. **Encrypt the SQLite database** (High effort, medium value)
6. **Add rate limiting to incident creation** (Medium effort, low value for local app)

---

*This document covers Phase 5 only. AWS-specific controls will be added in Phase 6.*
