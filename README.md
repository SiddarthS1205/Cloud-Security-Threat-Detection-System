# AI-Powered Cloud Security Threat Detection and Analytics System

**Wipro CPPE College Project**

A machine learning system that ingests cloud activity logs, detects suspicious
behaviour, and presents findings on an interactive security dashboard.

---

## CPPE Module Coverage

| Module | Features Built |
|---|---|
| AI Engineering | ML classifier, threat scoring, prediction dashboard |
| Data Engineering | Synthetic data generation, cleaning pipeline, SQLite storage |
| Cloud Engineering | AWS S3/CloudTrail/CloudWatch/KMS integration (Phase 6) |
| Security Engineering | Threat model, incident response workflow, input validation |

---

## Technology Stack

- **Python 3.10+**
- **Pandas / NumPy** — data pipeline
- **Scikit-learn** — machine learning
- **SQLite** — local storage (no database server needed)
- **Streamlit** — web dashboard
- **Plotly** — interactive charts
- **Faker** — synthetic log generation
- **Boto3** — AWS SDK (Phase 6 only)

---

## Quick Start (Local — No AWS Required)

### 1. Clone / open the project folder
```
cd d:\Wipro_CPPE_Cloud_Security_AI
```

### 2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Verify the installation
```
python -c "import pandas, sklearn, streamlit, plotly, faker; print('All packages OK')"
```

---

## Project Phases

| Phase | Description | AWS Required? |
|---|---|---|
| 1 | Project foundation — folders, dependencies | No |
| 2 | Data pipeline — generate, clean, store logs | No |
| 3 | Machine learning — train and evaluate classifier | No |
| 4 | Dashboard — predictions, alerts, charts | No |
| 5 | Security — threat model, incident response | No |
| 6 | AWS integration — real CloudTrail logs | Yes |

---

## Folder Structure

```
Wipro_CPPE_Cloud_Security_AI/
├── data/
│   ├── raw/          Synthetic CSV logs (input)
│   ├── processed/    Cleaned and feature-engineered data
│   └── database/     SQLite database file
├── src/
│   ├── data_pipeline/ Ingestion, validation, cleaning, features, storage
│   ├── ml/            Training, evaluation, prediction
│   ├── security/      Threat model, incident response, controls
│   └── cloud/         AWS connectors (Phase 6 only)
├── dashboard/         Streamlit app
├── models/            Saved trained model (.pkl)
├── tests/             Unit and integration tests
├── docs/              Architecture and AWS setup guides
├── requirements.txt
└── README.md
```

---

## Important Notes

- **No AWS credentials are ever hardcoded** in this project.
- AWS integration is isolated in `src/cloud/` and does not affect local operation.
- Accuracy metrics reported in documentation are taken from actual test runs.
- See `docs/aws_setup_guide.md` before starting Phase 6.

---

## Running Tests

```
pytest tests/ -v
```

## Running the Dashboard

```
streamlit run dashboard/app.py
```
*(Available from Phase 4 onwards)*
