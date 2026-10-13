"""
predictor.py
============
Phase 3 — AI Engineering: inference on new records.

Loads the saved model and FeatureEngineer from models/ and scores
either a single dict record or a DataFrame of records.

SYNTHETIC DATA NOTICE
---------------------
This predictor was trained on synthetic data. Predictions on real
AWS CloudTrail records require Phase 6 integration and retraining.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline.cleaner import clean
from src.data_pipeline.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "threat_classifier.pkl"
FEATURE_ENGINEER_FILE = MODELS_DIR / "feature_engineer.pkl"

# Columns that must be present in a raw record for prediction
REQUIRED_RAW_COLUMNS = [
    "event_id", "event_time", "event_source", "event_name",
    "source_ip", "user_identity", "aws_region", "error_code",
    "failed_attempts", "unusual_location", "privilege_change",
    "label",            # required by FeatureEngineer._build(); set to 0 for inference
    "data_source",
    "ingested_at",
]


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_model(model_path: Path = MODEL_FILE):
    """Load the saved RandomForestClassifier."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run 'python -m src.ml.trainer' to train the model first."
        )
    clf = joblib.load(model_path)
    logger.info("[predictor] Loaded model from %s", model_path)
    return clf


def load_feature_engineer(fe_path: Path = FEATURE_ENGINEER_FILE):
    """Load the saved fitted FeatureEngineer."""
    if not fe_path.exists():
        raise FileNotFoundError(
            f"FeatureEngineer file not found: {fe_path}\n"
            "Run 'python -m src.ml.trainer' to train the model first."
        )
    fe = joblib.load(fe_path)
    logger.info("[predictor] Loaded FeatureEngineer from %s", fe_path)
    return fe


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _prepare_dataframe(record_or_df) -> pd.DataFrame:
    """
    Accept either a single dict (one record) or a DataFrame.
    Adds placeholder columns required by the pipeline if missing.
    """
    if isinstance(record_or_df, dict):
        df = pd.DataFrame([record_or_df])
    elif isinstance(record_or_df, pd.DataFrame):
        df = record_or_df.copy()
    else:
        raise TypeError(
            "Input must be a dict (single record) or a pandas DataFrame."
        )

    # Add placeholder columns that the pipeline needs but may not be
    # present in a minimal inference request
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if "label" not in df.columns:
        df["label"] = 0          # placeholder — not used in prediction
    if "data_source" not in df.columns:
        df["data_source"] = "inference"
    if "ingested_at" not in df.columns:
        df["ingested_at"] = now
    if "event_id" not in df.columns:
        import uuid
        df["event_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    return df


def predict(
    record_or_df,
    clf=None,
    fe=None,
) -> pd.DataFrame:
    """
    Predict threat label and probability for one or more records.

    Parameters
    ----------
    record_or_df : dict or pd.DataFrame
        A single CloudTrail-style record (dict) or a DataFrame of records.
        Required fields: event_time, event_source, event_name, source_ip,
        user_identity, aws_region, error_code, failed_attempts,
        unusual_location, privilege_change.
    clf : trained RandomForestClassifier or None
        If None, loads from models/threat_classifier.pkl.
    fe : fitted FeatureEngineer or None
        If None, loads from models/feature_engineer.pkl.

    Returns
    -------
    pd.DataFrame with columns:
        event_id, predicted_label, threat_probability,
        threat_level (Low / Medium / High)
    """
    if clf is None:
        clf = load_model()
    if fe is None:
        fe = load_feature_engineer()

    df = _prepare_dataframe(record_or_df)
    df_clean = clean(df)

    # Use transform() — do NOT re-fit encoders on new records
    X_raw, _ = fe.transform(df_clean)
    X = X_raw.drop(columns=["event_id"])

    # Ensure column order matches training
    X = X[FEATURE_COLUMNS]

    probs = clf.predict_proba(X)[:, 1]   # probability of class 1 (suspicious)
    preds = clf.predict(X)

    def _threat_level(p: float) -> str:
        if p >= 0.70:
            return "High"
        if p >= 0.40:
            return "Medium"
        return "Low"

    results = pd.DataFrame({
        "event_id": X_raw["event_id"].values,
        "predicted_label": preds,
        "threat_probability": probs.round(4),
        "threat_level": [_threat_level(p) for p in probs],
    })

    return results


def predict_single(record: dict, clf=None, fe=None) -> dict:
    """
    Convenience wrapper: predict for a single record dict.

    Returns a dict with predicted_label, threat_probability, threat_level.
    """
    result_df = predict(record, clf=clf, fe=fe)
    row = result_df.iloc[0]
    return {
        "predicted_label": int(row["predicted_label"]),
        "threat_probability": float(row["threat_probability"]),
        "threat_level": row["threat_level"],
    }


# ---------------------------------------------------------------------------
# Script entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Example normal record
    normal_record = {
        "event_time": "2024-06-15T10:23:00Z",
        "event_source": "s3.amazonaws.com",
        "event_name": "GetObject",
        "source_ip": "192.0.2.10",
        "user_identity": "alice@example-corp.com",
        "aws_region": "us-east-1",
        "error_code": "",
        "failed_attempts": 0,
        "unusual_location": 0,
        "privilege_change": 0,
    }

    # Example suspicious record
    suspicious_record = {
        "event_time": "2024-06-15T03:17:00Z",
        "event_source": "iam.amazonaws.com",
        "event_name": "AttachUserPolicy",
        "source_ip": "203.0.113.99",
        "user_identity": "root",
        "aws_region": "ap-northeast-1",
        "error_code": "AccessDenied",
        "failed_attempts": 8,
        "unusual_location": 1,
        "privilege_change": 1,
    }

    print("=== Predictor smoke test ===\n")
    for name, record in [("Normal record", normal_record),
                          ("Suspicious record", suspicious_record)]:
        result = predict_single(record)
        print(f"{name}:")
        print(f"  Predicted label   : {result['predicted_label']} "
              f"({'suspicious' if result['predicted_label'] else 'normal'})")
        print(f"  Threat probability: {result['threat_probability']:.4f}")
        print(f"  Threat level      : {result['threat_level']}")
        print()
