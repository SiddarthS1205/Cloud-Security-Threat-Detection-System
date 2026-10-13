"""
trainer.py
==========
Phase 3 — AI Engineering: model training pipeline.

What this module does
---------------------
1. Loads the cleaned raw_logs table from the SQLite database.
2. Performs a stratified 80/20 train/test split on the *raw* rows —
   before any feature encoding is fitted.
3. Calls FeatureEngineer.fit_transform() on the training split only,
   then FeatureEngineer.transform() on the test split.
4. Strips event_id and label from the input matrix X.
5. Trains a RandomForestClassifier(class_weight="balanced") on X_train.
6. Saves the trained model, the fitted FeatureEngineer, and training
   metadata to the models/ directory.
7. Returns the test-set predictions for evaluator.py to score.

Data-leakage prevention
-----------------------
- The train/test split happens on raw DataFrame rows, before FeatureEngineer
  sees any data at all.
- FeatureEngineer.fit_transform() fits LabelEncoders only on training rows.
- FeatureEngineer.transform() applies those fitted encoders to test rows
  without re-fitting.
- 'label' is separated inside FeatureEngineer._build() before any
  transformation, so it never influences encoder fitting.
- 'event_id' is a string UUID used only for traceability; it is dropped
  from X before sklearn receives it.

SYNTHETIC DATA NOTICE
---------------------
The dataset is entirely synthetic. Labels are assigned by scenario
membership, not by human analysts reviewing real cloud activity.
Evaluation metrics reflect how well the model learned the synthetic
patterns. They do NOT demonstrate real-world cloud security detection
performance. Phase 6 (AWS CloudTrail integration) is required before
any operational conclusions can be drawn.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data_pipeline.cleaner import clean
from src.data_pipeline.features import FeatureEngineer, FEATURE_COLUMNS
from src.data_pipeline.storage import get_connection, read_raw_logs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.20        # 80/20 split
MODELS_DIR = Path("models")

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Saved artefact filenames
MODEL_FILE = MODELS_DIR / "threat_classifier.pkl"
FEATURE_ENGINEER_FILE = MODELS_DIR / "feature_engineer.pkl"
METADATA_FILE = MODELS_DIR / "training_metadata.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_logs(db_path: str) -> pd.DataFrame:
    """
    Load the cleaned raw_logs table from SQLite.

    Parameters
    ----------
    db_path : str
        Path to security_logs.db.

    Returns
    -------
    pd.DataFrame
        All rows from raw_logs, including event_id, event_time,
        event_source, event_name, error_code, user_identity,
        aws_region, failed_attempts, unusual_location,
        privilege_change, label.
    """
    conn = get_connection(db_path)
    df = read_raw_logs(conn)
    conn.close()
    logger.info("[trainer] Loaded %d rows from raw_logs.", len(df))
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified 80/20 split on raw DataFrame rows.

    Stratification on 'label' preserves the 80/20 class balance
    in both halves.

    The split happens on raw rows — BEFORE FeatureEngineer sees any data.

    Parameters
    ----------
    df        : raw DataFrame from raw_logs
    test_size : fraction for the test set (default 0.20)
    seed      : random seed (default 42)

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "[trainer] Split: train=%d  test=%d  "
        "(train_susp=%d  test_susp=%d)",
        len(train_df), len(test_df),
        (train_df["label"] == 1).sum(),
        (test_df["label"] == 1).sum(),
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Feature engineering with leakage prevention
# ---------------------------------------------------------------------------

def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, FeatureEngineer]:
    """
    Fit FeatureEngineer on training data ONLY, then transform both splits.

    Steps
    -----
    1. Clean both splits (normalise timestamps, fill error_code, etc.).
    2. Call fe.fit_transform(train_df) — fits LabelEncoders on training rows.
    3. Call fe.transform(test_df)     — applies fitted encoders, no re-fitting.
    4. Strip event_id from both X matrices before returning.

    Returns
    -------
    X_train, y_train, X_test, y_test, fitted_fe
    """
    # Clean each split independently (no info crosses the boundary)
    train_clean = clean(train_df)
    test_clean = clean(test_df)

    fe = FeatureEngineer()
    X_train_raw, y_train = fe.fit_transform(train_clean)
    X_test_raw,  y_test  = fe.transform(test_clean)

    # Drop event_id — UUID string, never a model input
    X_train = X_train_raw.drop(columns=["event_id"])
    X_test  = X_test_raw.drop(columns=["event_id"])

    # Sanity check: label must not be in X
    assert "label" not in X_train.columns, "LEAKAGE: label found in X_train"
    assert "label" not in X_test.columns,  "LEAKAGE: label found in X_test"
    assert "event_id" not in X_train.columns, "event_id still in X_train"

    # Confirm feature columns match the canonical list
    missing = [c for c in FEATURE_COLUMNS if c not in X_train.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    logger.info(
        "[trainer] Features: train=%s  test=%s",
        X_train.shape, X_test.shape,
    )
    return X_train, y_train, X_test, y_test, fe


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the training feature matrix.

    Hyperparameters
    ---------------
    n_estimators    = 100  (100 decision trees in the forest)
    max_depth       = 10   (limits tree depth to reduce overfitting)
    min_samples_leaf = 5   (each leaf must cover at least 5 samples)
    class_weight    = "balanced"  (compensates for 80/20 class imbalance)
    random_state    = 42   (reproducible)

    Parameters
    ----------
    X_train : feature matrix (no event_id, no label)
    y_train : label series (0=normal, 1=suspicious)

    Returns
    -------
    Fitted RandomForestClassifier
    """
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    logger.info(
        "[trainer] Model trained on %d samples, %d features.",
        len(X_train), X_train.shape[1],
    )
    return clf


# ---------------------------------------------------------------------------
# Saving artefacts
# ---------------------------------------------------------------------------

def save_artefacts(
    clf: RandomForestClassifier,
    fe: FeatureEngineer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save the trained model, fitted FeatureEngineer, and metadata to models/.

    Files written
    -------------
    models/threat_classifier.pkl   — trained RandomForestClassifier
    models/feature_engineer.pkl    — fitted FeatureEngineer (with encoders)
    models/training_metadata.json  — split sizes, feature list, timestamp
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_FILE)
    joblib.dump(fe, FEATURE_ENGINEER_FILE)

    metadata = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_suspicious": int((train_df["label"] == 1).sum()),
        "test_suspicious": int((test_df["label"] == 1).sum()),
        "feature_columns": FEATURE_COLUMNS,
        "model_params": RF_PARAMS,
        "data_source": "synthetic",
        "synthetic_data_notice": (
            "This model was trained on entirely synthetic data. "
            "Evaluation metrics do NOT demonstrate real-world "
            "cloud security detection performance."
        ),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("[trainer] Artefacts saved to %s/", MODELS_DIR)
    print(f"[trainer] Saved: {MODEL_FILE}")
    print(f"[trainer] Saved: {FEATURE_ENGINEER_FILE}")
    print(f"[trainer] Saved: {METADATA_FILE}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_training(
    db_path: str = os.path.join("data", "database", "security_logs.db"),
) -> tuple[RandomForestClassifier, FeatureEngineer,
           pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Full training pipeline: load → split → features → train → save.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database (default: data/database/security_logs.db).

    Returns
    -------
    clf, fe, X_train, y_train, X_test, y_test
        The trained classifier, fitted FeatureEngineer, and both splits.
        X_test / y_test are returned so evaluator.py can score them without
        loading the database a second time.
    """
    print("\n" + "=" * 60)
    print("PHASE 3 — Model Training (RandomForestClassifier)")
    print("SYNTHETIC DATA — results are not real-world performance")
    print("=" * 60 + "\n")

    # Step 1: Load
    df = load_raw_logs(db_path)

    # Step 2: Split (before any preprocessing)
    train_df, test_df = split_data(df)
    print(f"[trainer] Train rows: {len(train_df):,}  "
          f"(suspicious: {(train_df['label']==1).sum():,})")
    print(f"[trainer] Test  rows: {len(test_df):,}  "
          f"(suspicious: {(test_df['label']==1).sum():,})")

    # Step 3: Feature engineering — fit on train only
    X_train, y_train, X_test, y_test, fe = build_train_test_features(
        train_df, test_df
    )
    print(f"[trainer] Feature matrix: train={X_train.shape}  "
          f"test={X_test.shape}")
    print(f"[trainer] Features used: {list(X_train.columns)}")

    # Step 4: Train
    clf = train_model(X_train, y_train)

    # Step 5: Save
    save_artefacts(clf, fe, train_df, test_df)

    return clf, fe, X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_training()
