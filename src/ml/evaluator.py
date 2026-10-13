"""
evaluator.py
============
Phase 3 — AI Engineering: model evaluation.

Computes and reports evaluation metrics on HELD-OUT TEST SET predictions.

Metrics reported
----------------
- Accuracy
- Precision (macro and per-class)
- Recall    (macro and per-class)
- F1-score  (macro, weighted, and per-class)
- Confusion matrix  (TP, FP, FN, TN explained in plain English)
- Full sklearn classification report
- Feature importances (top-12 ranked)
- Single-feature F1 comparison (documents synthetic-data artefacts)

SYNTHETIC DATA DISCLOSURE
--------------------------
All metrics are computed on a synthetic test set whose labels were
assigned by deterministic rules, not by human security analysts.
Even after noise injection (v2), some features remain stronger
predictors than they would be in real AWS CloudTrail data.
This section of the report documents those limitations explicitly
so no reader mistakes synthetic accuracy for operational performance.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

REPORTS_DIR = Path("reports")
EVALUATION_FILE = REPORTS_DIR / "model_evaluation.txt"


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute all evaluation metrics from true and predicted labels.

    Parameters
    ----------
    y_true : ground-truth labels (test set only)
    y_pred : model predictions on the test set

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    classification_report
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred,
                                                  average="macro",
                                                  zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred,
                                            average="macro",
                                            zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred,
                                    average="macro",
                                    zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred,
                                       average="weighted",
                                       zero_division=0)),
        "f1_suspicious": float(f1_score(y_true, y_pred,
                                         pos_label=1,
                                         zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["normal (0)", "suspicious (1)"],
            zero_division=0,
        ),
    }


def feature_importances(
    clf: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 12,
) -> list[tuple[str, float]]:
    """Return feature importances sorted descending."""
    pairs = sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    return pairs[:top_n]


def single_feature_f1(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[tuple[str, float]]:
    """
    Measure what each single feature can achieve with a depth-1 decision tree.

    This exposes synthetic-data artefacts: features that are near-perfect
    predictors on their own indicate the labelling rules are visible in
    the feature space.

    Encoders are fitted on X_train to avoid leakage even in this diagnostic.
    """
    results = []
    for col in X_test.columns:
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        dt.fit(X_train[[col]], y_train)
        f1 = f1_score(y_test, dt.predict(X_test[[col]]),
                      zero_division=0)
        results.append((col, float(f1)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    metrics: dict,
    importances: list[tuple[str, float]],
    sf1: list[tuple[str, float]],
    train_rows: int,
    test_rows: int,
    train_suspicious: int,
    test_suspicious: int,
    output_path: Path = EVALUATION_FILE,
) -> None:
    """
    Write all evaluation results to a plain-text report file.

    Parameters
    ----------
    metrics         : dict from compute_metrics()
    importances     : list from feature_importances()
    sf1             : list from single_feature_f1()
    train_rows      : number of training rows
    test_rows       : number of test rows
    train_suspicious: suspicious count in training set
    test_suspicious : suspicious count in test set
    output_path     : where to write the report
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "=" * 70,
        "AI-POWERED CLOUD SECURITY THREAT DETECTION",
        "Model Evaluation Report — Phase 3",
        f"Generated: {now}",
        "=" * 70,
        "",
        "SYNTHETIC DATA NOTICE",
        "-" * 70,
        "This model was trained and evaluated on a SYNTHETIC dataset.",
        "Labels were assigned by a deterministic rule engine, not by",
        "human analysts reviewing real AWS CloudTrail activity.",
        "Performance metrics reflect how well the model learned the",
        "synthetic patterns. They do NOT establish real-world cloud",
        "security detection performance.",
        "Evaluation on real AWS CloudTrail logs (Phase 6) is required",
        "before any operational conclusions can be drawn.",
        "",
        "DATASET SUMMARY",
        "-" * 70,
        f"  Training rows : {train_rows:,}  "
        f"(normal: {train_rows - train_suspicious:,}  "
        f"suspicious: {train_suspicious:,})",
        f"  Test rows     : {test_rows:,}  "
        f"(normal: {test_rows - test_suspicious:,}  "
        f"suspicious: {test_suspicious:,})",
        f"  Split         : 80% train / 20% test, stratified, seed=42",
        f"  Data source   : synthetic (v2 noise-injected, seed=42)",
        "",
        "MODEL",
        "-" * 70,
        "  Algorithm     : RandomForestClassifier",
        "  n_estimators  : 100",
        "  max_depth     : 10",
        "  min_samples_leaf : 5",
        "  class_weight  : balanced",
        "  random_state  : 42",
        "",
        "LEAKAGE PREVENTION",
        "-" * 70,
        "  1. Train/test split performed on raw rows BEFORE feature engineering.",
        "  2. LabelEncoders fitted on training rows only; applied to test rows.",
        "  3. 'label' separated from features before any transformation.",
        "  4. 'event_id' (UUID) excluded from model inputs.",
        "",
        "EVALUATION METRICS (held-out test set)",
        "-" * 70,
        f"  Accuracy          : {metrics['accuracy']:.4f}",
        f"  Precision (macro) : {metrics['precision_macro']:.4f}",
        f"  Recall    (macro) : {metrics['recall_macro']:.4f}",
        f"  F1        (macro) : {metrics['f1_macro']:.4f}",
        f"  F1     (weighted) : {metrics['f1_weighted']:.4f}",
        f"  F1    (suspicious): {metrics['f1_suspicious']:.4f}  "
        f"<-- primary metric (imbalanced class)",
        "",
        "CONFUSION MATRIX",
        "-" * 70,
        "  Rows = actual class  |  Columns = predicted class",
        "",
        "                   Predicted Normal  Predicted Suspicious",
        f"  Actual Normal       {tn:^16d}  {fp:^20d}",
        f"  Actual Suspicious   {fn:^16d}  {tp:^20d}",
        "",
        f"  True Positives  (TP): {tp:4d}  — suspicious caught correctly",
        f"  True Negatives  (TN): {tn:4d}  — normal flagged correctly",
        f"  False Positives (FP): {fp:4d}  — normal wrongly flagged as suspicious",
        f"  False Negatives (FN): {fn:4d}  — suspicious missed by the model",
        "",
        "CLASSIFICATION REPORT",
        "-" * 70,
        metrics["classification_report"],
        "",
        "FEATURE IMPORTANCES (Random Forest — mean decrease in impurity)",
        "-" * 70,
    ]

    for rank, (fname, imp) in enumerate(importances, 1):
        bar = "#" * int(imp * 60)
        lines.append(f"  {rank:2d}. {fname:25s}  {imp:.4f}  {bar}")

    lines += [
        "",
        "SINGLE-FEATURE F1 ANALYSIS (synthetic-data artefact check)",
        "-" * 70,
        "  A depth-1 decision tree is trained on each feature alone.",
        "  High F1 for a single feature indicates the labelling rule",
        "  is visible in that feature — a known limitation of synthetic data.",
        "  In real CloudTrail data, these scores would be much lower.",
        "",
    ]

    for fname, f1 in sf1:
        bar = "#" * int(f1 * 40)
        flag = "  <-- SYNTHETIC ARTEFACT" if f1 > 0.70 else ""
        lines.append(f"  {fname:25s}  F1={f1:.3f}  {bar}{flag}")

    lines += [
        "",
        "KNOWN LIMITATIONS OF THIS EVALUATION",
        "-" * 70,
        "  1. SYNTHETIC DATA: Labels were assigned by rule engine, not analysts.",
        "     High accuracy reflects synthetic pattern learning, not real detection.",
        "",
        "  2. NOISE INJECTION (v2): Normal records now occasionally have errors,",
        "     unusual-location flags, privilege changes, and off-hours timestamps.",
        "     This reduces single-feature predictive power but the dataset is",
        "     still far more structured than real AWS CloudTrail logs.",
        "",
        "  3. SMALL DATASET: 5,000 records is sufficient for a student project",
        "     but too small for production threat detection systems.",
        "",
        "  4. NO REAL VALIDATION: The model has not been tested on any real",
        "     AWS CloudTrail data. Phase 6 integration is required for that.",
        "",
        "  5. CLASS IMBALANCE: 80/20 normal/suspicious. class_weight='balanced'",
        "     compensates during training, but real environments may have",
        "     far fewer than 20% suspicious events.",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[evaluator] Report written to {output_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_and_report(
    clf: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_rows: int,
    test_rows: int,
    train_suspicious: int,
    test_suspicious: int,
    output_path: Path = EVALUATION_FILE,
) -> dict:
    """
    Evaluate the classifier on the test set and write the full report.

    Parameters
    ----------
    clf             : fitted RandomForestClassifier
    X_train         : training feature matrix (for single-feature F1 baseline)
    y_train         : training labels
    X_test          : test feature matrix
    y_test          : test labels (held out — never used during training)
    train_rows      : total training rows (for report header)
    test_rows       : total test rows
    train_suspicious: suspicious count in training set
    test_suspicious : suspicious count in test set
    output_path     : path for the text report

    Returns
    -------
    dict : metrics (same as compute_metrics output)
    """
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    importances = feature_importances(clf, list(X_test.columns))
    sf1 = single_feature_f1(X_test, y_test, X_train, y_train)

    # Print summary to console
    print("\n[evaluator] === Test Set Metrics ===")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
    print(f"  Recall    (macro) : {metrics['recall_macro']:.4f}")
    print(f"  F1        (macro) : {metrics['f1_macro']:.4f}")
    print(f"  F1    (suspicious): {metrics['f1_suspicious']:.4f}")
    print()
    print(metrics["classification_report"])

    write_report(
        metrics, importances, sf1,
        train_rows, test_rows,
        train_suspicious, test_suspicious,
        output_path,
    )

    return metrics
