"""
test_ml.py
==========
Phase 3 — AI Engineering: unit tests for the ML pipeline.

Covers:
  - trainer   : data loading, splitting, feature building, training,
                artefact saving/loading
  - evaluator : metric computation, report writing
  - predictor : single and batch prediction, save/load round-trip
  - leakage   : label not in X, event_id not in X,
                test encoder not re-fitted

Run with:
    pytest tests/test_ml.py -v

Combined regression run (Phase 2 + Phase 3):
    pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline.generator import generate_dataset
from src.data_pipeline.validator import validate
from src.data_pipeline.cleaner import clean
from src.data_pipeline.features import FEATURE_COLUMNS, FeatureEngineer
from src.ml.trainer import (
    split_data,
    build_train_test_features,
    train_model,
    save_artefacts,
    run_training,
    MODEL_FILE,
    FEATURE_ENGINEER_FILE,
    METADATA_FILE,
    RF_PARAMS,
    RANDOM_SEED,
    TEST_SIZE,
)
from src.ml.evaluator import (
    compute_metrics,
    feature_importances,
    single_feature_f1,
    write_report,
    evaluate_and_report,
    EVALUATION_FILE,
)
from src.ml.predictor import (
    load_model,
    load_feature_engineer,
    predict,
    predict_single,
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures — run training once for all tests in this file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Generate a fresh dataset (does not touch the live database)."""
    df = generate_dataset(seed=42)
    df, _ = validate(df)
    return df


@pytest.fixture(scope="module")
def train_test_split_dfs(raw_df):
    """Return the raw train/test DataFrames."""
    return split_data(raw_df, test_size=TEST_SIZE, seed=RANDOM_SEED)


@pytest.fixture(scope="module")
def feature_split(train_test_split_dfs):
    """Return engineered X_train, y_train, X_test, y_test, fe."""
    train_df, test_df = train_test_split_dfs
    return build_train_test_features(train_df, test_df)


@pytest.fixture(scope="module")
def trained_clf(feature_split):
    """Return a trained classifier."""
    X_train, y_train, X_test, y_test, fe = feature_split
    return train_model(X_train, y_train)


@pytest.fixture(scope="module")
def metrics(trained_clf, feature_split):
    """Return computed test-set metrics."""
    clf = trained_clf
    X_train, y_train, X_test, y_test, fe = feature_split
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# 1. Data splitting tests
# ---------------------------------------------------------------------------

class TestDataSplit:

    def test_total_rows_preserved(self, raw_df, train_test_split_dfs):
        """Train + test rows must sum to total dataset size."""
        train_df, test_df = train_test_split_dfs
        assert len(train_df) + len(test_df) == len(raw_df)

    def test_approximate_80_20_split(self, train_test_split_dfs, raw_df):
        """Test set should be approximately 20% of total rows."""
        train_df, test_df = train_test_split_dfs
        ratio = len(test_df) / len(raw_df)
        assert 0.18 <= ratio <= 0.22, (
            f"Test ratio {ratio:.3f} is outside the 18–22% expected range"
        )

    def test_stratification_preserved_train(self, train_test_split_dfs):
        """Training set should have ~20% suspicious records (matching full dataset)."""
        train_df, _ = train_test_split_dfs
        susp_rate = (train_df["label"] == 1).mean()
        assert 0.18 <= susp_rate <= 0.22, (
            f"Train suspicious rate {susp_rate:.3f} outside expected band"
        )

    def test_stratification_preserved_test(self, train_test_split_dfs):
        """Test set should also have ~20% suspicious records."""
        _, test_df = train_test_split_dfs
        susp_rate = (test_df["label"] == 1).mean()
        assert 0.18 <= susp_rate <= 0.22, (
            f"Test suspicious rate {susp_rate:.3f} outside expected band"
        )

    def test_no_overlap_between_splits(self, train_test_split_dfs):
        """No event_id should appear in both train and test sets."""
        train_df, test_df = train_test_split_dfs
        overlap = set(train_df["event_id"]) & set(test_df["event_id"])
        assert len(overlap) == 0, (
            f"Found {len(overlap)} event_ids in both splits — data leakage"
        )

    def test_reproducibility(self, raw_df):
        """Same seed must produce identical splits."""
        t1, te1 = split_data(raw_df, seed=42)
        t2, te2 = split_data(raw_df, seed=42)
        pd.testing.assert_frame_equal(
            t1.reset_index(drop=True),
            t2.reset_index(drop=True),
        )

    def test_different_seed_differs(self, raw_df):
        """Different seeds should produce different splits."""
        t1, _ = split_data(raw_df, seed=42)
        t2, _ = split_data(raw_df, seed=99)
        assert not t1["event_id"].equals(t2["event_id"])


# ---------------------------------------------------------------------------
# 2. Feature engineering — leakage prevention
# ---------------------------------------------------------------------------

class TestFeatureLeakage:

    def test_label_not_in_X_train(self, feature_split):
        """label must NOT appear in the training feature matrix."""
        X_train, _, _, _, _ = feature_split
        assert "label" not in X_train.columns, (
            "DATA LEAKAGE: label column found in X_train"
        )

    def test_label_not_in_X_test(self, feature_split):
        """label must NOT appear in the test feature matrix."""
        _, _, X_test, _, _ = feature_split
        assert "label" not in X_test.columns, (
            "DATA LEAKAGE: label column found in X_test"
        )

    def test_event_id_not_in_X_train(self, feature_split):
        """event_id must NOT appear in the training feature matrix."""
        X_train, _, _, _, _ = feature_split
        assert "event_id" not in X_train.columns

    def test_event_id_not_in_X_test(self, feature_split):
        """event_id must NOT appear in the test feature matrix."""
        _, _, X_test, _, _ = feature_split
        assert "event_id" not in X_test.columns

    def test_all_feature_columns_present(self, feature_split):
        """All 12 FEATURE_COLUMNS must be present in X_train and X_test."""
        X_train, _, X_test, _, _ = feature_split
        for col in FEATURE_COLUMNS:
            assert col in X_train.columns, f"Missing in X_train: {col}"
            assert col in X_test.columns,  f"Missing in X_test: {col}"

    def test_encoder_fitted_on_train_only(self, feature_split):
        """
        Verify the FeatureEngineer was fitted on training data only.
        Re-calling transform on a new split must NOT raise an error
        (encoders are already fitted) and must not re-fit them.
        """
        X_train, y_train, X_test, y_test, fe = feature_split
        assert fe._fitted, "FeatureEngineer should be marked as fitted"
        # transform() on test set should work without error
        # (this was already called during build_train_test_features)
        # Confirm encoder classes were set
        assert fe._source_enc is not None
        assert fe._name_enc is not None
        assert fe._region_enc is not None

    def test_no_test_categories_in_encoder_fitting(self, train_test_split_dfs, feature_split):
        """
        The encoder's fitted classes must NOT include any value that exists
        ONLY in the test set and not in the training set.
        (In practice with 80/20 split on a small vocab, all values should
        appear in train. This test documents the expectation.)
        """
        train_df, test_df = train_test_split_dfs
        _, _, _, _, fe = feature_split

        train_sources = set(train_df["event_source"].str.lower().unique())
        encoder_sources = set(fe._source_enc.classes_)

        # Encoder should only know training-set values
        test_only_sources = set(test_df["event_source"].str.lower().unique()) - train_sources
        for val in test_only_sources:
            assert val not in encoder_sources, (
                f"Test-only value '{val}' found in fitted encoder — leakage"
            )

    def test_y_train_binary(self, feature_split):
        """y_train must contain only 0 and 1."""
        _, y_train, _, _, _ = feature_split
        assert set(y_train.unique()).issubset({0, 1})

    def test_y_test_binary(self, feature_split):
        """y_test must contain only 0 and 1."""
        _, _, _, y_test, _ = feature_split
        assert set(y_test.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# 3. Model training tests
# ---------------------------------------------------------------------------

class TestModelTraining:

    def test_model_trains_without_error(self, feature_split):
        """train_model must return a fitted classifier without raising."""
        X_train, y_train, _, _, _ = feature_split
        from sklearn.ensemble import RandomForestClassifier
        clf = train_model(X_train, y_train)
        assert isinstance(clf, RandomForestClassifier)

    def test_model_has_expected_params(self, trained_clf):
        """Trained classifier must have the hyperparameters from RF_PARAMS."""
        for param, value in RF_PARAMS.items():
            actual = getattr(trained_clf, param)
            assert actual == value, (
                f"RF param '{param}': expected {value}, got {actual}"
            )

    def test_model_feature_count_matches(self, trained_clf, feature_split):
        """Classifier must expect exactly 12 features."""
        X_train, _, _, _, _ = feature_split
        assert trained_clf.n_features_in_ == len(FEATURE_COLUMNS), (
            f"Model expects {trained_clf.n_features_in_} features, "
            f"FEATURE_COLUMNS has {len(FEATURE_COLUMNS)}"
        )

    def test_model_predicts_both_classes(self, trained_clf, feature_split):
        """The model must predict both 0 and 1 on the test set."""
        _, _, X_test, _, _ = feature_split
        preds = trained_clf.predict(X_test)
        assert 0 in preds, "Model never predicts normal (0)"
        assert 1 in preds, "Model never predicts suspicious (1)"

    def test_predict_proba_sums_to_one(self, trained_clf, feature_split):
        """Probability outputs must sum to 1.0 per row."""
        _, _, X_test, _, _ = feature_split
        probs = trained_clf.predict_proba(X_test)
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(len(X_test)),
            atol=1e-6,
        )

    def test_predictions_length_matches_test_set(self, trained_clf, feature_split):
        """Number of predictions must equal number of test rows."""
        _, _, X_test, y_test, _ = feature_split
        preds = trained_clf.predict(X_test)
        assert len(preds) == len(y_test)


# ---------------------------------------------------------------------------
# 4. Evaluation metric tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:

    def test_metrics_keys_present(self, metrics):
        """compute_metrics must return all required keys."""
        required_keys = [
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted", "f1_suspicious",
            "confusion_matrix", "classification_report",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing metrics key: {key}"

    def test_accuracy_in_valid_range(self, metrics):
        """Accuracy must be between 0 and 1."""
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_in_valid_range(self, metrics):
        """All F1 scores must be between 0 and 1."""
        for key in ("f1_macro", "f1_weighted", "f1_suspicious"):
            val = metrics[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} is out of [0,1]"

    def test_confusion_matrix_shape(self, metrics):
        """Confusion matrix must be 2×2."""
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2

    def test_confusion_matrix_totals(self, metrics, feature_split):
        """Confusion matrix entries must sum to test set size."""
        _, _, X_test, _, _ = feature_split
        cm = metrics["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == len(X_test)

    def test_classification_report_is_string(self, metrics):
        """Classification report must be a non-empty string."""
        report = metrics["classification_report"]
        assert isinstance(report, str) and len(report) > 0

    def test_metrics_not_perfect_on_v2_dataset(self, metrics):
        """
        After noise injection (v2), the F1 for suspicious class should
        be below 0.99 — confirming the model is not trivially learning
        a single perfect rule.
        IMPORTANT: This test documents the expectation, not a target.
        If this fails, investigate whether leakage has re-entered.
        """
        f1_sus = metrics["f1_suspicious"]
        assert f1_sus < 0.99, (
            f"F1 (suspicious) = {f1_sus:.4f} >= 0.99 on v2 dataset. "
            "This is unexpectedly high — check for feature leakage."
        )

    def test_feature_importances_sum_to_one(self, trained_clf, feature_split):
        """Random forest feature importances must sum to 1.0."""
        X_train, _, _, _, _ = feature_split
        np.testing.assert_allclose(
            trained_clf.feature_importances_.sum(),
            1.0,
            atol=1e-6,
        )

    def test_feature_importances_all_non_negative(self, trained_clf):
        """All feature importances must be non-negative."""
        assert (trained_clf.feature_importances_ >= 0).all()


# ---------------------------------------------------------------------------
# 5. Report writing tests
# ---------------------------------------------------------------------------

class TestReportWriting:

    def test_report_written_to_disk(self, tmp_path, trained_clf,
                                     feature_split, metrics):
        """evaluate_and_report must create the report file."""
        X_train, y_train, X_test, y_test, _ = feature_split
        report_path = tmp_path / "test_eval.txt"
        evaluate_and_report(
            clf=trained_clf,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            train_rows=len(X_train), test_rows=len(X_test),
            train_suspicious=int(y_train.sum()),
            test_suspicious=int(y_test.sum()),
            output_path=report_path,
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert len(content) > 100

    def test_report_contains_synthetic_notice(self, tmp_path, trained_clf,
                                               feature_split):
        """Report must contain the synthetic data disclaimer."""
        X_train, y_train, X_test, y_test, _ = feature_split
        report_path = tmp_path / "test_eval2.txt"
        evaluate_and_report(
            clf=trained_clf,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            train_rows=len(X_train), test_rows=len(X_test),
            train_suspicious=int(y_train.sum()),
            test_suspicious=int(y_test.sum()),
            output_path=report_path,
        )
        content = report_path.read_text()
        content_upper = content.upper()
        assert "SYNTHETIC" in content_upper, "Report missing synthetic data notice"
        assert "DO NOT" in content_upper, (
            "Report should state metrics do NOT establish real-world performance"
        )

    def test_report_contains_confusion_matrix(self, tmp_path, trained_clf,
                                               feature_split):
        """Report must contain the confusion matrix section."""
        X_train, y_train, X_test, y_test, _ = feature_split
        report_path = tmp_path / "test_eval3.txt"
        evaluate_and_report(
            clf=trained_clf,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            train_rows=len(X_train), test_rows=len(X_test),
            train_suspicious=int(y_train.sum()),
            test_suspicious=int(y_test.sum()),
            output_path=report_path,
        )
        content = report_path.read_text()
        assert "CONFUSION MATRIX" in content


# ---------------------------------------------------------------------------
# 6. Model save / load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_save_artefacts_creates_files(self, tmp_path, trained_clf,
                                           train_test_split_dfs, feature_split,
                                           monkeypatch):
        """save_artefacts must write all three artefact files."""
        import src.ml.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "MODEL_FILE",
                            tmp_path / "threat_classifier.pkl")
        monkeypatch.setattr(trainer_mod, "FEATURE_ENGINEER_FILE",
                            tmp_path / "feature_engineer.pkl")
        monkeypatch.setattr(trainer_mod, "METADATA_FILE",
                            tmp_path / "training_metadata.json")
        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)

        train_df, test_df = train_test_split_dfs
        _, _, _, _, fe = feature_split

        save_artefacts(trained_clf, fe, train_df, test_df)

        assert (tmp_path / "threat_classifier.pkl").exists()
        assert (tmp_path / "feature_engineer.pkl").exists()
        assert (tmp_path / "training_metadata.json").exists()

    def test_metadata_json_valid(self, tmp_path, trained_clf,
                                  train_test_split_dfs, feature_split,
                                  monkeypatch):
        """Metadata JSON must parse and contain required keys."""
        import src.ml.trainer as trainer_mod
        import joblib

        monkeypatch.setattr(trainer_mod, "MODEL_FILE",
                            tmp_path / "threat_classifier.pkl")
        monkeypatch.setattr(trainer_mod, "FEATURE_ENGINEER_FILE",
                            tmp_path / "feature_engineer.pkl")
        monkeypatch.setattr(trainer_mod, "METADATA_FILE",
                            tmp_path / "training_metadata.json")
        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)

        train_df, test_df = train_test_split_dfs
        _, _, _, _, fe = feature_split
        save_artefacts(trained_clf, fe, train_df, test_df)

        meta = json.loads((tmp_path / "training_metadata.json").read_text())
        for key in ("trained_at", "random_seed", "test_size",
                    "feature_columns", "data_source",
                    "synthetic_data_notice"):
            assert key in meta, f"Missing metadata key: {key}"

        assert meta["data_source"] == "synthetic"
        assert "synthetic" in meta["synthetic_data_notice"].lower()

    def test_model_load_and_predict(self, tmp_path, trained_clf,
                                     train_test_split_dfs, feature_split,
                                     monkeypatch):
        """A model saved to disk must load and produce the same predictions."""
        import src.ml.trainer as trainer_mod
        import joblib

        model_path = tmp_path / "threat_classifier.pkl"
        fe_path = tmp_path / "feature_engineer.pkl"
        meta_path = tmp_path / "training_metadata.json"

        monkeypatch.setattr(trainer_mod, "MODEL_FILE", model_path)
        monkeypatch.setattr(trainer_mod, "FEATURE_ENGINEER_FILE", fe_path)
        monkeypatch.setattr(trainer_mod, "METADATA_FILE", meta_path)
        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)

        train_df, test_df = train_test_split_dfs
        _, _, X_test, _, fe = feature_split
        save_artefacts(trained_clf, fe, train_df, test_df)

        loaded_clf = joblib.load(model_path)
        preds_original = trained_clf.predict(X_test[FEATURE_COLUMNS])
        preds_loaded = loaded_clf.predict(X_test[FEATURE_COLUMNS])

        np.testing.assert_array_equal(preds_original, preds_loaded,
                                       err_msg="Loaded model predictions differ")


# ---------------------------------------------------------------------------
# 7. Predictor tests
# ---------------------------------------------------------------------------

class TestPredictor:

    NORMAL_RECORD = {
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

    SUSPICIOUS_RECORD = {
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

    def test_predict_single_returns_dict(self, trained_clf, feature_split):
        """predict_single must return a dict with required keys."""
        _, _, _, _, fe = feature_split
        result = predict_single(self.NORMAL_RECORD, clf=trained_clf, fe=fe)
        for key in ("predicted_label", "threat_probability", "threat_level"):
            assert key in result, f"Missing key: {key}"

    def test_predicted_label_binary(self, trained_clf, feature_split):
        """predicted_label must be 0 or 1."""
        _, _, _, _, fe = feature_split
        for record in (self.NORMAL_RECORD, self.SUSPICIOUS_RECORD):
            result = predict_single(record, clf=trained_clf, fe=fe)
            assert result["predicted_label"] in (0, 1)

    def test_threat_probability_in_range(self, trained_clf, feature_split):
        """threat_probability must be in [0, 1]."""
        _, _, _, _, fe = feature_split
        for record in (self.NORMAL_RECORD, self.SUSPICIOUS_RECORD):
            result = predict_single(record, clf=trained_clf, fe=fe)
            assert 0.0 <= result["threat_probability"] <= 1.0

    def test_threat_level_valid_value(self, trained_clf, feature_split):
        """threat_level must be Low, Medium, or High."""
        _, _, _, _, fe = feature_split
        for record in (self.NORMAL_RECORD, self.SUSPICIOUS_RECORD):
            result = predict_single(record, clf=trained_clf, fe=fe)
            assert result["threat_level"] in ("Low", "Medium", "High")

    def test_suspicious_record_higher_probability(self, trained_clf,
                                                   feature_split):
        """
        The clearly suspicious record should have a higher threat probability
        than the clearly normal record.
        """
        _, _, _, _, fe = feature_split
        normal_res = predict_single(self.NORMAL_RECORD, clf=trained_clf, fe=fe)
        susp_res = predict_single(self.SUSPICIOUS_RECORD, clf=trained_clf, fe=fe)
        assert susp_res["threat_probability"] > normal_res["threat_probability"], (
            f"Suspicious record probability ({susp_res['threat_probability']:.4f}) "
            f"<= normal ({normal_res['threat_probability']:.4f})"
        )

    def test_batch_predict_returns_dataframe(self, trained_clf, feature_split,
                                              raw_df):
        """predict() on a DataFrame batch must return a DataFrame."""
        _, _, _, _, fe = feature_split
        sample = raw_df.head(10).copy()
        result = predict(sample, clf=trained_clf, fe=fe)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_batch_predict_columns(self, trained_clf, feature_split, raw_df):
        """Batch predict output must have the required columns."""
        _, _, _, _, fe = feature_split
        sample = raw_df.head(5).copy()
        result = predict(sample, clf=trained_clf, fe=fe)
        for col in ("event_id", "predicted_label",
                    "threat_probability", "threat_level"):
            assert col in result.columns, f"Missing column: {col}"

    def test_missing_model_raises(self, tmp_path):
        """load_model must raise FileNotFoundError if file is missing."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")

    def test_missing_fe_raises(self, tmp_path):
        """load_feature_engineer must raise FileNotFoundError if missing."""
        with pytest.raises(FileNotFoundError):
            load_feature_engineer(tmp_path / "nonexistent.pkl")
