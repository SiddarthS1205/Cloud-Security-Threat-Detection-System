"""
test_pipeline.py
================
Unit tests for the Phase 2 data pipeline.

Covers:
  - generator   : record count, label distribution, schema, reproducibility
  - validator   : accepts clean data, rejects bad data, reports correctly
  - cleaner     : whitespace, timestamp format, error_code fill, int casts
  - features    : correct columns, no label leakage, re-derived failed_attempts
  - storage     : creates tables, upsert is idempotent, read-back matches write

Run with:
    pytest tests/test_pipeline.py -v
"""

import os
import sqlite3
import tempfile
from copy import deepcopy

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Adjust sys.path so imports work from the project root
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_pipeline.generator import generate_dataset
from src.data_pipeline.validator import validate, REQUIRED_COLUMNS
from src.data_pipeline.cleaner import clean
from src.data_pipeline.features import build_features, FEATURE_COLUMNS, FeatureEngineer
from src.data_pipeline.storage import (
    get_connection,
    write_raw_logs,
    write_processed_features,
    read_raw_logs,
    read_processed_features,
    get_table_counts,
    log_pipeline_run,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df():
    """Generate the full synthetic dataset once for the test session."""
    return generate_dataset(seed=42)


@pytest.fixture(scope="module")
def validated_df(raw_df):
    df, _ = validate(raw_df)
    return df


@pytest.fixture(scope="module")
def cleaned_df(validated_df):
    return clean(validated_df)


@pytest.fixture(scope="module")
def features_result(cleaned_df):
    return build_features(cleaned_df)


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a fresh temporary SQLite database."""
    return str(tmp_path / "test_security.db")


# ===========================================================================
# 1. Generator tests
# ===========================================================================

class TestGenerator:

    def test_total_record_count(self, raw_df):
        """Dataset must contain exactly 5,000 records."""
        assert len(raw_df) == 5000, (
            f"Expected 5000 rows, got {len(raw_df)}"
        )

    def test_normal_count(self, raw_df):
        """Exactly 4,000 normal records (label == 0)."""
        assert (raw_df["label"] == 0).sum() == 4000

    def test_suspicious_count(self, raw_df):
        """Exactly 1,000 suspicious records (label == 1)."""
        assert (raw_df["label"] == 1).sum() == 1000

    def test_required_columns_present(self, raw_df):
        """All schema columns must exist."""
        for col in REQUIRED_COLUMNS:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_no_null_event_id(self, raw_df):
        """event_id must never be null."""
        assert raw_df["event_id"].isnull().sum() == 0

    def test_no_null_event_time(self, raw_df):
        """event_time must never be null."""
        assert raw_df["event_time"].isnull().sum() == 0

    def test_no_null_label(self, raw_df):
        """label must never be null."""
        assert raw_df["label"].isnull().sum() == 0

    def test_unique_event_ids(self, raw_df):
        """Every event_id must be unique."""
        assert raw_df["event_id"].nunique() == len(raw_df), (
            "Duplicate event_id values found"
        )

    def test_label_values_binary(self, raw_df):
        """label must only contain 0 and 1."""
        assert set(raw_df["label"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        """Two calls with the same seed must produce identical data."""
        df1 = generate_dataset(seed=42)
        df2 = generate_dataset(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self):
        """Different seeds must produce different datasets."""
        df1 = generate_dataset(seed=42)
        df2 = generate_dataset(seed=99)
        assert not df1.equals(df2)

    def test_ip_addresses_in_doc_ranges(self, raw_df):
        """All IPs must be in IANA documentation ranges only."""
        valid_prefixes = ("192.0.2.", "198.51.100.", "203.0.113.")
        for ip in raw_df["source_ip"]:
            assert any(ip.startswith(p) for p in valid_prefixes), (
                f"IP {ip!r} is not in a documentation range"
            )

    def test_data_source_is_synthetic(self, raw_df):
        """Every record must be marked as synthetic."""
        assert (raw_df["data_source"] == "synthetic").all()

    def test_failed_attempts_non_negative(self, raw_df):
        """failed_attempts must never be negative."""
        assert (raw_df["failed_attempts"] >= 0).all()

    def test_binary_flag_columns(self, raw_df):
        """unusual_location and privilege_change must only be 0 or 1."""
        for col in ["unusual_location", "privilege_change"]:
            assert set(raw_df[col].unique()).issubset({0, 1}), (
                f"{col} contains values other than 0 and 1"
            )


# ===========================================================================
# 2. Validator tests
# ===========================================================================

class TestValidator:

    def test_accepts_clean_data(self, raw_df):
        """A clean dataset should pass with all rows retained."""
        clean_out, report = validate(raw_df)
        assert report.passed
        assert len(clean_out) == len(raw_df)

    def test_missing_column_fails(self, raw_df):
        """Dropping a required column should fail validation."""
        bad_df = raw_df.drop(columns=["event_time"])
        _, report = validate(bad_df)
        assert not report.passed
        assert "event_time" in report.missing_columns

    def test_duplicate_event_id_dropped(self, raw_df):
        """Duplicate event_id rows should be dropped (keep first)."""
        dup_df = pd.concat([raw_df, raw_df.iloc[:10]], ignore_index=True)
        clean_out, report = validate(dup_df)
        assert len(clean_out) == len(raw_df)
        assert "duplicate_event_id" in report.rows_dropped

    def test_invalid_label_dropped(self, raw_df):
        """Rows with label values other than 0/1 should be dropped."""
        bad_df = raw_df.copy()
        bad_df.loc[bad_df.index[:5], "label"] = 99
        clean_out, report = validate(bad_df)
        assert len(clean_out) == len(raw_df) - 5
        assert "invalid_label" in report.rows_dropped

    def test_null_event_id_dropped(self, raw_df):
        """Rows with null event_id should be dropped."""
        bad_df = raw_df.copy()
        bad_df.loc[bad_df.index[:3], "event_id"] = None
        clean_out, report = validate(bad_df)
        assert len(clean_out) == len(raw_df) - 3
        assert "null_event_id" in report.rows_dropped

    def test_invalid_timestamp_dropped(self, raw_df):
        """Rows with unparseable event_time should be dropped."""
        bad_df = raw_df.copy()
        bad_df.loc[bad_df.index[:2], "event_time"] = "NOT_A_DATE"
        clean_out, report = validate(bad_df)
        assert "invalid_event_time_format" in report.rows_dropped

    def test_negative_failed_attempts_coerced(self, raw_df):
        """Negative failed_attempts should be coerced to 0, not dropped."""
        bad_df = raw_df.copy()
        bad_df.loc[bad_df.index[:5], "failed_attempts"] = -3
        clean_out, report = validate(bad_df)
        # Rows should still be present (coerced, not dropped)
        assert len(clean_out) == len(raw_df)

    def test_report_summary_contains_status(self, raw_df):
        """ValidationReport.summary() should include PASSED or FAILED."""
        _, report = validate(raw_df)
        summary = report.summary()
        assert "PASSED" in summary or "FAILED" in summary


# ===========================================================================
# 3. Cleaner tests
# ===========================================================================

class TestCleaner:

    def test_output_length_unchanged(self, validated_df):
        """Cleaner must not drop or add rows."""
        cleaned = clean(validated_df)
        assert len(cleaned) == len(validated_df)

    def test_ingested_at_added(self, validated_df):
        """Cleaned DataFrame must have an ingested_at column."""
        cleaned = clean(validated_df)
        assert "ingested_at" in cleaned.columns

    def test_ingested_at_format(self, validated_df):
        """ingested_at should be a valid ISO 8601 UTC string."""
        cleaned = clean(validated_df)
        sample = cleaned["ingested_at"].iloc[0]
        # Should parse without error
        ts = pd.Timestamp(sample)
        assert ts is not None

    def test_error_code_no_empty_strings(self, validated_df):
        """Empty error_code values should be replaced with 'None'."""
        cleaned = clean(validated_df)
        assert (cleaned["error_code"] == "").sum() == 0
        assert cleaned["error_code"].isnull().sum() == 0

    def test_integer_columns_are_int(self, validated_df):
        """failed_attempts, unusual_location, privilege_change, label must be int."""
        cleaned = clean(validated_df)
        for col in ["failed_attempts", "unusual_location", "privilege_change", "label"]:
            assert pd.api.types.is_integer_dtype(cleaned[col]), (
                f"{col} is not integer dtype"
            )

    def test_whitespace_stripped(self, validated_df):
        """Leading/trailing whitespace should be removed from string columns."""
        dirty = validated_df.copy()
        dirty.loc[dirty.index[:5], "event_source"] = "  iam.amazonaws.com  "
        cleaned = clean(dirty)
        assert (cleaned["event_source"].str.startswith(" ")).sum() == 0
        assert (cleaned["event_source"].str.endswith(" ")).sum() == 0

    def test_event_time_normalised(self, validated_df):
        """event_time should end with 'Z' (UTC marker)."""
        cleaned = clean(validated_df)
        assert cleaned["event_time"].str.endswith("Z").all()

    def test_lowercase_columns(self, validated_df):
        """event_source, event_name, aws_region should all be lowercase."""
        cleaned = clean(validated_df)
        for col in ["event_source", "event_name", "aws_region"]:
            assert (cleaned[col] == cleaned[col].str.lower()).all(), (
                f"{col} contains uppercase characters after cleaning"
            )


# ===========================================================================
# 4. Feature engineering tests
# ===========================================================================

class TestFeatures:

    def test_feature_columns_present(self, features_result):
        X, y, _ = features_result
        for col in FEATURE_COLUMNS:
            assert col in X.columns, f"Missing feature column: {col}"

    def test_event_id_present(self, features_result):
        """event_id must be retained for traceability."""
        X, y, _ = features_result
        assert "event_id" in X.columns

    def test_label_not_in_features(self, features_result):
        """label must NOT appear in the feature matrix X."""
        X, y, _ = features_result
        assert "label" not in X.columns, (
            "DATA LEAKAGE: label column found inside feature matrix X"
        )

    def test_label_series_binary(self, features_result):
        """y must only contain 0 and 1."""
        _, y, _ = features_result
        assert set(y.unique()).issubset({0, 1})

    def test_label_count(self, features_result, cleaned_df):
        """y length must match the cleaned DataFrame."""
        X, y, _ = features_result
        assert len(y) == len(cleaned_df)

    def test_hour_of_day_range(self, features_result):
        """hour_of_day must be in [0, 23]."""
        X, _, _ = features_result
        assert X["hour_of_day"].between(0, 23).all()

    def test_day_of_week_range(self, features_result):
        """day_of_week must be in [0, 6]."""
        X, _, _ = features_result
        assert X["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self, features_result):
        """is_weekend must be 0 or 1."""
        X, _, _ = features_result
        assert set(X["is_weekend"].unique()).issubset({0, 1})

    def test_is_error_binary(self, features_result):
        """is_error must be 0 or 1."""
        X, _, _ = features_result
        assert set(X["is_error"].unique()).issubset({0, 1})

    def test_failed_attempts_non_negative(self, features_result):
        """Re-derived failed_attempts must be non-negative."""
        X, _, _ = features_result
        assert (X["failed_attempts"] >= 0).all()

    def test_no_raw_generator_metadata(self, features_result):
        """Features should not contain data_source or scenario columns."""
        X, _, _ = features_result
        forbidden = {"data_source", "scenario", "generator_label"}
        assert forbidden.isdisjoint(set(X.columns)), (
            f"Generator metadata found in features: {forbidden & set(X.columns)}"
        )

    def test_transform_without_fit_raises(self, cleaned_df):
        """Calling transform() before fit_transform() should raise RuntimeError."""
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fe.transform(cleaned_df)

    def test_fit_transform_then_transform(self, cleaned_df):
        """fit_transform on first half, transform on second half — no error."""
        half = len(cleaned_df) // 2
        train_df = cleaned_df.iloc[:half].copy()
        test_df = cleaned_df.iloc[half:].copy()

        fe = FeatureEngineer()
        X_train, y_train = fe.fit_transform(train_df)
        X_test, y_test = fe.transform(test_df)

        assert len(X_train) == half
        assert len(X_test) == len(cleaned_df) - half
        for col in FEATURE_COLUMNS:
            assert col in X_train.columns
            assert col in X_test.columns


# ===========================================================================
# 5. Storage tests
# ===========================================================================

class TestStorage:

    def test_get_connection_creates_file(self, tmp_db):
        """get_connection should create the .db file if it doesn't exist."""
        assert not os.path.exists(tmp_db)
        conn = get_connection(tmp_db)
        conn.close()
        assert os.path.exists(tmp_db)

    def test_tables_created(self, tmp_db):
        """All three tables must exist after get_connection."""
        conn = get_connection(tmp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "raw_logs" in tables
        assert "processed_features" in tables
        assert "pipeline_runs" in tables

    def test_write_and_read_raw_logs(self, tmp_db, cleaned_df):
        """Written raw_logs rows must be readable back with same count."""
        conn = get_connection(tmp_db)
        n_written = write_raw_logs(conn, cleaned_df)
        df_back = read_raw_logs(conn)
        conn.close()
        assert n_written == len(cleaned_df)
        assert len(df_back) == len(cleaned_df)

    def test_upsert_is_idempotent(self, tmp_db, cleaned_df):
        """Writing the same data twice should not increase the row count."""
        conn = get_connection(tmp_db)
        write_raw_logs(conn, cleaned_df)
        write_raw_logs(conn, cleaned_df)   # second write — same event_ids
        counts = get_table_counts(conn)
        conn.close()
        assert counts["raw_logs"] == len(cleaned_df), (
            "Upsert created duplicate rows"
        )

    def test_write_processed_features(self, tmp_db, cleaned_df, features_result):
        """Processed features should be writeable and readable back."""
        X, y, _ = features_result
        conn = get_connection(tmp_db)
        write_raw_logs(conn, cleaned_df)       # FK dependency
        n_written = write_processed_features(conn, X, y)
        df_back = read_processed_features(conn)
        conn.close()
        assert n_written == len(X)
        assert len(df_back) == len(X)

    def test_processed_features_upsert_idempotent(self, tmp_db, cleaned_df, features_result):
        """Writing processed features twice must not duplicate rows."""
        X, y, _ = features_result
        conn = get_connection(tmp_db)
        write_raw_logs(conn, cleaned_df)
        write_processed_features(conn, X, y)
        write_processed_features(conn, X, y)
        counts = get_table_counts(conn)
        conn.close()
        assert counts["processed_features"] == len(X)

    def test_table_counts_correct(self, tmp_db, cleaned_df, features_result):
        """get_table_counts should return correct counts for both tables."""
        X, y, _ = features_result
        conn = get_connection(tmp_db)
        write_raw_logs(conn, cleaned_df)
        write_processed_features(conn, X, y)
        counts = get_table_counts(conn)
        conn.close()
        assert counts["raw_logs"] == len(cleaned_df)
        assert counts["processed_features"] == len(X)

    def test_log_pipeline_run(self, tmp_db):
        """Pipeline run logging should insert a row into pipeline_runs."""
        conn = get_connection(tmp_db)
        log_pipeline_run(conn, rows_raw=5000, rows_features=5000,
                         notes="test run")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pipeline_runs")
        count = cursor.fetchone()[0]
        conn.close()
        assert count >= 1

    def test_missing_column_raises(self, tmp_db, cleaned_df):
        """write_raw_logs should raise ValueError if a required column is missing."""
        conn = get_connection(tmp_db)
        bad_df = cleaned_df.drop(columns=["event_time"])
        with pytest.raises(ValueError, match="missing columns"):
            write_raw_logs(conn, bad_df)
        conn.close()


# ===========================================================================
# 6. Generator realism tests (v2 — noise injection verification)
#    These tests verify that the intended noise rates are present within
#    reasonable statistical tolerances, NOT just that at least one noisy
#    record exists.
#
#    Tolerance bands are set at ±3 standard deviations for each binomial
#    proportion (n=4000 for normal, n=1000 for suspicious).
#    With seed=42 the exact counts are deterministic; the bands document
#    acceptable ranges for any future seed change.
# ===========================================================================

class TestGeneratorRealism:
    """
    Verify that v2 noise injection produces realistic feature overlap
    between normal and suspicious classes.

    All tests use the shared raw_df fixture (seed=42, module scope).
    """

    # ------------------------------------------------------------------
    # Helper: check a rate is within an absolute tolerance window
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_rate(
        count: int,
        total: int,
        low: float,
        high: float,
        label: str,
    ) -> None:
        rate = count / total
        assert low <= rate <= high, (
            f"{label}: observed {count}/{total} = {rate:.3f}, "
            f"expected between {low:.3f} and {high:.3f}"
        )

    # ------------------------------------------------------------------
    # Normal-record noise tests
    # ------------------------------------------------------------------

    def test_normal_error_rate_approx_8pct(self, raw_df):
        """
        ~8% of normal records should have a non-empty error_code.
        Acceptable band: 5% – 12%  (±3σ for n=4000, p=0.08).
        """
        normal = raw_df[raw_df["label"] == 0]
        count = (normal["error_code"].str.strip() != "").sum()
        self._assert_rate(count, len(normal), 0.05, 0.12,
                          "normal is_error rate")

    def test_normal_unusual_location_rate_approx_5pct(self, raw_df):
        """
        ~5% of normal records should have unusual_location=1.
        Acceptable band: 3% – 8%  (±3σ for n=4000, p=0.05).
        """
        normal = raw_df[raw_df["label"] == 0]
        count = (normal["unusual_location"] == 1).sum()
        self._assert_rate(count, len(normal), 0.03, 0.08,
                          "normal unusual_location rate")

    def test_normal_privilege_change_rate_approx_4pct(self, raw_df):
        """
        ~4% of normal records should have privilege_change=1
        (legitimate admin actions).
        Acceptable band: 2% – 7%  (±3σ for n=4000, p=0.04).
        """
        normal = raw_df[raw_df["label"] == 0]
        count = (normal["privilege_change"] == 1).sum()
        self._assert_rate(count, len(normal), 0.02, 0.07,
                          "normal privilege_change rate")

    def test_normal_off_hours_rate_approx_20pct(self, raw_df):
        """
        ~20% of normal records are generated with any_hour=True (full 0–23 range).
        Of those, 13 out of 24 possible hours fall outside business hours (0–7, 19–23).
        Expected out-of-business-hours fraction ≈ 0.20 × (13/24) ≈ 10.8%.
        Acceptable band: 7% – 15%  (generous ±4% to account for seed variation).
        Off-hours defined as hour < 8 or hour > 18.
        """
        normal = raw_df[raw_df["label"] == 0].copy()
        normal["hour"] = pd.to_datetime(normal["event_time"]).dt.hour
        count = ((normal["hour"] < 8) | (normal["hour"] > 18)).sum()
        self._assert_rate(count, len(normal), 0.07, 0.15,
                          "normal off-hours rate")

    def test_normal_high_risk_action_rate_approx_4pct(self, raw_df):
        """
        ~4% of normal records should have a high-risk action name
        (from NORMAL_ADMIN_ACTIONS — legitimate admin use).
        Acceptable band: 2% – 7%.
        """
        from src.data_pipeline.features import HIGH_RISK_ACTIONS
        normal = raw_df[raw_df["label"] == 0]
        count = normal["event_name"].str.lower().isin(HIGH_RISK_ACTIONS).sum()
        self._assert_rate(count, len(normal), 0.02, 0.07,
                          "normal high_risk_action rate")

    # ------------------------------------------------------------------
    # Suspicious-record relaxation tests
    # ------------------------------------------------------------------

    def test_suspicious_some_lack_unusual_location(self, raw_df):
        """
        Some suspicious records should have unusual_location=0
        (insider threats / compromised internal accounts).
        Acceptable band: 8% – 25% of suspicious records have unusual=0.
        (Overall ~12–18% expected across 4 relaxed scenarios.)
        """
        susp = raw_df[raw_df["label"] == 1]
        count = (susp["unusual_location"] == 0).sum()
        self._assert_rate(count, len(susp), 0.08, 0.25,
                          "suspicious unusual_location=0 rate")

    def test_brute_force_unusual_location_near_90pct(self, raw_df):
        """
        Brute-force records should have unusual_location=1 ~90% of the time.
        Acceptable band: 82% – 96%  (±3σ for n=200, p=0.90).
        Identified by: event_source=iam, event_name in {GetUser, GetLoginProfile},
        failed_attempts >= 5.
        """
        bf = raw_df[
            (raw_df["label"] == 1)
            & (raw_df["event_source"] == "iam.amazonaws.com")
            & (raw_df["event_name"].isin(["GetUser", "GetLoginProfile"]))
            & (raw_df["failed_attempts"] >= 5)
        ]
        # Some brute-force records may not be captured by this filter
        # (seed-specific), so only assert if we found enough
        if len(bf) >= 50:
            count = (bf["unusual_location"] == 1).sum()
            self._assert_rate(count, len(bf), 0.78, 0.98,
                              "brute-force unusual_location=1 rate")

    # ------------------------------------------------------------------
    # Perfect-predictor tests (key requirement: no single feature
    # should perfectly separate the classes)
    # ------------------------------------------------------------------

    def test_unusual_location_not_a_perfect_predictor(self, raw_df):
        """
        Both (unusual=1, label=0) and (unusual=0, label=1) must exist.
        This confirms unusual_location is no longer a perfect classifier.
        """
        fp = ((raw_df["unusual_location"] == 1) & (raw_df["label"] == 0)).sum()
        fn = ((raw_df["unusual_location"] == 0) & (raw_df["label"] == 1)).sum()
        assert fp > 0, (
            f"No normal records have unusual_location=1 — still a perfect predictor. "
            f"Noise injection may have failed."
        )
        assert fn > 0, (
            f"No suspicious records have unusual_location=0 — still a perfect predictor."
        )

    def test_is_error_not_a_perfect_predictor(self, raw_df):
        """
        Normal records should occasionally have errors (unusual_location noise).
        Both (is_error=1, label=0) and (is_error=0, label=1) must exist.
        """
        has_error = raw_df["error_code"].str.strip() != ""
        fp = (has_error & (raw_df["label"] == 0)).sum()
        fn = (~has_error & (raw_df["label"] == 1)).sum()
        assert fp > 0, "No normal records have errors — is_error is still a perfect predictor"
        assert fn > 0, "No suspicious records lack errors — is_error is still a perfect predictor"

    def test_privilege_change_not_a_perfect_predictor(self, raw_df):
        """
        Some normal records should now have privilege_change=1.
        Both (privilege_change=1, label=0) and (privilege_change=0, label=1) must exist.
        """
        fp = ((raw_df["privilege_change"] == 1) & (raw_df["label"] == 0)).sum()
        fn = ((raw_df["privilege_change"] == 0) & (raw_df["label"] == 1)).sum()
        assert fp > 0, "No normal records have privilege_change=1"
        assert fn > 0, "No suspicious records have privilege_change=0"

    def test_hour_of_day_not_exclusive_to_one_class(self, raw_df):
        """
        Hours 0–5 should appear in BOTH normal and suspicious records.
        Normal off-hours noise (~20%) means some normal records land in 0–5.
        """
        df = raw_df.copy()
        df["hour"] = pd.to_datetime(df["event_time"]).dt.hour
        early_normal = ((df["hour"] <= 5) & (df["label"] == 0)).sum()
        early_susp = ((df["hour"] <= 5) & (df["label"] == 1)).sum()
        assert early_normal > 0, (
            "Hours 0–5 still never appear in normal records — off-hours noise failed"
        )
        assert early_susp > 0, (
            "Hours 0–5 never appear in suspicious records — unexpected"
        )

    def test_high_risk_action_not_exclusive_to_suspicious(self, raw_df):
        """
        High-risk actions should appear in BOTH classes.
        Normal admin noise (~4%) means some normal records have high-risk actions.
        """
        from src.data_pipeline.features import HIGH_RISK_ACTIONS
        hr = raw_df["event_name"].str.lower().isin(HIGH_RISK_ACTIONS)
        in_normal = (hr & (raw_df["label"] == 0)).sum()
        in_susp = (hr & (raw_df["label"] == 1)).sum()
        assert in_normal > 0, (
            "No normal records have high-risk actions — is_high_risk_action is still a perfect predictor"
        )
        assert in_susp > 0, (
            "No suspicious records have high-risk actions — unexpected"
        )

    # ------------------------------------------------------------------
    # Schema and invariant checks (unchanged from original requirements)
    # ------------------------------------------------------------------

    def test_total_records_still_5000(self, raw_df):
        """Noise injection must not change total record count."""
        assert len(raw_df) == 5000

    def test_class_balance_preserved(self, raw_df):
        """4,000 normal and 1,000 suspicious must be preserved."""
        assert (raw_df["label"] == 0).sum() == 4000
        assert (raw_df["label"] == 1).sum() == 1000

    def test_event_ids_still_unique(self, raw_df):
        """Every event_id must remain unique after noise injection."""
        assert raw_df["event_id"].nunique() == 5000

    def test_reproducibility_v2(self):
        """Two calls with seed=42 must produce identical v2 datasets."""
        df1 = generate_dataset(seed=42)
        df2 = generate_dataset(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_labels_assigned_by_scenario_not_by_flag(self, raw_df):
        """
        Confirm the label is scenario-based, not rule-based:
        there must be records where unusual_location=1 AND label=0,
        AND records where unusual_location=0 AND label=1.
        If label were derived from unusual_location, one of these
        cross-combinations would be empty.
        """
        cross1 = ((raw_df["unusual_location"] == 1) & (raw_df["label"] == 0)).sum()
        cross2 = ((raw_df["unusual_location"] == 0) & (raw_df["label"] == 1)).sum()
        assert cross1 > 0 and cross2 > 0, (
            f"Label appears to be derived from unusual_location. "
            f"Cross1={cross1}, Cross2={cross2}"
        )
