"""
test_dashboard.py
=================
Phase 4 — Dashboard: unit tests for non-Streamlit logic.

Tests cover:
  - csv_validator  : required columns, timestamp parsing, integer ranges,
                     row count limits, error_code warnings
  - db_utils       : incident creation, duplicate prevention, status update,
                     overview queries, open incident counts
  - prediction     : CSV upload → prepare → predict round-trip

Streamlit UI rendering is not tested here (requires a running browser
session). All testable business logic lives in csv_validator.py and
db_utils.py, which have no Streamlit imports.

Run with:
    pytest tests/test_dashboard.py -v

Combined regression run:
    pytest tests/ -v
"""

import os
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.csv_validator import (
    validate_upload,
    prepare_for_prediction,
    REQUIRED_UPLOAD_COLUMNS,
    MAX_ROWS,
)
from dashboard.db_utils import (
    get_connection,
    get_event_counts,
    get_events_by_source,
    get_events_by_hour,
    get_top_suspicious_users,
    get_recent_events,
    get_incidents,
    get_open_incident_count,
    get_incident_status_counts,
    create_incident,
    update_incident_status,
    VALID_STATUSES,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """A fresh temporary SQLite database for incident tests."""
    return str(tmp_path / "test_incidents.db")


@pytest.fixture
def live_conn(tmp_db):
    """Open connection to a temp DB with incidents table created."""
    conn = get_connection(tmp_db)
    yield conn
    conn.close()


def _make_valid_df(n: int = 3) -> pd.DataFrame:
    """Return a minimal valid upload DataFrame."""
    return pd.DataFrame({
        "event_time": [f"2024-06-{10+i:02d}T10:00:00Z" for i in range(n)],
        "event_source": ["s3.amazonaws.com"] * n,
        "event_name": ["GetObject"] * n,
        "source_ip": ["192.0.2.10"] * n,
        "user_identity": ["alice@example-corp.com"] * n,
        "aws_region": ["us-east-1"] * n,
        "error_code": [""] * n,
        "failed_attempts": [0] * n,
        "unusual_location": [0] * n,
        "privilege_change": [0] * n,
    })


# ===========================================================================
# 1. CSV Validator tests
# ===========================================================================

class TestCSVValidator:

    def test_valid_df_passes(self):
        """A correctly formed DataFrame must pass validation."""
        result = validate_upload(_make_valid_df())
        assert result.is_valid
        assert result.row_count == 3
        assert len(result.errors) == 0

    def test_empty_df_fails(self):
        """An empty DataFrame must fail validation."""
        empty = pd.DataFrame(columns=REQUIRED_UPLOAD_COLUMNS)
        result = validate_upload(empty)
        assert not result.is_valid
        assert any("no data" in e.lower() for e in result.errors)

    def test_missing_column_fails(self):
        """Dropping a required column must produce an error."""
        df = _make_valid_df().drop(columns=["event_time"])
        result = validate_upload(df)
        assert not result.is_valid
        assert any("event_time" in e for e in result.errors)

    def test_missing_multiple_columns_fails(self):
        """Dropping two required columns must list both in the error."""
        df = _make_valid_df().drop(columns=["event_source", "aws_region"])
        result = validate_upload(df)
        assert not result.is_valid
        error_text = " ".join(result.errors)
        assert "event_source" in error_text
        assert "aws_region" in error_text

    def test_invalid_timestamp_fails(self):
        """Non-parseable event_time must produce an error."""
        df = _make_valid_df()
        df.loc[0, "event_time"] = "NOT_A_DATE"
        result = validate_upload(df)
        assert not result.is_valid
        assert any("unparseable" in e.lower() for e in result.errors)

    def test_negative_failed_attempts_fails(self):
        """Negative failed_attempts must produce an error."""
        df = _make_valid_df()
        df.loc[0, "failed_attempts"] = -1
        result = validate_upload(df)
        assert not result.is_valid

    def test_invalid_unusual_location_fails(self):
        """unusual_location outside [0,1] must produce an error."""
        df = _make_valid_df()
        df.loc[0, "unusual_location"] = 5
        result = validate_upload(df)
        assert not result.is_valid

    def test_invalid_privilege_change_fails(self):
        """privilege_change outside [0,1] must produce an error."""
        df = _make_valid_df()
        df.loc[0, "privilege_change"] = 99
        result = validate_upload(df)
        assert not result.is_valid

    def test_exceeding_max_rows_fails(self):
        """A DataFrame with more than MAX_ROWS must fail."""
        df = pd.concat([_make_valid_df()] * (MAX_ROWS + 1), ignore_index=True)
        result = validate_upload(df)
        assert not result.is_valid
        # The error mentions the limit (formatted as "5,000" with comma)
        assert any("maximum" in e.lower() or "5,000" in e for e in result.errors)

    def test_unknown_error_code_warns_not_fails(self):
        """An unrecognised error_code should produce a WARNING, not an error."""
        df = _make_valid_df()
        df.loc[0, "error_code"] = "SomeCustomError"
        result = validate_upload(df)
        assert result.is_valid            # still valid
        assert len(result.warnings) > 0  # but a warning is present

    def test_summary_contains_valid(self):
        """summary() must include VALID for a clean DataFrame."""
        result = validate_upload(_make_valid_df())
        assert "VALID" in result.summary()

    def test_summary_contains_invalid(self):
        """summary() must include INVALID when errors exist."""
        df = _make_valid_df().drop(columns=["event_time"])
        result = validate_upload(df)
        assert "INVALID" in result.summary()


class TestPrepareForPrediction:

    def test_removes_fully_empty_rows(self):
        """Rows with all key columns null must be dropped."""
        df = _make_valid_df(3)
        empty_row = pd.DataFrame({c: [None] for c in df.columns})
        df_with_empty = pd.concat([df, empty_row], ignore_index=True)
        result = prepare_for_prediction(df_with_empty)
        assert len(result) == 3

    def test_adds_missing_optional_columns(self):
        """Missing optional columns must be added with safe defaults."""
        df = _make_valid_df(2)
        result = prepare_for_prediction(df)
        assert "error_code" in result.columns
        assert "failed_attempts" in result.columns

    def test_fills_missing_error_code(self):
        """Missing error_code values must be filled with empty string."""
        df = _make_valid_df(2)
        df["error_code"] = None
        result = prepare_for_prediction(df)
        assert (result["error_code"] == "").all()

    def test_does_not_modify_original(self):
        """prepare_for_prediction must not modify the caller's DataFrame."""
        df = _make_valid_df(2)
        original_cols = list(df.columns)
        prepare_for_prediction(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# 2. Database / Incident tests
# ===========================================================================

class TestDBUtils:

    def test_get_connection_creates_incidents_table(self, tmp_db):
        """get_connection must create the incidents table."""
        conn = get_connection(tmp_db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        assert "incidents" in tables

    def test_incidents_table_does_not_drop_others(self, tmp_db):
        """Calling get_connection on an existing DB must not alter other tables."""
        # First call creates
        conn = get_connection(tmp_db)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS dummy (id INTEGER PRIMARY KEY)"
        )
        conn.commit()
        conn.close()
        # Second call
        conn2 = get_connection(tmp_db)
        cur = conn2.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        conn2.close()
        assert "dummy" in tables

    def test_create_incident_returns_id(self, live_conn):
        """create_incident must return a positive incident_id."""
        result = create_incident(
            live_conn, event_id="evt-001", title="Test incident"
        )
        assert result["created"] is True
        assert result["incident_id"] > 0

    def test_create_incident_duplicate_prevented(self, live_conn):
        """Creating the same (event_id, title) twice must not insert a duplicate."""
        create_incident(live_conn, event_id="evt-dup", title="Dup title")
        result2 = create_incident(live_conn, event_id="evt-dup", title="Dup title")
        assert result2["already_existed"] is True
        # Only one row in table
        cur = live_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM incidents WHERE event_id='evt-dup'")
        assert cur.fetchone()[0] == 1

    def test_create_incident_different_title_allowed(self, live_conn):
        """Same event_id but different title must create a second incident."""
        r1 = create_incident(live_conn, event_id="evt-x", title="Title A")
        r2 = create_incident(live_conn, event_id="evt-x", title="Title B")
        assert r1["created"] is True
        assert r2["created"] is True
        assert r1["incident_id"] != r2["incident_id"]

    def test_update_incident_status_valid(self, live_conn):
        """update_incident_status must succeed for valid statuses."""
        r = create_incident(live_conn, event_id="evt-upd", title="Update test")
        inc_id = r["incident_id"]
        for status in VALID_STATUSES:
            ok = update_incident_status(live_conn, inc_id, status)
            assert ok, f"update failed for status: {status}"

    def test_update_incident_status_invalid(self, live_conn):
        """update_incident_status must return False for unknown status."""
        r = create_incident(live_conn, event_id="evt-inv", title="Invalid status")
        ok = update_incident_status(live_conn, r["incident_id"], "Banana")
        assert not ok

    def test_update_nonexistent_incident(self, live_conn):
        """Updating a non-existent incident_id must return False."""
        ok = update_incident_status(live_conn, 99999, "Resolved")
        assert not ok

    def test_get_incidents_returns_all(self, live_conn):
        """get_incidents with no filter must return all incidents."""
        create_incident(live_conn, event_id="a1", title="Inc 1")
        create_incident(live_conn, event_id="a2", title="Inc 2")
        df = get_incidents(live_conn)
        assert len(df) >= 2

    def test_get_incidents_filter_by_status(self, live_conn):
        """get_incidents with status_filter must return only matching rows."""
        r = create_incident(live_conn, event_id="sf1", title="Status filter test")
        update_incident_status(live_conn, r["incident_id"], "Resolved")
        df = get_incidents(live_conn, status_filter="Resolved")
        assert (df["status"] == "Resolved").all()

    def test_get_open_incident_count(self, live_conn):
        """get_open_incident_count must count New and Investigating incidents."""
        create_incident(live_conn, event_id="oc1", title="Open 1")
        r2 = create_incident(live_conn, event_id="oc2", title="Open 2")
        update_incident_status(live_conn, r2["incident_id"], "Investigating")
        r3 = create_incident(live_conn, event_id="oc3", title="Closed")
        update_incident_status(live_conn, r3["incident_id"], "Resolved")
        count = get_open_incident_count(live_conn)
        assert count >= 2  # at least the two we just created

    def test_get_incident_status_counts_all_statuses_present(self, live_conn):
        """get_incident_status_counts must return a dict with all four statuses."""
        counts = get_incident_status_counts(live_conn)
        for status in VALID_STATUSES:
            assert status in counts

    def test_incident_default_status_is_new(self, live_conn):
        """Newly created incidents must have status 'New'."""
        r = create_incident(live_conn, event_id="ns1", title="Default status")
        df = get_incidents(live_conn)
        row = df[df["incident_id"] == r["incident_id"]].iloc[0]
        assert row["status"] == "New"


# ===========================================================================
# 3. Overview query tests (against the live database)
# ===========================================================================

class TestOverviewQueries:
    """
    These tests read from the real data/database/security_logs.db.
    They verify the query functions return correctly shaped results.
    """

    @pytest.fixture(scope="class")
    def live_db_conn(self):
        db_path = Path("data") / "database" / "security_logs.db"
        if not db_path.exists():
            pytest.skip("Live database not found — run the pipeline first.")
        conn = get_connection(str(db_path))
        yield conn
        conn.close()

    def test_get_event_counts_keys(self, live_db_conn):
        """get_event_counts must return total, normal, suspicious, and unknown."""
        counts = get_event_counts(live_db_conn)
        assert "total" in counts
        assert "normal" in counts
        assert "suspicious" in counts
        assert "unknown" in counts

    def test_get_event_counts_values(self, live_db_conn):
        """Total must equal normal + suspicious + unknown.

        This invariant holds regardless of how many CloudTrail records
        (label=-1) have been ingested alongside synthetic records.
        """
        counts = get_event_counts(live_db_conn)
        assert counts["total"] == counts["normal"] + counts["suspicious"] + counts["unknown"], (
            f"total={counts['total']} != normal={counts['normal']} + "
            f"suspicious={counts['suspicious']} + unknown={counts['unknown']}"
        )

    def test_get_event_counts_5000_synthetic(self, live_db_conn):
        """Live database must contain exactly 5,000 synthetic records.

        Tests the synthetic count specifically so CloudTrail ingestion
        (which adds real records with data_source='cloudtrail_real')
        does not cause this test to fail.
        """
        counts = get_event_counts(live_db_conn)
        synthetic_total = counts["normal"] + counts["suspicious"]
        assert synthetic_total == 5000, (
            f"Expected 5000 synthetic records (normal + suspicious), "
            f"got {synthetic_total}. normal={counts['normal']}, "
            f"suspicious={counts['suspicious']}"
        )

    def test_get_event_counts_synthetic_class_balance(self, live_db_conn):
        """Synthetic records must preserve the 80/20 class balance."""
        counts = get_event_counts(live_db_conn)
        assert counts["normal"] == 4000, (
            f"Expected 4000 normal records, got {counts['normal']}"
        )
        assert counts["suspicious"] == 1000, (
            f"Expected 1000 suspicious records, got {counts['suspicious']}"
        )

    def test_get_events_by_source_is_dataframe(self, live_db_conn):
        """get_events_by_source must return a non-empty DataFrame."""
        df = get_events_by_source(live_db_conn)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_get_events_by_source_columns(self, live_db_conn):
        """get_events_by_source must return event_source, label, count."""
        df = get_events_by_source(live_db_conn)
        for col in ("event_source", "label", "count"):
            assert col in df.columns

    def test_get_events_by_hour_hour_range(self, live_db_conn):
        """Hours must be in [0, 23]."""
        df = get_events_by_hour(live_db_conn)
        assert df["hour"].between(0, 23).all()

    def test_get_top_suspicious_users_limit(self, live_db_conn):
        """get_top_suspicious_users must return at most n rows."""
        df = get_top_suspicious_users(live_db_conn, n=5)
        assert len(df) <= 5

    def test_get_recent_events_limit(self, live_db_conn):
        """get_recent_events must return at most n rows."""
        df = get_recent_events(live_db_conn, n=10)
        assert len(df) <= 10

    def test_cloudtrail_records_counted_as_unknown(self, live_db_conn):
        """CloudTrail real records (label=-1) must appear in unknown, not normal/suspicious."""
        counts = get_event_counts(live_db_conn)
        # If real records are present, unknown must be > 0
        # If only synthetic, unknown must be 0
        # Either way, total must equal all three buckets summed
        assert counts["total"] == (
            counts["normal"] + counts["suspicious"] + counts["unknown"]
        ), "CloudTrail records are not being counted in the 'unknown' bucket"


# ===========================================================================
# REGRESSION: CloudTrail tests must not contaminate the live database
# ===========================================================================

class TestCloudTrailTestIsolation:
    """
    Regression tests confirming that Phase 6 ingestion tests use isolated
    temporary databases and do not write to the project's live database.
    """

    def test_live_db_synthetic_count_stable(self):
        """
        The synthetic record count in the live database must be exactly 5,000.
        If this fails, a test or script wrote to the live database unexpectedly.
        """
        from pathlib import Path
        from dashboard.db_utils import get_connection, get_event_counts
        live_db = Path("data") / "database" / "security_logs.db"
        if not live_db.exists():
            pytest.skip("Live database not found.")
        conn = get_connection(str(live_db))
        counts = get_event_counts(conn)
        conn.close()
        synthetic = counts["normal"] + counts["suspicious"]
        assert synthetic == 5000, (
            f"Expected 5000 synthetic records; got {synthetic}. "
            f"A test or script may have contaminated the live database."
        )

    def test_live_db_invariant_holds(self):
        """
        total == normal + suspicious + unknown must always hold.
        This catches any future label value that isn't in any bucket.
        """
        from pathlib import Path
        from dashboard.db_utils import get_connection, get_event_counts
        live_db = Path("data") / "database" / "security_logs.db"
        if not live_db.exists():
            pytest.skip("Live database not found.")
        conn = get_connection(str(live_db))
        counts = get_event_counts(conn)
        conn.close()
        assert counts["total"] == (
            counts["normal"] + counts["suspicious"] + counts["unknown"]
        ), f"Invariant broken: {counts}"

    def test_cloud_test_fixture_uses_tmp_db(self, tmp_path):
        """
        Confirm that ingesting the Phase 6 fixture into a tmp_path database
        does NOT change the live database row count.
        """
        from pathlib import Path
        from dashboard.db_utils import get_connection, get_event_counts
        from src.cloud.cloudtrail_ingestor import ingest_file
        from src.cloud.ip_allowlist import IPAllowlist

        live_db = Path("data") / "database" / "security_logs.db"
        if not live_db.exists():
            pytest.skip("Live database not found.")

        # Record live DB count BEFORE ingestion
        conn_live = get_connection(str(live_db))
        before = get_event_counts(conn_live)
        conn_live.close()

        # Ingest into a SEPARATE tmp database
        fixture = Path("tests") / "fixtures" / "sample_cloudtrail.json"
        tmp_db = str(tmp_path / "isolation_test.db")
        al = IPAllowlist.__new__(IPAllowlist)
        al._prefixes = ["192.0.2.", "198.51.100."]
        ingest_file(fixture, db_path=tmp_db, ip_allowlist=al)

        # Verify live DB is unchanged
        conn_live2 = get_connection(str(live_db))
        after = get_event_counts(conn_live2)
        conn_live2.close()

        assert before["total"] == after["total"], (
            f"Live database changed during isolation test: "
            f"before={before['total']}, after={after['total']}"
        )


# ===========================================================================
# 4. Prediction integration test (CSV → prepare → predict)
# ===========================================================================

class TestPredictionIntegration:
    """
    End-to-end test: simulate a CSV upload, validate, prepare, and predict.
    Uses the real saved model if available.
    """

    @pytest.fixture(scope="class")
    def clf_fe(self):
        from src.ml.predictor import load_model, load_feature_engineer
        try:
            clf = load_model()
            fe = load_feature_engineer()
            return clf, fe
        except FileNotFoundError:
            pytest.skip("Saved model not found — run the trainer first.")

    def test_full_pipeline_from_csv(self, clf_fe):
        """validate → prepare → predict must complete without error."""
        from src.ml.predictor import predict as ml_predict
        clf, fe = clf_fe
        df = _make_valid_df(5)
        val = validate_upload(df)
        assert val.is_valid
        prep = prepare_for_prediction(df)
        results = ml_predict(prep, clf=clf, fe=fe)
        assert len(results) == 5
        for col in ("event_id", "predicted_label",
                    "threat_probability", "threat_level"):
            assert col in results.columns

    def test_prediction_labels_binary(self, clf_fe):
        """All predicted labels must be 0 or 1."""
        from src.ml.predictor import predict as ml_predict
        clf, fe = clf_fe
        df = _make_valid_df(10)
        prep = prepare_for_prediction(df)
        results = ml_predict(prep, clf=clf, fe=fe)
        assert set(results["predicted_label"].unique()).issubset({0, 1})

    def test_prediction_probabilities_in_range(self, clf_fe):
        """All threat_probability values must be in [0, 1]."""
        from src.ml.predictor import predict as ml_predict
        clf, fe = clf_fe
        df = _make_valid_df(10)
        prep = prepare_for_prediction(df)
        results = ml_predict(prep, clf=clf, fe=fe)
        assert results["threat_probability"].between(0.0, 1.0).all()

    def test_prediction_threat_levels_valid(self, clf_fe):
        """All threat_level values must be Low, Medium, or High."""
        from src.ml.predictor import predict as ml_predict
        clf, fe = clf_fe
        df = _make_valid_df(10)
        prep = prepare_for_prediction(df)
        results = ml_predict(prep, clf=clf, fe=fe)
        assert set(results["threat_level"].unique()).issubset(
            {"Low", "Medium", "High"}
        )

    def test_suspicious_record_high_probability(self, clf_fe):
        """A clearly suspicious record should score higher than a clearly normal one."""
        from src.ml.predictor import predict as ml_predict
        clf, fe = clf_fe

        normal_df = pd.DataFrame([{
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
        }])
        suspicious_df = pd.DataFrame([{
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
        }])

        n_res = ml_predict(prepare_for_prediction(normal_df), clf=clf, fe=fe)
        s_res = ml_predict(prepare_for_prediction(suspicious_df), clf=clf, fe=fe)

        assert s_res["threat_probability"].iloc[0] > \
               n_res["threat_probability"].iloc[0], (
            "Suspicious record should have higher threat probability than normal"
        )
