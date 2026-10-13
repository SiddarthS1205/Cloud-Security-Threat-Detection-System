"""
test_security.py
================
Phase 5 — Security Engineering: tests for input validation,
sanitisation, SQL safety, and incident security controls.

Run with:    pytest tests/test_security.py -v
Combined:    pytest tests/ -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.security.input_validator import (
    sanitise_title, sanitise_description, sanitise_event_id,
    validate_severity, validate_status, validate_incident_id,
    clamp_limit, contains_sql_metacharacters, flag_suspicious_input,
    VALID_SEVERITIES, VALID_STATUSES,
    MAX_TITLE_LEN, MAX_DESCRIPTION_LEN, MAX_EVENT_ID_LEN,
    MAX_QUERY_LIMIT, MIN_QUERY_LIMIT,
)
from dashboard.csv_validator import (
    validate_upload, prepare_for_prediction,
    REQUIRED_UPLOAD_COLUMNS, MAX_ROWS,
)
from dashboard.db_utils import (
    get_connection, create_incident, update_incident_status,
    VALID_STATUSES as DB_VALID_STATUSES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "event_time":        [f"2024-06-{10+i:02d}T10:00:00Z" for i in range(n)],
        "event_source":      ["s3.amazonaws.com"] * n,
        "event_name":        ["GetObject"] * n,
        "source_ip":         ["192.0.2.10"] * n,
        "user_identity":     ["alice@example-corp.com"] * n,
        "aws_region":        ["us-east-1"] * n,
        "error_code":        [""] * n,
        "failed_attempts":   [0] * n,
        "unusual_location":  [0] * n,
        "privilege_change":  [0] * n,
    })


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "sec_test.db")


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


# ===========================================================================
# 1. Input sanitiser tests
# ===========================================================================

class TestSanitiseTitle:
    def test_normal_title_unchanged(self):
        assert sanitise_title("Brute force attempt") == "Brute force attempt"

    def test_strips_whitespace(self):
        assert sanitise_title("  Alert  ") == "Alert"

    def test_truncates_to_max(self):
        result = sanitise_title("A" * (MAX_TITLE_LEN + 50))
        assert len(result) == MAX_TITLE_LEN

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            sanitise_title("   ")

    def test_sql_accepted_as_text(self):
        # SQL meta-characters are stored as text; safety via parameterised queries
        t = sanitise_title("Test'; DROP TABLE incidents; --")
        assert "DROP TABLE" in t

    def test_unicode_accepted(self):
        assert "sécurité" in sanitise_title("Alerte de sécurité")


class TestSanitiseDescription:
    def test_empty_allowed(self):
        assert sanitise_description("") == ""

    def test_strips_whitespace(self):
        assert sanitise_description("  notes  ") == "notes"

    def test_truncates_to_max(self):
        assert len(sanitise_description("B" * (MAX_DESCRIPTION_LEN + 100))) == MAX_DESCRIPTION_LEN

    def test_newlines_preserved(self):
        d = "Line 1\nLine 2"
        assert sanitise_description(d) == d


class TestSanitiseEventId:
    def test_uuid_unchanged(self):
        uid = "cc530e36-addc-3e13-ab3b-4d37560c95ee"
        assert sanitise_event_id(uid) == uid

    def test_empty_defaults_to_manual(self):
        assert sanitise_event_id("") == "manual"
        assert sanitise_event_id("   ") == "manual"

    def test_strips_whitespace(self):
        assert sanitise_event_id("  abc-123  ") == "abc-123"

    def test_truncates_to_max(self):
        assert len(sanitise_event_id("x" * (MAX_EVENT_ID_LEN + 50))) == MAX_EVENT_ID_LEN


class TestValidateSeverity:
    def test_all_valid_accepted(self):
        for s in VALID_SEVERITIES:
            assert validate_severity(s) == s

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid severity"):
            validate_severity("EXTREME")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            validate_severity("")

    def test_lowercase_rejected(self):
        with pytest.raises(ValueError):
            validate_severity("high")


class TestValidateStatus:
    def test_all_valid_accepted(self):
        for s in VALID_STATUSES:
            assert validate_status(s) == s

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            validate_status("Pending")

    def test_sql_injection_as_status_raises(self):
        with pytest.raises(ValueError):
            validate_status("New'; DROP TABLE incidents; --")


class TestValidateIncidentId:
    def test_positive_int(self):
        assert validate_incident_id(1) == 1
        assert validate_incident_id(9999) == 9999

    def test_string_int(self):
        assert validate_incident_id("42") == 42

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            validate_incident_id(0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="greater than 0"):
            validate_incident_id(-5)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            validate_incident_id("abc")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            validate_incident_id(None)

    def test_float_truncated(self):
        assert validate_incident_id(3.7) == 3


class TestClampLimit:
    def test_normal_value_unchanged(self):
        assert clamp_limit(10) == 10

    def test_zero_clamped_to_min(self):
        assert clamp_limit(0) == MIN_QUERY_LIMIT

    def test_negative_clamped_to_min(self):
        assert clamp_limit(-100) == MIN_QUERY_LIMIT

    def test_oversized_clamped_to_max(self):
        assert clamp_limit(999999) == MAX_QUERY_LIMIT

    def test_none_returns_default(self):
        assert clamp_limit(None, default=20) == 20

    def test_string_int(self):
        assert clamp_limit("5") == 5

    def test_non_numeric_returns_default(self):
        assert clamp_limit("abc", default=15) == 15


class TestSQLMetaCharacters:
    def test_clean_string_false(self):
        assert not contains_sql_metacharacters("normal text")

    def test_single_quote_detected(self):
        assert contains_sql_metacharacters("O'Brien")

    def test_semicolon_detected(self):
        assert contains_sql_metacharacters("value; DROP TABLE")

    def test_full_injection_detected(self):
        assert contains_sql_metacharacters("'; DROP TABLE incidents; --")

    def test_flag_returns_message_for_suspicious(self):
        msg = flag_suspicious_input("title", "'; DROP TABLE --")
        assert msg is not None and "title" in msg

    def test_flag_returns_none_for_clean(self):
        assert flag_suspicious_input("title", "Normal title") is None


# ===========================================================================
# 2. CSV security tests
# ===========================================================================

class TestCSVSecurityValidation:

    def test_sql_in_event_name_accepted_as_data(self):
        """SQL injection in column values must be accepted (not blocked).
        Safety comes from parameterised queries downstream."""
        df = _valid_df()
        df.loc[0, "event_name"] = "'; DROP TABLE raw_logs; --"
        result = validate_upload(df)
        assert result.is_valid

    def test_sql_in_user_identity_accepted(self):
        df = _valid_df()
        df.loc[0, "user_identity"] = "admin' OR '1'='1"
        assert validate_upload(df).is_valid

    def test_binary_like_content_no_crash(self):
        """Unusual bytes in string fields must not crash the validator."""
        df = _valid_df()
        df.loc[0, "user_identity"] = "user\x00\x01@example.com"
        result = validate_upload(df)
        assert isinstance(result.is_valid, bool)

    def test_oversized_upload_rejected(self):
        df = pd.concat([_valid_df()] * (MAX_ROWS + 1), ignore_index=True)
        result = validate_upload(df)
        assert not result.is_valid
        assert any("maximum" in e.lower() or "5,000" in e for e in result.errors)

    def test_negative_failed_attempts_rejected(self):
        df = _valid_df()
        df["failed_attempts"] = -99
        assert not validate_upload(df).is_valid

    def test_out_of_range_unusual_location_rejected(self):
        df = _valid_df()
        df["unusual_location"] = 2
        assert not validate_upload(df).is_valid

    def test_string_in_numeric_column_rejected(self):
        df = _valid_df()
        df["failed_attempts"] = "many"
        assert not validate_upload(df).is_valid

    def test_all_columns_missing_rejected(self):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        assert not validate_upload(df).is_valid

    def test_valid_csv_passes_security_checks(self):
        result = validate_upload(_valid_df(10))
        assert result.is_valid
        assert len(result.errors) == 0


class TestPrepareForPredictionSecurity:

    def test_sql_values_preserved_as_text(self):
        """SQL patterns must survive prepare_for_prediction unchanged."""
        df = _valid_df()
        malicious = "'; DROP TABLE raw_logs; --"
        df.loc[0, "event_name"] = malicious
        result = prepare_for_prediction(df)
        assert result.loc[0, "event_name"] == malicious

    def test_does_not_execute_content(self):
        """No exception means no execution; values treated as plain strings."""
        df = _valid_df()
        df.loc[0, "event_name"] = "__import__('os').system('echo test')"
        # Must not raise and must not execute
        result = prepare_for_prediction(df)
        assert isinstance(result, pd.DataFrame)


# ===========================================================================
# 3. Incident management security tests
# ===========================================================================

class TestIncidentSecurityControls:

    def test_long_title_stored_truncated(self, conn):
        long_title = "T" * (MAX_TITLE_LEN + 100)
        result = create_incident(conn, event_id="x1", title=long_title)
        assert result["created"] is True
        cur = conn.cursor()
        cur.execute("SELECT title FROM incidents WHERE incident_id = ?",
                    (result["incident_id"],))
        stored = cur.fetchone()[0]
        assert len(stored) <= MAX_TITLE_LEN

    def test_empty_title_returns_error(self, conn):
        result = create_incident(conn, event_id="x2", title="   ")
        assert result.get("error") is not None or result["created"] is False

    def test_invalid_severity_returns_error(self, conn):
        result = create_incident(conn, event_id="x3", title="Test", severity="NUCLEAR")
        assert result.get("error") is not None
        assert result["created"] is False

    def test_sql_in_title_stored_not_executed(self, conn):
        malicious = "'; DROP TABLE incidents; --"
        result = create_incident(conn, event_id="x4", title=malicious)
        assert result["created"] is True
        # Verify the incidents table still exists
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE name='incidents'")
        assert cur.fetchone() is not None

    def test_invalid_status_rejected(self, conn):
        r = create_incident(conn, event_id="y1", title="Status test")
        assert not update_incident_status(conn, r["incident_id"], "HACKED")

    def test_sql_injection_as_status_rejected(self, conn):
        r = create_incident(conn, event_id="y2", title="SQL status")
        ok = update_incident_status(
            conn, r["incident_id"], "New'; DROP TABLE incidents; --"
        )
        assert ok is False
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE name='incidents'")
        assert cur.fetchone() is not None

    def test_string_incident_id_rejected(self, conn):
        assert not update_incident_status(conn, "not-an-id", "Resolved")

    def test_negative_incident_id_rejected(self, conn):
        assert not update_incident_status(conn, -1, "Resolved")

    def test_zero_incident_id_rejected(self, conn):
        assert not update_incident_status(conn, 0, "Resolved")

    def test_duplicate_not_inserted(self, conn):
        create_incident(conn, event_id="dup1", title="Dup test")
        r2 = create_incident(conn, event_id="dup1", title="Dup test")
        assert r2["already_existed"] is True
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM incidents WHERE event_id='dup1'"
        )
        assert cur.fetchone()[0] == 1

    def test_all_valid_statuses_accepted(self, conn):
        r = create_incident(conn, event_id="vs1", title="Valid statuses")
        for status in DB_VALID_STATUSES:
            ok = update_incident_status(conn, r["incident_id"], status)
            assert ok is True, f"Expected True for status: {status}"


# ===========================================================================
# 4. Allowlist completeness
# ===========================================================================

class TestAllowlists:

    def test_valid_severities_complete(self):
        for s in ("Low", "Medium", "High", "Critical"):
            assert s in VALID_SEVERITIES

    def test_valid_statuses_complete(self):
        for s in ("New", "Investigating", "Resolved", "False Positive"):
            assert s in VALID_STATUSES

    def test_db_valid_statuses_match_validator(self):
        assert set(DB_VALID_STATUSES) == set(VALID_STATUSES)
