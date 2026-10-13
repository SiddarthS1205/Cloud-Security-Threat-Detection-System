"""
test_cloud.py
=============
Phase 6 — Cloud Engineering: unit tests for the CloudTrail integration pipeline.

All tests use only the hand-crafted fixture file
(tests/fixtures/sample_cloudtrail.json) — ZERO live AWS calls, ZERO
network connections, NO boto3 imports in test code.

Test classes
------------
  TestCloudTrailParser       — parses fixture; handles missing/corrupt files
  TestAnonymiser             — pseudonymises users; redacts IPs; drops PII
  TestSchemaMapper           — maps CT fields to flat schema; label=-1 sentinel
  TestPrivilegeChangeDeriver — derives privilege_change from event_name
  TestUnusualLocationDeriver — IP allowlist logic; empty list conservative default
  TestLabelSentinel          — label=-1 survives cleaner and predictor
  TestIngestionPipeline      — full pipeline produces DB rows with correct metadata
  TestUnseenCategories       — novel event_source/event_name handled without crash
  TestReporter               — report explicitly states metrics not computed
  TestRealVsSyntheticLabel   — data_source distinguishes real from synthetic

Run with:
    pytest tests/test_cloud.py -v

Full regression run:
    pytest tests/ -v
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_cloudtrail.json"

from src.cloud.cloudtrail_parser import parse_file, LABEL_UNKNOWN, _REQUIRED_FIELDS
from src.cloud.cloudtrail_anonymiser import CloudTrailAnonymiser
from src.cloud.cloudtrail_mapper import map_event, map_events, DATA_SOURCE_REAL
from src.cloud.ip_allowlist import IPAllowlist
from src.cloud.cloudtrail_reporter import generate_report
from src.data_pipeline.generator import PRIVILEGE_CHANGE_ACTIONS
from src.data_pipeline.cleaner import clean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_event(**kwargs) -> dict:
    """Return a minimal valid raw CT event, with optional overrides."""
    base = {
        "eventTime": "2024-06-15T10:05:12Z",
        "eventSource": "s3.amazonaws.com",
        "eventName": "GetObject",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "192.0.2.10",
        "userIdentity": {"type": "IAMUser", "userName": "alice", "accountId": "111122223333"},
        "errorCode": None,
        "eventID": "test-evt-001",
    }
    base.update(kwargs)
    return base


def _anonymise_one(event: dict, allowlist: IPAllowlist | None = None) -> dict:
    anon = CloudTrailAnonymiser(ip_allowlist=allowlist)
    return anon.anonymise(event)


def _map_one(event: dict, allowlist: IPAllowlist | None = None) -> dict | None:
    anon_event = _anonymise_one(event, allowlist=allowlist)
    return map_event(anon_event)


@pytest.fixture(scope="module")
def fixture_records():
    result = parse_file(FIXTURE_PATH)
    return result.records


@pytest.fixture(scope="module")
def configured_allowlist():
    """Allowlist matching the fixture's known-safe IPs."""
    al = IPAllowlist.__new__(IPAllowlist)
    al._prefixes = ["192.0.2.", "198.51.100."]
    return al


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "ct_test.db")


# ===========================================================================
# 1. Parser tests
# ===========================================================================

class TestCloudTrailParser:

    def test_fixture_parses_15_records(self):
        """The fixture file must contain exactly 15 records."""
        result = parse_file(FIXTURE_PATH)
        assert result.total_records == 15, (
            f"Expected 15 records, got {result.total_records}"
        )

    def test_parse_succeeds(self):
        result = parse_file(FIXTURE_PATH)
        assert result.succeeded

    def test_records_are_dicts(self, fixture_records):
        for rec in fixture_records:
            assert isinstance(rec, dict)

    def test_required_fields_present(self, fixture_records):
        """Every parsed record must have the three required fields."""
        for rec in fixture_records:
            for field in _REQUIRED_FIELDS:
                assert rec.get(field), (
                    f"Record {rec.get('eventID')} missing required field: {field}"
                )

    def test_event_ids_unique(self, fixture_records):
        ids = [r.get("eventID") for r in fixture_records if r.get("eventID")]
        assert len(ids) == len(set(ids)), "Duplicate eventIDs in fixture"

    def test_missing_file_returns_error(self, tmp_path):
        result = parse_file(tmp_path / "nonexistent.json")
        assert not result.succeeded
        assert result.parse_errors

    def test_empty_file_returns_error(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("", encoding="utf-8")
        result = parse_file(empty)
        assert not result.succeeded

    def test_malformed_json_returns_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json at all !!!}", encoding="utf-8")
        result = parse_file(bad)
        assert not result.succeeded or result.total_records == 0

    def test_records_missing_required_fields_are_skipped(self, tmp_path):
        """Records with missing eventSource should be skipped."""
        data = {"Records": [
            {"eventTime": "2024-01-01T00:00:00Z", "eventName": "GetObject"},  # missing eventSource
            {"eventTime": "2024-01-01T00:01:00Z", "eventSource": "s3.amazonaws.com", "eventName": "PutObject"},
        ]}
        f = tmp_path / "partial.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = parse_file(f)
        assert result.total_records == 1
        assert result.skipped_records == 1

    def test_line_delimited_format_parsed(self, tmp_path):
        """Line-delimited JSON format should be parsed correctly."""
        events = [
            {"eventTime": "2024-01-01T00:00:00Z", "eventSource": "s3.amazonaws.com", "eventName": "GetObject"},
            {"eventTime": "2024-01-01T00:01:00Z", "eventSource": "ec2.amazonaws.com", "eventName": "DescribeInstances"},
        ]
        f = tmp_path / "lines.json"
        f.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
        result = parse_file(f)
        assert result.total_records == 2


# ===========================================================================
# 2. Anonymiser tests
# ===========================================================================

class TestAnonymiser:

    def test_username_not_in_output(self, fixture_records):
        """Original usernames must not appear in the user_identity field."""
        anon = CloudTrailAnonymiser()
        # Collect all original usernames from the fixture
        original_names = set()
        for raw in fixture_records:
            name = (raw.get("userIdentity") or {}).get("userName", "")
            if name:
                original_names.add(name)

        for raw in fixture_records:
            out = anon.anonymise(raw)
            pseudonym = out.get("userIdentityPseudonym", "")
            for name in original_names:
                assert name != pseudonym, (
                    f"Original username {name!r} used as pseudonym"
                )
            # Pseudonym must match user_N pattern
            assert pseudonym.startswith("user_"), (
                f"Pseudonym {pseudonym!r} does not follow user_N pattern"
            )

    def test_account_id_not_in_output(self, fixture_records):
        """AWS account ID must not appear in anonymised output."""
        anon = CloudTrailAnonymiser()
        for raw in fixture_records:
            out = anon.anonymise(raw)
            acct_id = (raw.get("userIdentity") or {}).get("accountId", "")
            if acct_id:
                assert acct_id not in str(out), (
                    f"Account ID {acct_id!r} leaked into output"
                )

    def test_source_ip_redacted(self, fixture_records):
        """sourceIPAddress must be replaced with REDACTED."""
        anon = CloudTrailAnonymiser()
        for raw in fixture_records:
            out = anon.anonymise(raw)
            assert out.get("sourceIPAddress") == "REDACTED"

    def test_original_ip_not_in_output(self, fixture_records):
        """Original IP address must not appear anywhere in the output."""
        anon = CloudTrailAnonymiser()
        for raw in fixture_records:
            raw_ip = raw.get("sourceIPAddress", "")
            out = anon.anonymise(raw)
            # Original IP must not be in output values
            for key, val in out.items():
                if key != "unusual_location":
                    assert raw_ip not in str(val), (
                        f"Original IP {raw_ip!r} leaked in field {key!r}"
                    )

    def test_request_parameters_dropped(self, fixture_records):
        """requestParameters must not appear in anonymised output."""
        anon = CloudTrailAnonymiser()
        for raw in fixture_records:
            out = anon.anonymise(raw)
            assert "requestParameters" not in out

    def test_response_elements_dropped(self, fixture_records):
        """responseElements must not appear in anonymised output."""
        anon = CloudTrailAnonymiser()
        for raw in fixture_records:
            out = anon.anonymise(raw)
            assert "responseElements" not in out

    def test_pseudonyms_are_sequential(self, fixture_records):
        """Pseudonyms must follow user_1, user_2, ... pattern."""
        anon = CloudTrailAnonymiser()
        pseudonyms = set()
        for raw in fixture_records:
            out = anon.anonymise(raw)
            p = out.get("userIdentityPseudonym", "")
            if p:
                pseudonyms.add(p)
        for p in pseudonyms:
            assert p.startswith("user_"), f"Pseudonym {p!r} does not start with 'user_'"
            suffix = p[5:]
            assert suffix.isdigit(), f"Pseudonym {p!r} suffix is not a number"

    def test_same_user_gets_same_pseudonym(self):
        """The same username must always get the same pseudonym within a session."""
        anon = CloudTrailAnonymiser()
        e1 = _make_minimal_event(
            eventID="e1",
            userIdentity={"type": "IAMUser", "userName": "alice", "accountId": "111"},
        )
        e2 = _make_minimal_event(
            eventID="e2",
            userIdentity={"type": "IAMUser", "userName": "alice", "accountId": "111"},
        )
        out1 = anon.anonymise(e1)
        out2 = anon.anonymise(e2)
        assert out1["userIdentityPseudonym"] == out2["userIdentityPseudonym"]

    def test_different_users_get_different_pseudonyms(self):
        """Different usernames must receive different pseudonyms."""
        anon = CloudTrailAnonymiser()
        ea = _make_minimal_event(
            userIdentity={"type": "IAMUser", "userName": "alice", "accountId": "111"},
        )
        eb = _make_minimal_event(
            userIdentity={"type": "IAMUser", "userName": "bob", "accountId": "111"},
        )
        pa = anon.anonymise(ea)["userIdentityPseudonym"]
        pb = anon.anonymise(eb)["userIdentityPseudonym"]
        assert pa != pb

    def test_null_username_gets_pseudonym(self):
        """Events with null userName (AssumedRole) must still get a pseudonym."""
        anon = CloudTrailAnonymiser()
        ev = _make_minimal_event(
            userIdentity={"type": "AssumedRole", "userName": None, "accountId": "111",
                          "principalId": "AROA123:session"}
        )
        out = anon.anonymise(ev)
        assert out.get("userIdentityPseudonym", "").startswith("user_")

    def test_unusual_location_computed_before_redaction(self, configured_allowlist):
        """unusual_location must be set in the anonymised output."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        # Known-safe IP (in allowlist) → unusual=0
        safe_ev = _make_minimal_event(sourceIPAddress="192.0.2.10")
        out_safe = anon.anonymise(safe_ev)
        assert out_safe["unusual_location"] == 0
        assert out_safe["sourceIPAddress"] == "REDACTED"

        # Unknown IP → unusual=1
        anon2 = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        unknown_ev = _make_minimal_event(sourceIPAddress="203.0.113.99")
        out_unknown = anon2.anonymise(unknown_ev)
        assert out_unknown["unusual_location"] == 1
        assert out_unknown["sourceIPAddress"] == "REDACTED"


# ===========================================================================
# 3. Schema mapper tests
# ===========================================================================

class TestSchemaMapper:

    REQUIRED_SCHEMA_COLS = [
        "event_id", "event_time", "event_source", "event_name",
        "source_ip", "user_identity", "aws_region", "error_code",
        "failed_attempts", "unusual_location", "privilege_change",
        "label", "data_source", "ingested_at",
    ]

    def test_all_required_columns_present(self, fixture_records, configured_allowlist):
        """Every mapped record must have all required schema columns."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            anon_event = anon.anonymise(raw)
            mapped = map_event(anon_event)
            assert mapped is not None, f"map_event returned None for {raw.get('eventID')}"
            for col in self.REQUIRED_SCHEMA_COLS:
                assert col in mapped, (
                    f"Column {col!r} missing in mapped record for {raw.get('eventID')}"
                )

    def test_label_is_sentinel_minus_one(self, fixture_records, configured_allowlist):
        """label must be LABEL_UNKNOWN (-1) for all real records."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            anon_event = anon.anonymise(raw)
            mapped = map_event(anon_event)
            assert mapped is not None
            assert mapped["label"] == LABEL_UNKNOWN, (
                f"Expected label={LABEL_UNKNOWN}, got {mapped['label']} "
                f"for event {raw.get('eventID')}"
            )

    def test_label_is_not_zero(self, fixture_records, configured_allowlist):
        """label must NOT be 0 — 0 means confirmed normal, not unknown."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            anon_event = anon.anonymise(raw)
            mapped = map_event(anon_event)
            assert mapped is not None
            assert mapped["label"] != 0, (
                "label=0 would mean 'confirmed normal' — real events must use sentinel -1"
            )

    def test_data_source_is_cloudtrail_real(self, fixture_records, configured_allowlist):
        """data_source must be 'cloudtrail_real' for all real records."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            anon_event = anon.anonymise(raw)
            mapped = map_event(anon_event)
            assert mapped["data_source"] == DATA_SOURCE_REAL

    def test_event_source_lowercased(self, fixture_records, configured_allowlist):
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            out = anon.anonymise(raw)
            mapped = map_event(out)
            assert mapped["event_source"] == mapped["event_source"].lower()

    def test_event_name_lowercased(self, fixture_records, configured_allowlist):
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            out = anon.anonymise(raw)
            mapped = map_event(out)
            assert mapped["event_name"] == mapped["event_name"].lower()

    def test_failed_attempts_is_zero(self, fixture_records, configured_allowlist):
        """failed_attempts must be 0 in raw (re-derived by FeatureEngineer)."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            out = anon.anonymise(raw)
            mapped = map_event(out)
            assert mapped["failed_attempts"] == 0

    def test_source_ip_is_redacted(self, fixture_records, configured_allowlist):
        """source_ip must be REDACTED (anonymisation happened before mapping)."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            out = anon.anonymise(raw)
            mapped = map_event(out)
            assert mapped["source_ip"] == "REDACTED"

    def test_missing_required_field_returns_none(self):
        """map_event must return None if a required field is missing."""
        anon = CloudTrailAnonymiser()
        # Missing eventName
        ev = _make_minimal_event(eventName=None)
        ev.pop("eventName", None)
        out = anon.anonymise(ev)
        result = map_event(out)
        assert result is None

    def test_missing_aws_region_uses_default(self):
        """Missing awsRegion should default to 'us-east-1', not crash."""
        anon = CloudTrailAnonymiser()
        ev = _make_minimal_event()
        ev.pop("awsRegion", None)
        out = anon.anonymise(ev)
        mapped = map_event(out)
        assert mapped is not None
        assert mapped["aws_region"] == "us-east-1"

    def test_null_error_code_becomes_empty_string(self):
        """errorCode=null must become empty string."""
        anon = CloudTrailAnonymiser()
        ev = _make_minimal_event(errorCode=None)
        out = anon.anonymise(ev)
        mapped = map_event(out)
        assert mapped["error_code"] == ""

    def test_15_records_all_map_successfully(self, fixture_records, configured_allowlist):
        """All 15 fixture records must map successfully."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        anon_events = [anon.anonymise(r) for r in fixture_records]
        mapped, report = map_events(anon_events)
        assert report.total_mapped == 15
        assert report.total_skipped == 0


# ===========================================================================
# 4. privilege_change derivation
# ===========================================================================

class TestPrivilegeChangeDeriver:

    @pytest.mark.parametrize("event_name", sorted(PRIVILEGE_CHANGE_ACTIONS))
    def test_privilege_actions_set_flag(self, event_name):
        """Each action in PRIVILEGE_CHANGE_ACTIONS must produce privilege_change=1."""
        ev = _make_minimal_event(eventName=event_name)
        mapped = _map_one(ev)
        assert mapped is not None
        assert mapped["privilege_change"] == 1, (
            f"Expected privilege_change=1 for {event_name!r}"
        )

    def test_normal_action_does_not_set_flag(self):
        """Non-privilege event_name must produce privilege_change=0."""
        for action in ["GetObject", "DescribeInstances", "GetUser", "DescribeKey"]:
            ev = _make_minimal_event(eventName=action)
            mapped = _map_one(ev)
            assert mapped["privilege_change"] == 0, (
                f"Expected privilege_change=0 for {action!r}"
            )

    def test_fixture_privilege_events_flagged(self, fixture_records, configured_allowlist):
        """ct-fix-011 (AttachUserPolicy) and ct-fix-014 (CreateUser) must have flag=1."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        priv_ids = {"ct-fix-011", "ct-fix-014"}
        for raw in fixture_records:
            if raw.get("eventID") in priv_ids:
                out = anon.anonymise(raw)
                mapped = map_event(out)
                assert mapped["privilege_change"] == 1, (
                    f"Expected privilege_change=1 for {raw['eventID']} "
                    f"({raw['eventName']})"
                )

    def test_fixture_normal_events_not_flagged(self, fixture_records, configured_allowlist):
        """ct-fix-001..010 (normal events) must have privilege_change=0."""
        normal_ids = {f"ct-fix-{i:03d}" for i in range(1, 11)}
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            if raw.get("eventID") in normal_ids:
                out = anon.anonymise(raw)
                mapped = map_event(out)
                assert mapped["privilege_change"] == 0, (
                    f"Expected privilege_change=0 for {raw['eventID']} "
                    f"({raw['eventName']})"
                )


# ===========================================================================
# 5. unusual_location derivation
# ===========================================================================

class TestUnusualLocationDeriver:

    def test_empty_allowlist_returns_zero(self):
        """Conservative default: empty allowlist returns unusual_location=0."""
        al = IPAllowlist.__new__(IPAllowlist)
        al._prefixes = []
        assert al.is_unusual("203.0.113.99") == 0

    def test_known_ip_returns_zero(self, configured_allowlist):
        assert configured_allowlist.is_unusual("192.0.2.10") == 0
        assert configured_allowlist.is_unusual("198.51.100.5") == 0

    def test_unknown_ip_returns_one(self, configured_allowlist):
        assert configured_allowlist.is_unusual("203.0.113.99") == 1
        assert configured_allowlist.is_unusual("10.0.0.1") == 1

    def test_service_endpoint_returns_zero(self, configured_allowlist):
        """AWS service endpoints are not real IPs — always not-unusual."""
        assert configured_allowlist.is_unusual("lambda.amazonaws.com") == 0
        assert configured_allowlist.is_unusual("ecs.amazonaws.com") == 0

    def test_none_ip_returns_zero(self, configured_allowlist):
        assert configured_allowlist.is_unusual(None) == 0

    def test_empty_string_ip_returns_zero(self, configured_allowlist):
        assert configured_allowlist.is_unusual("") == 0

    def test_is_configured_false_when_empty(self):
        al = IPAllowlist.__new__(IPAllowlist)
        al._prefixes = []
        assert not al.is_configured

    def test_is_configured_true_when_prefixes_present(self, configured_allowlist):
        assert configured_allowlist.is_configured

    def test_fixture_known_ips_not_unusual(self, fixture_records, configured_allowlist):
        """Events ct-fix-001..010 use 192.0.2.x or 198.51.100.x — not unusual."""
        normal_ids = {f"ct-fix-{i:03d}" for i in range(1, 11)}
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            if raw.get("eventID") in normal_ids:
                out = anon.anonymise(raw)
                assert out["unusual_location"] == 0, (
                    f"Expected unusual_location=0 for {raw['eventID']} "
                    f"(IP: {raw.get('sourceIPAddress')})"
                )

    def test_fixture_suspicious_ips_are_unusual(self, fixture_records, configured_allowlist):
        """Events ct-fix-011..015 use 203.0.113.x — unusual."""
        suspicious_ids = {f"ct-fix-{i:03d}" for i in range(11, 16)}
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        for raw in fixture_records:
            if raw.get("eventID") in suspicious_ids:
                out = anon.anonymise(raw)
                assert out["unusual_location"] == 1, (
                    f"Expected unusual_location=1 for {raw['eventID']} "
                    f"(IP: {raw.get('sourceIPAddress')})"
                )


# ===========================================================================
# 6. Label sentinel — cleaner and predictor safety
# ===========================================================================

class TestLabelSentinel:

    def test_label_minus_one_preserved_through_cleaner(self, fixture_records,
                                                        configured_allowlist):
        """cleaner.clean() must preserve label=-1 (not convert to 0)."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        anon_events = [anon.anonymise(r) for r in fixture_records]
        mapped, _ = map_events(anon_events)
        df = pd.DataFrame(mapped)
        df_clean = clean(df)
        assert (df_clean["label"] == LABEL_UNKNOWN).all(), (
            f"cleaner converted label=-1. Got: {df_clean['label'].unique()}"
        )

    def test_label_minus_one_not_zero(self, fixture_records, configured_allowlist):
        """Confirm the sentinel is -1 not 0 after cleaning."""
        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        anon_events = [anon.anonymise(r) for r in fixture_records]
        mapped, _ = map_events(anon_events)
        df = pd.DataFrame(mapped)
        df_clean = clean(df)
        assert not (df_clean["label"] == 0).any(), (
            "label=0 found — real events must not be labelled as normal"
        )

    def test_predictor_discards_label_safely(self, fixture_records, configured_allowlist):
        """predictor.predict() must work without error when label=-1."""
        from src.ml.predictor import load_model, load_feature_engineer, predict
        try:
            clf = load_model()
            fe = load_feature_engineer()
        except FileNotFoundError:
            pytest.skip("Saved model not found — run the trainer first.")

        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        anon_events = [anon.anonymise(r) for r in fixture_records]
        mapped, _ = map_events(anon_events)
        df = pd.DataFrame(mapped)
        df_clean = clean(df)

        # predict() calls fe.transform() which extracts and discards y
        result = predict(df_clean, clf=clf, fe=fe)
        assert len(result) == len(df_clean)
        assert "predicted_label" in result.columns
        assert "threat_probability" in result.columns
        # predicted_label must only contain 0 or 1 — not the sentinel -1
        assert set(result["predicted_label"].unique()).issubset({0, 1})


# ===========================================================================
# 7. Full ingestion pipeline
# ===========================================================================

class TestIngestionPipeline:

    def test_ingest_writes_correct_row_count(self, tmp_db, configured_allowlist):
        """ingest_file must write all 15 fixture events to raw_logs."""
        from src.cloud.cloudtrail_ingestor import ingest_file
        report = ingest_file(FIXTURE_PATH, db_path=tmp_db,
                             ip_allowlist=configured_allowlist)
        assert report.rows_written == 15, (
            f"Expected 15 rows written, got {report.rows_written}"
        )

    def test_ingest_sets_data_source(self, tmp_db, configured_allowlist):
        """All ingested rows must have data_source='cloudtrail_real'."""
        from src.cloud.cloudtrail_ingestor import ingest_file
        ingest_file(FIXTURE_PATH, db_path=tmp_db,
                    ip_allowlist=configured_allowlist)
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT data_source FROM raw_logs")
        sources = {r[0] for r in cur.fetchall()}
        conn.close()
        assert sources == {DATA_SOURCE_REAL}

    def test_ingest_label_sentinel_stored(self, tmp_db, configured_allowlist):
        """All ingested rows must have label=-1 in the database."""
        from src.cloud.cloudtrail_ingestor import ingest_file
        ingest_file(FIXTURE_PATH, db_path=tmp_db,
                    ip_allowlist=configured_allowlist)
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT label FROM raw_logs")
        labels = {r[0] for r in cur.fetchall()}
        conn.close()
        assert labels == {LABEL_UNKNOWN}, (
            f"Expected only label={LABEL_UNKNOWN}, got {labels}"
        )

    def test_ingest_no_pii_in_database(self, tmp_db, configured_allowlist):
        """Original usernames and account IDs must not appear in user_identity column."""
        from src.cloud.cloudtrail_ingestor import ingest_file
        with open(FIXTURE_PATH, encoding="utf-8") as f:
            raw_json = json.load(f)
        original_names = {
            rec.get("userIdentity", {}).get("userName", "")
            for rec in raw_json.get("Records", [])
            if rec.get("userIdentity", {}).get("userName")
        }
        original_acct_ids = {
            rec.get("userIdentity", {}).get("accountId", "")
            for rec in raw_json.get("Records", [])
            if rec.get("userIdentity", {}).get("accountId")
        }

        ingest_file(FIXTURE_PATH, db_path=tmp_db,
                    ip_allowlist=configured_allowlist)
        conn = sqlite3.connect(tmp_db)
        # Check user_identity column specifically (not the whole CSV)
        identities = pd.read_sql_query(
            "SELECT user_identity FROM raw_logs", conn
        )["user_identity"].tolist()
        source_ips = pd.read_sql_query(
            "SELECT source_ip FROM raw_logs", conn
        )["source_ip"].tolist()
        conn.close()

        for identity in identities:
            for name in original_names:
                assert identity != name, (
                    f"Original username {name!r} stored as user_identity"
                )
            # Must follow user_N pattern
            assert str(identity).startswith("user_"), (
                f"user_identity {identity!r} is not a pseudonym"
            )

        # All source IPs must be REDACTED
        for ip in source_ips:
            assert ip == "REDACTED", f"Non-redacted IP found: {ip!r}"

        # Account IDs must not appear in user_identity
        for identity in identities:
            for acct_id in original_acct_ids:
                if acct_id:
                    assert acct_id not in identity, (
                        f"Account ID {acct_id!r} found in user_identity: {identity!r}"
                    )

    def test_ingest_is_safe_to_rerun(self, tmp_db, configured_allowlist):
        """Running ingest twice on the same file must not duplicate rows."""
        from src.cloud.cloudtrail_ingestor import ingest_file
        ingest_file(FIXTURE_PATH, db_path=tmp_db,
                    ip_allowlist=configured_allowlist)
        ingest_file(FIXTURE_PATH, db_path=tmp_db,
                    ip_allowlist=configured_allowlist)
        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM raw_logs")
        count = cur.fetchone()[0]
        conn.close()
        assert count == 15, (
            f"Expected 15 rows after two ingestion runs, got {count} (upsert failed)"
        )


# ===========================================================================
# 8. Unseen category handling
# ===========================================================================

class TestUnseenCategories:

    def test_novel_event_source_no_crash(self):
        """An event_source not seen during training must not crash the predictor."""
        from src.ml.predictor import load_model, load_feature_engineer, predict
        try:
            clf = load_model()
            fe = load_feature_engineer()
        except FileNotFoundError:
            pytest.skip("Saved model not found — run the trainer first.")

        ev = _make_minimal_event(eventSource="lambda.amazonaws.com")
        anon = CloudTrailAnonymiser()
        out = anon.anonymise(ev)
        mapped = map_event(out)
        df = pd.DataFrame([mapped])
        df_clean = clean(df)
        result = predict(df_clean, clf=clf, fe=fe)
        assert len(result) == 1
        assert result["predicted_label"].iloc[0] in (0, 1)

    def test_novel_event_name_no_crash(self):
        """An event_name not seen during training must not crash the predictor."""
        from src.ml.predictor import load_model, load_feature_engineer, predict
        try:
            clf = load_model()
            fe = load_feature_engineer()
        except FileNotFoundError:
            pytest.skip("Saved model not found — run the trainer first.")

        ev = _make_minimal_event(eventName="InvokeFunction")
        anon = CloudTrailAnonymiser()
        out = anon.anonymise(ev)
        mapped = map_event(out)
        df = pd.DataFrame([mapped])
        df_clean = clean(df)
        result = predict(df_clean, clf=clf, fe=fe)
        assert len(result) == 1

    def test_novel_region_no_crash(self):
        """An aws_region not seen during training must not crash the predictor."""
        from src.ml.predictor import load_model, load_feature_engineer, predict
        try:
            clf = load_model()
            fe = load_feature_engineer()
        except FileNotFoundError:
            pytest.skip("Saved model not found — run the trainer first.")

        ev = _make_minimal_event(awsRegion="sa-east-1")
        anon = CloudTrailAnonymiser()
        out = anon.anonymise(ev)
        mapped = map_event(out)
        df = pd.DataFrame([mapped])
        df_clean = clean(df)
        result = predict(df_clean, clf=clf, fe=fe)
        assert len(result) == 1

    def test_only_known_features_passed_to_model(self, fixture_records,
                                                   configured_allowlist):
        """The feature matrix passed to the classifier must match FEATURE_COLUMNS exactly."""
        from src.ml.predictor import load_model, load_feature_engineer
        from src.data_pipeline.features import FEATURE_COLUMNS
        try:
            clf = load_model()
            fe = load_feature_engineer()
        except FileNotFoundError:
            pytest.skip("Saved model not found — run the trainer first.")

        anon = CloudTrailAnonymiser(ip_allowlist=configured_allowlist)
        anon_events = [anon.anonymise(r) for r in fixture_records]
        mapped, _ = map_events(anon_events)
        df = pd.DataFrame(mapped)
        df_clean = clean(df)
        X_raw, _ = fe.transform(df_clean)
        X = X_raw.drop(columns=["event_id"])
        assert list(X.columns) == FEATURE_COLUMNS


# ===========================================================================
# 9. Reporter tests
# ===========================================================================

class TestReporter:

    @pytest.fixture
    def sample_predictions(self):
        return pd.DataFrame({
            "event_id": [f"ct-fix-{i:03d}" for i in range(1, 16)],
            "predicted_label": [0]*10 + [1]*5,
            "threat_probability": [0.05, 0.08, 0.10, 0.12, 0.07,
                                   0.09, 0.11, 0.06, 0.13, 0.08,
                                   0.82, 0.90, 0.75, 0.88, 0.65],
            "threat_level": ["Low"]*10 + ["High"]*4 + ["Medium"]*1,
        })

    def test_report_written_to_file(self, tmp_path, sample_predictions):
        out = tmp_path / "ct_eval.txt"
        generate_report(sample_predictions, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_report_states_labels_unavailable(self, tmp_path, sample_predictions):
        """Report must explicitly state ground-truth labels are unavailable."""
        out = tmp_path / "ct_eval.txt"
        content = generate_report(sample_predictions, output_path=out)
        assert "ground-truth labels are not available" in content.lower(), (
            "Report must state ground-truth labels are not available"
        )

    def test_report_explains_metrics_not_computed(self, tmp_path, sample_predictions):
        """Report must explain that classification metrics are not computed."""
        out = tmp_path / "ct_eval.txt"
        content = generate_report(sample_predictions, output_path=out)
        # Join lines to handle phrase wrapping across line breaks
        content_joined = " ".join(content.upper().split())
        assert "NOT CALCULATED" in content_joined, (
            "Report must explicitly state that classification metrics are NOT CALCULATED"
        )

    def test_report_does_not_claim_accuracy(self, tmp_path, sample_predictions):
        """Report must not state a numeric accuracy, precision, recall, or F1 score."""
        out = tmp_path / "ct_eval.txt"
        content = generate_report(sample_predictions, output_path=out)
        # Check no numeric metric claim like "accuracy: 0.95" or "F1: 0.92"
        import re
        # Pattern: metric name followed by colon and a float (e.g. "accuracy: 0.95")
        metric_claim = re.search(
            r'\b(accuracy|precision|recall|f1)\s*[:=]\s*0\.\d+',
            content, re.IGNORECASE
        )
        assert metric_claim is None, (
            f"Report contains a numeric metric claim: {metric_claim.group()!r}"
        )

    def test_report_label_sentinel_documented(self, tmp_path, sample_predictions):
        """Report must document that label -1 means unknown, not normal."""
        out = tmp_path / "ct_eval.txt"
        content = generate_report(sample_predictions, output_path=out)
        assert "-1" in content and "unknown" in content.lower(), (
            "Report must document that label=-1 means unknown"
        )

    def test_report_with_empty_predictions(self, tmp_path):
        """Report must handle empty predictions gracefully."""
        out = tmp_path / "ct_eval_empty.txt"
        content = generate_report(pd.DataFrame(), output_path=out)
        assert "0 records" in content.lower() or "no predictions" in content.lower()


# ===========================================================================
# 10. Real vs synthetic data distinction
# ===========================================================================

class TestRealVsSyntheticLabel:

    def test_synthetic_data_source_unchanged(self):
        """Existing synthetic records in the live DB must still have data_source='synthetic'."""
        live_db = Path("data") / "database" / "security_logs.db"
        if not live_db.exists():
            pytest.skip("Live database not found.")
        conn = sqlite3.connect(str(live_db))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM raw_logs WHERE data_source='synthetic'")
        count = cur.fetchone()[0]
        conn.close()
        assert count == 5000, (
            f"Expected 5000 synthetic records, got {count}"
        )

    def test_real_and_synthetic_can_coexist(self, tmp_db, configured_allowlist):
        """Both data_source values can coexist in the same database."""
        from src.cloud.cloudtrail_ingestor import ingest_file
        from src.data_pipeline.generator import generate_dataset
        from src.data_pipeline.validator import validate
        from src.data_pipeline.cleaner import clean as pipeline_clean
        from src.data_pipeline.storage import get_connection, write_raw_logs

        # Write synthetic records
        df_syn = generate_dataset(seed=42).head(10)
        df_syn_valid, _ = validate(df_syn)
        df_syn_clean = pipeline_clean(df_syn_valid)
        conn = get_connection(tmp_db)
        write_raw_logs(conn, df_syn_clean)
        conn.close()

        # Write real (fixture) records
        ingest_file(FIXTURE_PATH, db_path=tmp_db,
                    ip_allowlist=configured_allowlist)

        conn = sqlite3.connect(tmp_db)
        cur = conn.cursor()
        cur.execute(
            "SELECT data_source, COUNT(*) FROM raw_logs GROUP BY data_source"
        )
        dist = dict(cur.fetchall())
        conn.close()

        assert "synthetic" in dist
        assert DATA_SOURCE_REAL in dist
        assert dist["synthetic"] == 10
        assert dist[DATA_SOURCE_REAL] == 15
