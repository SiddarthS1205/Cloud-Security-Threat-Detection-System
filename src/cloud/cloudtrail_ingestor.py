"""
cloudtrail_ingestor.py
======================
Phase 6 — Cloud Engineering: orchestrates the full CloudTrail ingestion pipeline.

Pipeline order
--------------
  parse_file()          -> list[raw_event_dict]
  anonymiser.anonymise()-> list[anon_event_dict]   (PII removed BEFORE storage)
  map_events()          -> list[flat_schema_dict]  (label=-1 for unknown)
  cleaner.clean()       -> DataFrame               (normalises types/timestamps)
  write_raw_logs()      -> SQLite raw_logs table

WHAT THIS MODULE DOES NOT DO
-----------------------------
- Does not connect to AWS or make any network call.
- Does not retrain or re-fit the ML model.
- Does not store the original-to-pseudonym mapping.
- Does not calculate precision, recall, F1, or accuracy.
- Does not call the strict validator (validator.py) — that module enforces
  the synthetic dataset's narrow allowlists. The ingestor uses a permissive
  check: only eventTime, eventSource, eventName must be non-null.

LABEL HANDLING
--------------
All ingested real CloudTrail events receive label = LABEL_UNKNOWN (-1).
This sentinel cannot be mistaken for normal (0) or suspicious (1).
The ML predictor discards the label value entirely; it only uses features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.cloud.cloudtrail_anonymiser import CloudTrailAnonymiser
from src.cloud.cloudtrail_mapper import map_events, MappingReport
from src.cloud.cloudtrail_parser import parse_file, ParseResult, LABEL_UNKNOWN
from src.cloud.ip_allowlist import IPAllowlist
from src.data_pipeline.cleaner import clean
from src.data_pipeline.storage import get_connection, write_raw_logs

logger = logging.getLogger(__name__)

DB_PATH = Path("data") / "database" / "security_logs.db"


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------

@dataclass
class IngestionReport:
    """Summary of one ingestion run."""
    source_file: str = ""
    parse: ParseResult = field(default_factory=ParseResult)
    mapping: MappingReport = field(default_factory=MappingReport)
    rows_written: int = 0
    rows_skipped_cleaning: int = 0
    label_sentinel: int = LABEL_UNKNOWN   # always -1; documented here explicitly

    def summary(self) -> str:
        lines = [
            "=== CloudTrail Ingestion Report ===",
            f"  Source file      : {self.source_file}",
            f"  Records parsed   : {self.parse.total_records}",
            f"  Parse errors     : {len(self.parse.parse_errors)}",
            f"  Records mapped   : {self.mapping.total_mapped}",
            f"  Mapping skipped  : {self.mapping.total_skipped}",
            f"  Rows written     : {self.rows_written}",
            f"  Label assigned   : {self.label_sentinel} (unknown — not 0 or 1)",
            "",
            "  NOTICE: All records are labelled as UNKNOWN (-1).",
            "  They cannot be interpreted as confirmed normal events.",
            "  Precision, recall, F1, and accuracy are NOT calculated",
            "  because ground-truth labels are not available.",
        ]
        for err in self.parse.parse_errors:
            lines.append(f"  PARSE ERROR: {err}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_file(
    json_path: str | Path,
    db_path: str | Path = DB_PATH,
    ip_allowlist: IPAllowlist | None = None,
) -> IngestionReport:
    """
    Ingest one CloudTrail JSON file into the project's SQLite database.

    Parameters
    ----------
    json_path    : path to CloudTrail JSON file (local disk, no network)
    db_path      : path to SQLite database (default: data/database/security_logs.db)
    ip_allowlist : IPAllowlist instance (default: creates one with default config)

    Returns
    -------
    IngestionReport with full statistics
    """
    report = IngestionReport(source_file=str(json_path))

    if ip_allowlist is None:
        ip_allowlist = IPAllowlist()

    # Step 1: Parse
    report.parse = parse_file(json_path)
    if not report.parse.records:
        logger.warning("[ingestor] No records parsed from %s.", json_path)
        return report

    # Step 2: Anonymise — PII removed BEFORE any persistence
    anonymiser = CloudTrailAnonymiser(ip_allowlist=ip_allowlist)
    anon_events = [anonymiser.anonymise(rec) for rec in report.parse.records]
    logger.info(
        "[ingestor] Anonymised %d events (%d unique identities pseudonymised).",
        len(anon_events),
        anonymiser.unique_users_seen,
    )

    # Step 3: Map to flat schema (label=-1 for all real records)
    mapped_records, report.mapping = map_events(anon_events)
    if not mapped_records:
        logger.warning("[ingestor] No records survived mapping.")
        return report

    # Step 4: Clean (normalise timestamps, fill error_code, cast types)
    df_raw = pd.DataFrame(mapped_records)
    df_clean = clean(df_raw)

    # Verify label sentinel was preserved through cleaning
    unique_labels = df_clean["label"].unique().tolist()
    if any(lbl not in (LABEL_UNKNOWN,) for lbl in unique_labels):
        logger.warning(
            "[ingestor] Unexpected label values after cleaning: %s. "
            "Expected only %d (unknown sentinel).",
            unique_labels, LABEL_UNKNOWN,
        )

    # Step 5: Write to SQLite
    conn = get_connection(str(db_path))
    report.rows_written = write_raw_logs(conn, df_clean)
    conn.close()

    logger.info("[ingestor] Ingestion complete.\n%s", report.summary())
    return report


# ---------------------------------------------------------------------------
# Convenience: ingest and predict
# ---------------------------------------------------------------------------

def ingest_and_predict(
    json_path: str | Path,
    db_path: str | Path = DB_PATH,
    ip_allowlist: IPAllowlist | None = None,
    clf=None,
    fe=None,
) -> tuple[IngestionReport, pd.DataFrame]:
    """
    Ingest a CloudTrail JSON file, then score the ingested records.

    Anonymisation and ingestion happen before any ML scoring.
    Labels remain -1 throughout; the predictor discards them.

    Parameters
    ----------
    json_path    : path to CloudTrail JSON file
    db_path      : SQLite database path
    ip_allowlist : IPAllowlist instance
    clf          : loaded classifier (or None to load from disk)
    fe           : loaded FeatureEngineer (or None to load from disk)

    Returns
    -------
    (IngestionReport, predictions_DataFrame)
    """
    from src.ml.predictor import load_model, load_feature_engineer, predict

    if clf is None:
        clf = load_model()
    if fe is None:
        fe = load_feature_engineer()

    report = ingest_file(json_path, db_path=db_path, ip_allowlist=ip_allowlist)

    if report.rows_written == 0:
        return report, pd.DataFrame()

    # Load the just-ingested records for prediction
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    df_ingested = pd.read_sql_query(
        "SELECT * FROM raw_logs WHERE data_source = 'cloudtrail_real' "
        "ORDER BY ingested_at DESC LIMIT ?",
        conn,
        params=(report.rows_written,),
    )
    conn.close()

    if df_ingested.empty:
        return report, pd.DataFrame()

    predictions = predict(df_ingested, clf=clf, fe=fe)
    return report, predictions
