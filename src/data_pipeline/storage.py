"""
storage.py
==========
SQLite read/write for raw_logs and processed_features tables.

Key design decisions:
  - Uses INSERT OR REPLACE (upsert) so the pipeline can be re-run
    safely without creating duplicate rows.  The PRIMARY KEY on
    event_id ensures exactly one row per event in each table.
  - Creates the database file and tables automatically if they do
    not yet exist.
  - Never uses string formatting to build SQL — all values are
    passed as parameterised queries to prevent SQL injection.
  - Provides separate write functions for raw_logs and
    processed_features, and read functions for both.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL: table creation SQL
# ---------------------------------------------------------------------------

_CREATE_RAW_LOGS = """
CREATE TABLE IF NOT EXISTS raw_logs (
    event_id          TEXT PRIMARY KEY,
    event_time        TEXT NOT NULL,
    event_source      TEXT NOT NULL,
    event_name        TEXT NOT NULL,
    source_ip         TEXT NOT NULL,
    user_identity     TEXT NOT NULL,
    aws_region        TEXT NOT NULL,
    error_code        TEXT,
    failed_attempts   INTEGER DEFAULT 0,
    unusual_location  INTEGER DEFAULT 0,
    privilege_change  INTEGER DEFAULT 0,
    label             INTEGER NOT NULL,
    data_source       TEXT DEFAULT 'synthetic',
    ingested_at       TEXT NOT NULL
);
"""

_CREATE_PROCESSED_FEATURES = """
CREATE TABLE IF NOT EXISTS processed_features (
    event_id              TEXT PRIMARY KEY,
    hour_of_day           INTEGER,
    day_of_week           INTEGER,
    is_weekend            INTEGER,
    is_error              INTEGER,
    failed_attempts       INTEGER,
    unusual_location      INTEGER,
    privilege_change      INTEGER,
    is_iam_event          INTEGER,
    is_high_risk_action   INTEGER,
    event_source_enc      INTEGER,
    event_name_enc        INTEGER,
    region_enc            INTEGER,
    label                 INTEGER,
    FOREIGN KEY (event_id) REFERENCES raw_logs(event_id)
);
"""

_CREATE_PIPELINE_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        TEXT NOT NULL,
    rows_raw      INTEGER,
    rows_features INTEGER,
    notes         TEXT
);
"""

# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at db_path and ensure all
    required tables exist.

    Parameters
    ----------
    db_path : str
        Absolute or relative path to the .db file.

    Returns
    -------
    sqlite3.Connection
        An open connection with foreign-key enforcement enabled.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")

    # Create tables if they don't exist
    conn.execute(_CREATE_RAW_LOGS)
    conn.execute(_CREATE_PROCESSED_FEATURES)
    conn.execute(_CREATE_PIPELINE_RUNS)
    conn.commit()

    logger.info("[storage] Database ready at: %s", path.resolve())
    return conn


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------

def write_raw_logs(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Upsert cleaned records into the raw_logs table.

    Uses INSERT OR REPLACE so re-running the pipeline never creates
    duplicate rows.

    Parameters
    ----------
    conn : sqlite3.Connection
    df   : pd.DataFrame
        Must contain all raw_logs columns.

    Returns
    -------
    int
        Number of rows written.
    """
    required = [
        "event_id", "event_time", "event_source", "event_name",
        "source_ip", "user_identity", "aws_region", "error_code",
        "failed_attempts", "unusual_location", "privilege_change",
        "label", "data_source", "ingested_at",
    ]
    _check_columns(df, required, "raw_logs")

    rows = df[required].to_dict(orient="records")

    sql = """
        INSERT OR REPLACE INTO raw_logs (
            event_id, event_time, event_source, event_name,
            source_ip, user_identity, aws_region, error_code,
            failed_attempts, unusual_location, privilege_change,
            label, data_source, ingested_at
        ) VALUES (
            :event_id, :event_time, :event_source, :event_name,
            :source_ip, :user_identity, :aws_region, :error_code,
            :failed_attempts, :unusual_location, :privilege_change,
            :label, :data_source, :ingested_at
        )
    """
    conn.executemany(sql, rows)
    conn.commit()
    logger.info("[storage] Wrote %d rows to raw_logs.", len(rows))
    return len(rows)


def write_processed_features(
    conn: sqlite3.Connection,
    X: pd.DataFrame,
    y: pd.Series,
) -> int:
    """
    Upsert ML-ready features into the processed_features table.

    Parameters
    ----------
    conn : sqlite3.Connection
    X    : pd.DataFrame
        Feature matrix — must include 'event_id' and all feature columns.
    y    : pd.Series
        Label series — index must align with X.

    Returns
    -------
    int
        Number of rows written.
    """
    feature_cols = [
        "event_id",
        "hour_of_day", "day_of_week", "is_weekend",
        "is_error", "failed_attempts",
        "unusual_location", "privilege_change",
        "is_iam_event", "is_high_risk_action",
        "event_source_enc", "event_name_enc", "region_enc",
    ]
    _check_columns(X, feature_cols, "processed_features")

    df_out = X[feature_cols].copy()
    df_out["label"] = y.values

    rows = df_out.to_dict(orient="records")

    sql = """
        INSERT OR REPLACE INTO processed_features (
            event_id,
            hour_of_day, day_of_week, is_weekend,
            is_error, failed_attempts,
            unusual_location, privilege_change,
            is_iam_event, is_high_risk_action,
            event_source_enc, event_name_enc, region_enc,
            label
        ) VALUES (
            :event_id,
            :hour_of_day, :day_of_week, :is_weekend,
            :is_error, :failed_attempts,
            :unusual_location, :privilege_change,
            :is_iam_event, :is_high_risk_action,
            :event_source_enc, :event_name_enc, :region_enc,
            :label
        )
    """
    conn.executemany(sql, rows)
    conn.commit()
    logger.info("[storage] Wrote %d rows to processed_features.", len(rows))
    return len(rows)


def log_pipeline_run(
    conn: sqlite3.Connection,
    rows_raw: int,
    rows_features: int,
    notes: str = "",
) -> None:
    """Record metadata about the pipeline run for audit purposes."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO pipeline_runs (run_at, rows_raw, rows_features, notes) "
        "VALUES (?, ?, ?, ?)",
        (now, rows_raw, rows_features, notes),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Read functions
# ---------------------------------------------------------------------------

def read_raw_logs(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all records from raw_logs into a DataFrame."""
    return pd.read_sql_query("SELECT * FROM raw_logs", conn)


def read_processed_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all records from processed_features into a DataFrame."""
    return pd.read_sql_query("SELECT * FROM processed_features", conn)


def get_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return row counts for both main tables."""
    cursor = conn.cursor()
    counts = {}
    for table in ("raw_logs", "processed_features"):
        cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 — table name is hardcoded
        counts[table] = cursor.fetchone()[0]
    return counts


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: list[str], table: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing columns required for '{table}': {missing}"
        )
