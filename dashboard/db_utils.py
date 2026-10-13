"""
db_utils.py
===========
Dashboard-specific SQLite helpers for Phase 4.

Adds an 'incidents' table to the existing security_logs.db database.
All existing tables (raw_logs, processed_features, pipeline_runs) are
left completely untouched.

Incident lifecycle
------------------
  New  →  Investigating  →  Resolved
                         →  False Positive

Duplicate prevention
--------------------
  An incident is identified by (event_id, title). Attempting to create
  one that already exists returns the existing record without inserting
  a duplicate row.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.security.input_validator import (
    clamp_limit,
    sanitise_description,
    sanitise_event_id,
    sanitise_title,
    validate_severity,
    validate_status,
    validate_incident_id,
    flag_suspicious_input,
    VALID_STATUSES as _VALIDATED_STATUSES,
)

logger = logging.getLogger(__name__)

DB_PATH = Path("data") / "database" / "security_logs.db"

VALID_STATUSES = ("New", "Investigating", "Resolved", "False Positive")

_CREATE_INCIDENTS = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT,
    title         TEXT NOT NULL,
    description   TEXT,
    severity      TEXT DEFAULT 'Medium',
    status        TEXT DEFAULT 'New',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""

_CREATE_INCIDENT_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_incidents_event_title
ON incidents (event_id, title);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open the existing database and ensure the incidents table exists."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute(_CREATE_INCIDENTS)
    try:
        conn.execute(_CREATE_INCIDENT_INDEX)
    except sqlite3.OperationalError:
        pass  # index already exists
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Overview queries
# ---------------------------------------------------------------------------

def get_event_counts(conn: sqlite3.Connection) -> dict:
    """
    Return event counts from raw_logs, broken down by label class.

    Returns
    -------
    dict with keys:
        total      — all rows regardless of data_source or label
        normal     — rows with label = 0 (confirmed normal, synthetic only)
        suspicious — rows with label = 1 (confirmed suspicious, synthetic only)
        unknown    — rows with label = -1 (unlabelled real CloudTrail events)

    Invariant: total == normal + suspicious + unknown

    Note: 'unknown' will be 0 for a synthetic-only database. It becomes
    non-zero when real CloudTrail records are ingested (Phase 6).
    """
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM raw_logs")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM raw_logs WHERE label = 0")
    normal = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM raw_logs WHERE label = 1")
    suspicious = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM raw_logs WHERE label = -1")
    unknown = cur.fetchone()[0]
    return {
        "total": total,
        "normal": normal,
        "suspicious": suspicious,
        "unknown": unknown,
    }


def get_events_by_source(conn: sqlite3.Connection) -> pd.DataFrame:
    """Count events grouped by event_source and label."""
    return pd.read_sql_query(
        "SELECT event_source, label, COUNT(*) as count "
        "FROM raw_logs GROUP BY event_source, label "
        "ORDER BY count DESC",
        conn,
    )


def get_events_by_hour(conn: sqlite3.Connection) -> pd.DataFrame:
    """Count events by hour of day for timeline chart."""
    return pd.read_sql_query(
        "SELECT CAST(SUBSTR(event_time, 12, 2) AS INTEGER) AS hour, "
        "label, COUNT(*) AS count "
        "FROM raw_logs GROUP BY hour, label ORDER BY hour",
        conn,
    )


def get_top_suspicious_users(conn: sqlite3.Connection, n: int = 10) -> pd.DataFrame:
    """Return the top-N users with the most suspicious events.
    n is clamped to [1, 500] to prevent unbounded queries.
    """
    safe_n = clamp_limit(n, default=10)
    return pd.read_sql_query(
        "SELECT user_identity, COUNT(*) as suspicious_count "
        "FROM raw_logs WHERE label = 1 "
        "GROUP BY user_identity ORDER BY suspicious_count DESC LIMIT ?",
        conn,
        params=(safe_n,),
    )


def get_recent_events(conn: sqlite3.Connection, n: int = 20) -> pd.DataFrame:
    """Return the most recently ingested events.
    n is clamped to [1, 500] to prevent unbounded queries.
    """
    safe_n = clamp_limit(n, default=20)
    return pd.read_sql_query(
        "SELECT event_id, event_time, event_source, event_name, "
        "user_identity, aws_region, error_code, label "
        "FROM raw_logs ORDER BY ingested_at DESC LIMIT ?",
        conn,
        params=(safe_n,),
    )


# ---------------------------------------------------------------------------
# Incident management
# ---------------------------------------------------------------------------

def get_incidents(
    conn: sqlite3.Connection,
    status_filter: str | None = None,
) -> pd.DataFrame:
    """
    Load all incidents, optionally filtered by status.

    Parameters
    ----------
    conn          : open database connection
    status_filter : one of VALID_STATUSES, or None for all
    """
    if status_filter and status_filter in VALID_STATUSES:
        return pd.read_sql_query(
            "SELECT * FROM incidents WHERE status = ? "
            "ORDER BY created_at DESC",
            conn,
            params=(status_filter,),
        )
    return pd.read_sql_query(
        "SELECT * FROM incidents ORDER BY created_at DESC", conn
    )


def get_open_incident_count(conn: sqlite3.Connection) -> int:
    """Return the number of New or Investigating incidents."""
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM incidents WHERE status IN ('New','Investigating')"
    )
    return cur.fetchone()[0]


def create_incident(
    conn: sqlite3.Connection,
    event_id: str,
    title: str,
    description: str = "",
    severity: str = "Medium",
) -> dict:
    """
    Create a new incident, or return the existing one if (event_id, title)
    already exists.

    Inputs are sanitised and validated before any database operation.
    All SQL uses parameterised placeholders — no string interpolation.

    Returns
    -------
    dict with keys: incident_id, created, already_existed
    """
    # Sanitise and validate all inputs before touching the database
    try:
        clean_title = sanitise_title(title)
        clean_event_id = sanitise_event_id(event_id)
        clean_description = sanitise_description(description)
        clean_severity = validate_severity(severity)
    except ValueError as exc:
        return {
            "incident_id": None,
            "created": False,
            "already_existed": False,
            "error": str(exc),
        }

    # Audit log for inputs containing SQL meta-characters
    for field, val in [("title", clean_title), ("event_id", clean_event_id)]:
        warning = flag_suspicious_input(field, val)
        if warning:
            logger.warning("[security] %s", warning)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.cursor()

    # Duplicate check — parameterised
    cur.execute(
        "SELECT incident_id FROM incidents WHERE event_id = ? AND title = ?",
        (clean_event_id, clean_title),
    )
    existing = cur.fetchone()
    if existing:
        return {
            "incident_id": existing[0],
            "created": False,
            "already_existed": True,
        }

    cur.execute(
        "INSERT INTO incidents "
        "(event_id, title, description, severity, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'New', ?, ?)",
        (clean_event_id, clean_title, clean_description,
         clean_severity, now, now),
    )
    conn.commit()
    return {
        "incident_id": cur.lastrowid,
        "created": True,
        "already_existed": False,
    }


def update_incident_status(
    conn: sqlite3.Connection,
    incident_id: int,
    new_status: str,
) -> bool:
    """
    Update the status of an incident.

    incident_id is validated as a positive integer.
    new_status is validated against the VALID_STATUSES allowlist.
    All SQL uses parameterised placeholders.

    Returns True if a row was updated, False otherwise.
    """
    try:
        safe_id = validate_incident_id(incident_id)
        validate_status(new_status)
    except ValueError:
        return False

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.cursor()
    cur.execute(
        "UPDATE incidents SET status = ?, updated_at = ? WHERE incident_id = ?",
        (new_status, now, safe_id),
    )
    conn.commit()
    return cur.rowcount > 0


def get_incident_status_counts(conn: sqlite3.Connection) -> dict:
    """Return a count for each incident status."""
    cur = conn.cursor()
    counts = {s: 0 for s in VALID_STATUSES}
    cur.execute("SELECT status, COUNT(*) FROM incidents GROUP BY status")
    for status, count in cur.fetchall():
        if status in counts:
            counts[status] = count
    return counts
