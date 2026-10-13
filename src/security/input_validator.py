"""
input_validator.py
==================
Phase 5 — Security Engineering: centralised input sanitisation.

Purpose
-------
Provides sanitisation and validation helpers for all user-supplied strings
that reach the database (incident title, description, event_id) and for
integer parameters that are interpolated into SQL queries (LIMIT clauses).

Design decisions
----------------
- We STRIP and TRUNCATE strings rather than rejecting them outright.
  This prevents stored XSS if output is ever rendered as HTML, while
  avoiding unhelpful error messages for minor formatting issues.
- We NEVER execute, eval, or interpret user-supplied text.
- We use an ALLOWLIST for severity values and status values.
- Integer query parameters are CLAMPED to a safe range rather than
  passed through raw.

What this module does NOT provide
----------------------------------
- Authentication or authorisation (no user accounts exist in this phase).
- Rate limiting (not applicable to a local Streamlit app).
- Encryption at rest (SQLite file permissions are the OS's responsibility).
- Network security (this application runs locally; Phase 6 adds AWS scope).

These limitations are documented explicitly in docs/security_controls.md.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strict allowlist for severity values
VALID_SEVERITIES: frozenset[str] = frozenset(
    {"Low", "Medium", "High", "Critical"}
)

# Imported by db_utils to validate status transitions
VALID_STATUSES: frozenset[str] = frozenset(
    {"New", "Investigating", "Resolved", "False Positive"}
)

# Maximum character lengths for stored text fields
MAX_TITLE_LEN: int = 200
MAX_DESCRIPTION_LEN: int = 2000
MAX_EVENT_ID_LEN: int = 128

# Safe limit range for SQL LIMIT clauses
MIN_QUERY_LIMIT: int = 1
MAX_QUERY_LIMIT: int = 500

# Pattern that matches clearly dangerous SQL meta-characters
# Used for detection/logging only — we truncate, not inject
_SQL_META_PATTERN = re.compile(r"[;'\"\-\-\/\*\\]")


# ---------------------------------------------------------------------------
# String sanitisers
# ---------------------------------------------------------------------------

def sanitise_title(value: str) -> str:
    """
    Sanitise an incident title for safe database storage.

    - Strips leading/trailing whitespace.
    - Truncates to MAX_TITLE_LEN characters.
    - Raises ValueError if the result is empty.
    """
    cleaned = str(value).strip()[:MAX_TITLE_LEN]
    if not cleaned:
        raise ValueError("Incident title must not be empty after stripping whitespace.")
    return cleaned


def sanitise_description(value: str) -> str:
    """
    Sanitise an incident description.

    - Strips leading/trailing whitespace.
    - Truncates to MAX_DESCRIPTION_LEN characters.
    - Empty descriptions are allowed (returns empty string).
    """
    return str(value).strip()[:MAX_DESCRIPTION_LEN]


def sanitise_event_id(value: str) -> str:
    """
    Sanitise an event_id before it is stored as a foreign reference.

    - Strips whitespace.
    - Truncates to MAX_EVENT_ID_LEN characters.
    - Falls back to "manual" if the value is empty.
    """
    cleaned = str(value).strip()[:MAX_EVENT_ID_LEN]
    return cleaned if cleaned else "manual"


def sanitise_string(value: str, max_len: int = 200) -> str:
    """
    Generic string sanitiser: strip + truncate.

    Parameters
    ----------
    value   : raw user string
    max_len : maximum allowed length (default 200)
    """
    return str(value).strip()[:max_len]


# ---------------------------------------------------------------------------
# Value validators (allowlist-based)
# ---------------------------------------------------------------------------

def validate_severity(value: str) -> str:
    """
    Validate a severity value against the allowlist.

    Returns the value unchanged if valid.
    Raises ValueError if not in VALID_SEVERITIES.
    """
    if value not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity {value!r}. "
            f"Allowed values: {sorted(VALID_SEVERITIES)}"
        )
    return value


def validate_status(value: str) -> str:
    """
    Validate a status value against the allowlist.

    Returns the value unchanged if valid.
    Raises ValueError if not in VALID_STATUSES.
    """
    if value not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {value!r}. "
            f"Allowed values: {sorted(VALID_STATUSES)}"
        )
    return value


def validate_incident_id(value) -> int:
    """
    Validate that an incident_id is a positive integer.

    Accepts int or string representations.
    Raises ValueError if conversion fails or value is not positive.
    """
    try:
        int_val = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Incident ID must be a positive integer, got {value!r}."
        )
    if int_val <= 0:
        raise ValueError(
            f"Incident ID must be greater than 0, got {int_val}."
        )
    return int_val


# ---------------------------------------------------------------------------
# Integer query-parameter helpers
# ---------------------------------------------------------------------------

def clamp_limit(n, default: int = 20) -> int:
    """
    Clamp a SQL LIMIT parameter to [MIN_QUERY_LIMIT, MAX_QUERY_LIMIT].

    Prevents unexpectedly large or negative limits from being interpolated
    into query strings.

    Parameters
    ----------
    n       : raw limit value (int, float, or None)
    default : fallback if n is None or non-numeric
    """
    if n is None:
        return default
    try:
        int_n = int(n)
    except (TypeError, ValueError):
        return default
    return max(MIN_QUERY_LIMIT, min(int_n, MAX_QUERY_LIMIT))


# ---------------------------------------------------------------------------
# SQL meta-character detector (for logging / auditing only)
# ---------------------------------------------------------------------------

def contains_sql_metacharacters(value: str) -> bool:
    """
    Return True if value contains SQL meta-characters (;, ', ", --, etc.).

    This is used for LOGGING and ALERTING only — the application always
    uses parameterised queries and does not need this check for safety.
    It is provided as an extra detection layer.
    """
    return bool(_SQL_META_PATTERN.search(str(value)))


def flag_suspicious_input(field_name: str, value: str) -> str | None:
    """
    Return a warning message if value contains SQL meta-characters,
    or None if the value looks clean.

    Intended for audit logging — does not block the operation.
    """
    if contains_sql_metacharacters(value):
        return (
            f"Input for field '{field_name}' contains SQL meta-characters. "
            f"Value was accepted but flagged for audit: {value[:50]!r}"
        )
    return None
