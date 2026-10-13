"""
cleaner.py
==========
Data cleaning for the validated CloudTrail-style dataset.

What this module does:
  1. Strips leading/trailing whitespace from all string columns.
  2. Normalises event_time to a UTC timezone-aware ISO 8601 string.
  3. Lowercases event_source, event_name, aws_region for consistent matching.
  4. Replaces empty/null error_code with the string "None"
     (avoids Python None confusion in downstream steps).
  5. Casts failed_attempts, unusual_location, privilege_change, label to int.
  6. Adds an ingested_at column (current UTC time) for audit trail purposes.

Cleaning does NOT:
  - Drop any rows (that is the validator's job).
  - Touch the label column value (only casts it to int).
  - Add or remove columns beyond ingested_at.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that should be stored as integers
INTEGER_COLUMNS = ["failed_attempts", "unusual_location", "privilege_change", "label"]

# String columns that get whitespace stripping
STRING_COLUMNS = [
    "event_id", "event_time", "event_source", "event_name",
    "source_ip", "user_identity", "aws_region", "error_code",
    "data_source",
]

# Columns normalised to lowercase
LOWERCASE_COLUMNS = ["event_source", "event_name", "aws_region"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning transformations to a validated DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that has already passed through ``validator.validate``.

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the input. The original is not modified.
    """
    out = df.copy()
    original_len = len(out)

    # ------------------------------------------------------------------
    # Step 1: Strip whitespace from all string columns
    # ------------------------------------------------------------------
    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    # ------------------------------------------------------------------
    # Step 2: Normalise event_time to UTC ISO 8601
    # ------------------------------------------------------------------
    def _normalise_ts(val: str) -> str:
        """Parse any ISO 8601 string and return a UTC 'Z'-suffixed string."""
        try:
            ts = pd.Timestamp(val)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return val   # leave as-is; validator already screened these

    out["event_time"] = out["event_time"].apply(_normalise_ts)
    logger.debug("Normalised event_time to UTC for %d rows.", original_len)

    # ------------------------------------------------------------------
    # Step 3: Lowercase categorical columns
    # ------------------------------------------------------------------
    for col in LOWERCASE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].str.lower().str.strip()

    # ------------------------------------------------------------------
    # Step 4: Fill empty/null error_code with "None"
    # ------------------------------------------------------------------
    out["error_code"] = out["error_code"].replace("", "None").fillna("None")

    # ------------------------------------------------------------------
    # Step 5: Cast integer columns
    # ------------------------------------------------------------------
    for col in INTEGER_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Step 6: Add ingested_at audit timestamp
    # ------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["ingested_at"] = now_utc

    logger.info(
        "[cleaner] Cleaning complete. %d rows, %d columns.",
        len(out), len(out.columns),
    )
    return out
