"""
validator.py
============
Schema and quality validation for the synthetic CloudTrail-style dataset.

What this module checks:
  1. Required columns are all present.
  2. No null values in critical columns.
  3. event_time parses as a valid ISO 8601 UTC datetime.
  4. label is exactly 0 or 1 (integer).
  5. failed_attempts is a non-negative integer.
  6. unusual_location and privilege_change are 0 or 1.
  7. No duplicate event_id values.
  8. event_source is one of the known allowed values.

Outcome:
  - Returns a cleaned DataFrame with invalid rows removed.
  - Returns a ValidationReport dataclass describing what was found.
  - Every dropped row is logged with the reason so nothing is silently lost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "event_id", "event_time", "event_source", "event_name",
    "source_ip", "user_identity", "aws_region", "error_code",
    "failed_attempts", "unusual_location", "privilege_change",
    "label", "data_source",
]

ALLOWED_EVENT_SOURCES = {
    "iam.amazonaws.com",
    "s3.amazonaws.com",
    "ec2.amazonaws.com",
    "sts.amazonaws.com",
    "kms.amazonaws.com",
    "cloudtrail.amazonaws.com",
}

ALLOWED_REGIONS = {
    "us-east-1", "us-west-2", "eu-west-1",
    "ap-southeast-1", "ap-northeast-1",
}

BINARY_COLUMNS = ["unusual_location", "privilege_change", "label"]
NON_NEGATIVE_INT_COLUMNS = ["failed_attempts"]


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Summary of what the validator found and did."""
    total_input_rows: int = 0
    total_output_rows: int = 0
    missing_columns: list[str] = field(default_factory=list)
    rows_dropped: dict[str, int] = field(default_factory=dict)

    @property
    def total_dropped(self) -> int:
        return sum(self.rows_dropped.values())

    @property
    def passed(self) -> bool:
        """True if the file had all required columns and at least one valid row."""
        return not self.missing_columns and self.total_output_rows > 0

    def summary(self) -> str:
        lines = [
            "=== Validation Report ===",
            f"  Input rows  : {self.total_input_rows:,}",
            f"  Output rows : {self.total_output_rows:,}",
            f"  Dropped     : {self.total_dropped:,}",
        ]
        if self.missing_columns:
            lines.append(f"  Missing cols: {self.missing_columns}")
        for reason, count in self.rows_dropped.items():
            lines.append(f"    - {reason}: {count:,}")
        lines.append(f"  Status      : {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate a raw DataFrame of CloudTrail-style log records.

    Parameters
    ----------
    df : pd.DataFrame
        Raw records as loaded from CSV or generated in memory.

    Returns
    -------
    (clean_df, report) : tuple
        clean_df  — DataFrame with invalid rows removed.
        report    — ValidationReport with full audit trail.
    """
    report = ValidationReport(total_input_rows=len(df))

    # ------------------------------------------------------------------
    # Check 1: Required columns present
    # ------------------------------------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.missing_columns = missing
        logger.error("Missing required columns: %s", missing)
        report.total_output_rows = 0
        return pd.DataFrame(columns=df.columns), report

    # Work on a copy to avoid mutating the caller's DataFrame
    out = df.copy()
    initial = len(out)

    # ------------------------------------------------------------------
    # Check 2: Null event_id
    # ------------------------------------------------------------------
    mask_null_id = out["event_id"].isnull() | (out["event_id"].astype(str).str.strip() == "")
    n_dropped = mask_null_id.sum()
    if n_dropped:
        logger.warning("Dropping %d rows with null/empty event_id.", n_dropped)
        report.rows_dropped["null_event_id"] = int(n_dropped)
    out = out[~mask_null_id].copy()

    # ------------------------------------------------------------------
    # Check 3: Null event_time
    # ------------------------------------------------------------------
    mask_null_ts = out["event_time"].isnull() | (out["event_time"].astype(str).str.strip() == "")
    n_dropped = mask_null_ts.sum()
    if n_dropped:
        logger.warning("Dropping %d rows with null/empty event_time.", n_dropped)
        report.rows_dropped["null_event_time"] = int(n_dropped)
    out = out[~mask_null_ts].copy()

    # ------------------------------------------------------------------
    # Check 4: event_time parses as a valid ISO 8601 datetime
    # ------------------------------------------------------------------
    def _is_valid_ts(val) -> bool:
        try:
            pd.Timestamp(val)
            return True
        except Exception:
            return False

    mask_bad_ts = ~out["event_time"].apply(_is_valid_ts)
    n_dropped = mask_bad_ts.sum()
    if n_dropped:
        logger.warning("Dropping %d rows with unparseable event_time.", n_dropped)
        report.rows_dropped["invalid_event_time_format"] = int(n_dropped)
    out = out[~mask_bad_ts].copy()

    # ------------------------------------------------------------------
    # Check 5: label must be 0 or 1
    # ------------------------------------------------------------------
    try:
        label_vals = pd.to_numeric(out["label"], errors="coerce")
        mask_bad_label = ~label_vals.isin([0, 1])
        n_dropped = mask_bad_label.sum()
        if n_dropped:
            logger.warning("Dropping %d rows with invalid label value.", n_dropped)
            report.rows_dropped["invalid_label"] = int(n_dropped)
        out = out[~mask_bad_label].copy()
    except Exception as exc:
        logger.error("Error validating label column: %s", exc)

    # ------------------------------------------------------------------
    # Check 6: failed_attempts is a non-negative integer
    # ------------------------------------------------------------------
    try:
        fa_vals = pd.to_numeric(out["failed_attempts"], errors="coerce")
        mask_bad_fa = fa_vals.isnull() | (fa_vals < 0)
        n_dropped = mask_bad_fa.sum()
        if n_dropped:
            logger.warning(
                "Coercing %d rows with invalid failed_attempts to 0.", n_dropped
            )
            report.rows_dropped["coerced_failed_attempts"] = int(n_dropped)
            out.loc[mask_bad_fa, "failed_attempts"] = 0
    except Exception as exc:
        logger.error("Error validating failed_attempts: %s", exc)

    # ------------------------------------------------------------------
    # Check 7: unusual_location and privilege_change are 0 or 1
    # ------------------------------------------------------------------
    for col in ["unusual_location", "privilege_change"]:
        try:
            vals = pd.to_numeric(out[col], errors="coerce")
            mask_bad = ~vals.isin([0, 1])
            n_dropped = mask_bad.sum()
            if n_dropped:
                logger.warning(
                    "Dropping %d rows with invalid %s value.", n_dropped, col
                )
                report.rows_dropped[f"invalid_{col}"] = int(n_dropped)
            out = out[~mask_bad].copy()
        except Exception as exc:
            logger.error("Error validating %s: %s", col, exc)

    # ------------------------------------------------------------------
    # Check 8: event_source is a known AWS service
    # ------------------------------------------------------------------
    mask_bad_source = ~out["event_source"].isin(ALLOWED_EVENT_SOURCES)
    n_dropped = mask_bad_source.sum()
    if n_dropped:
        logger.warning(
            "Dropping %d rows with unrecognised event_source.", n_dropped
        )
        report.rows_dropped["unknown_event_source"] = int(n_dropped)
    out = out[~mask_bad_source].copy()

    # ------------------------------------------------------------------
    # Check 9: Duplicate event_id — keep first occurrence
    # ------------------------------------------------------------------
    n_before = len(out)
    out = out.drop_duplicates(subset=["event_id"], keep="first").copy()
    n_dupes = n_before - len(out)
    if n_dupes:
        logger.warning("Dropped %d duplicate event_id rows.", n_dupes)
        report.rows_dropped["duplicate_event_id"] = n_dupes

    report.total_output_rows = len(out)
    logger.info(report.summary())
    return out, report
