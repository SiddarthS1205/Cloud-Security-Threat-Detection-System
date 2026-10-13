"""
csv_validator.py
================
Validates a user-uploaded CSV file before passing it to the ML predictor.

Validation checks
-----------------
1. Required columns are present.
2. event_time values parse as ISO 8601 datetime strings.
3. Numeric columns (failed_attempts, unusual_location, privilege_change)
   contain valid integer values within expected ranges.
4. error_code is a string (empty or one of the known AWS error codes).
5. Row count is within the accepted limit (1 – MAX_ROWS).
6. No completely empty rows.

The validator does NOT modify the DataFrame. It returns a
ValidationResult dataclass that the dashboard can display to the user.

Columns required in uploaded CSV
---------------------------------
Minimum required (dashboard adds defaults for missing optional columns):
  event_time, event_source, event_name, source_ip,
  user_identity, aws_region, error_code,
  failed_attempts, unusual_location, privilege_change
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MAX_ROWS = 5000

# Minimum columns that must be in an uploaded file
REQUIRED_UPLOAD_COLUMNS = [
    "event_time",
    "event_source",
    "event_name",
    "source_ip",
    "user_identity",
    "aws_region",
    "error_code",
    "failed_attempts",
    "unusual_location",
    "privilege_change",
]

KNOWN_AWS_ERRORS = {
    "", "None", "AccessDenied", "InvalidClientTokenId",
    "NoSuchBucket", "NoSuchKey", "Throttling",
    "UnauthorizedOperation", "AuthFailure",
}

INTEGER_COLUMNS = {
    "failed_attempts": (0, 1000),
    "unusual_location": (0, 1),
    "privilege_change": (0, 1),
}


@dataclass
class UploadValidationResult:
    """Outcome of validating a user-uploaded CSV."""
    is_valid: bool = True
    row_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Validation: {status}  |  Rows: {self.row_count}"]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def validate_upload(df: pd.DataFrame) -> UploadValidationResult:
    """
    Validate a DataFrame loaded from a user-uploaded CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The raw DataFrame as read from pd.read_csv().

    Returns
    -------
    UploadValidationResult
    """
    result = UploadValidationResult(row_count=len(df))

    # ------------------------------------------------------------------
    # 1. Row count
    # ------------------------------------------------------------------
    if len(df) == 0:
        result.add_error("The uploaded file contains no data rows.")
        return result   # no point continuing

    if len(df) > MAX_ROWS:
        result.add_error(
            f"File has {len(df):,} rows; maximum allowed is {MAX_ROWS:,}. "
            "Please split into smaller files."
        )

    # ------------------------------------------------------------------
    # 2. Required columns
    # ------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing_cols:
        result.add_error(f"Missing required columns: {missing_cols}")
        return result   # cannot validate further without required columns

    # ------------------------------------------------------------------
    # 3. No completely empty rows
    # ------------------------------------------------------------------
    key_cols = ["event_time", "event_source", "event_name"]
    empty_rows = df[key_cols].isnull().all(axis=1).sum()
    if empty_rows > 0:
        result.add_warning(
            f"{empty_rows} completely empty row(s) found. "
            "They will be skipped during prediction."
        )

    # ------------------------------------------------------------------
    # 4. event_time must parse as ISO 8601
    # ------------------------------------------------------------------
    bad_ts_count = 0
    for val in df["event_time"].dropna():
        try:
            pd.Timestamp(val)
        except Exception:
            bad_ts_count += 1
    if bad_ts_count > 0:
        result.add_error(
            f"{bad_ts_count} row(s) have unparseable event_time values. "
            "Expected ISO 8601 format, e.g. 2024-06-15T10:23:00Z"
        )

    # ------------------------------------------------------------------
    # 5. Integer columns: type and range
    # ------------------------------------------------------------------
    for col, (lo, hi) in INTEGER_COLUMNS.items():
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        null_count = coerced.isnull().sum()
        if null_count > 0:
            result.add_error(
                f"Column '{col}' has {null_count} non-numeric value(s). "
                f"Expected integers in [{lo}, {hi}]."
            )
        else:
            out_of_range = ((coerced < lo) | (coerced > hi)).sum()
            if out_of_range > 0:
                result.add_error(
                    f"Column '{col}' has {out_of_range} value(s) outside "
                    f"the valid range [{lo}, {hi}]."
                )

    # ------------------------------------------------------------------
    # 6. error_code: warn about unknown values (don't block)
    # ------------------------------------------------------------------
    if "error_code" in df.columns:
        error_vals = df["error_code"].fillna("").astype(str).str.strip()
        unknown = error_vals[~error_vals.isin(KNOWN_AWS_ERRORS)]
        if len(unknown) > 0:
            unique_unknown = unknown.unique()[:5]
            result.add_warning(
                f"{len(unknown)} row(s) have unrecognised error_code values "
                f"(e.g. {list(unique_unknown)}). "
                "These will still be processed."
            )

    return result


def prepare_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any missing optional columns with safe defaults so the predictor
    pipeline does not raise KeyError on a partial upload.

    Does NOT modify the caller's DataFrame — returns a new copy.
    """
    out = df.copy()

    # Drop rows where all key columns are null
    key_cols = ["event_time", "event_source", "event_name"]
    out = out.dropna(subset=key_cols, how="all").reset_index(drop=True)

    # Fill missing optional columns with safe defaults
    defaults = {
        "failed_attempts": 0,
        "unusual_location": 0,
        "privilege_change": 0,
        "error_code": "",
        "source_ip": "0.0.0.0",
        "user_identity": "unknown",
        "aws_region": "us-east-1",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            filled = out[col].fillna(default)
            # Suppress pandas FutureWarning about silent downcasting
            try:
                out[col] = filled.infer_objects(copy=False)
            except Exception:
                out[col] = filled

    return out
