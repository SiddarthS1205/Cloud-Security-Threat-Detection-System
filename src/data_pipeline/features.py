"""
features.py
===========
Feature engineering for the ML-ready processed_features table.

Design principles enforced here:
  1. NO DATA LEAKAGE — the label column is separated before any
     transformation and never passed to the encoder.
  2. Encoders (LabelEncoder) are fitted ONLY on the training split.
     The fitted encoder is then applied to validation and test splits.
  3. All features are derived from observable event properties only —
     no generator metadata is carried over.
  4. 'failed_attempts' is re-derived as a rolling per-user count
     from the event log itself, not taken blindly from the raw column.
     The raw value is still stored in raw_logs for reference.

Features produced
-----------------
Time-based (from event_time):
  hour_of_day       int  0-23   Night-time activity signal
  day_of_week       int  0-6    (0=Monday)
  is_weekend        int  0/1    Saturday or Sunday

Error/failure signals:
  is_error          int  0/1    1 if error_code != "None"
  failed_attempts   int  ≥0     Re-derived rolling count (see note above)

Location / privilege flags (already binary in raw data):
  unusual_location  int  0/1    Passed through unchanged
  privilege_change  int  0/1    Passed through unchanged

Derived risk flags (from event_name / event_source):
  is_iam_event         int  0/1  1 if event_source == 'iam.amazonaws.com'
  is_high_risk_action  int  0/1  1 if event_name is in HIGH_RISK_ACTIONS set

Encoded categoricals:
  event_source_enc  int   Label-encoded AWS service
  event_name_enc    int   Label-encoded API action
  region_enc        int   Label-encoded AWS region

Target (kept but NOT used as a feature):
  label             int  0/1   Passed through for the ML module to use

Note on re-deriving failed_attempts:
  We sort the dataset by (user_identity, event_time) and count how many
  AccessDenied errors each user has accumulated up to (but not including)
  the current event. This is a genuine observable metric, not a look-ahead
  from the generator.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIGH_RISK_ACTIONS = {
    "createuser", "attachuserpolicy", "putrolepolicy",
    "addusertogroup", "updateassumerolepolicy",
    "deletetrail", "stoplogging", "updatetrail",
    "deletebucket", "putbucketacl", "getbucketacl",
    "assumerolewithwebidentity", "getfederationtoken", "getsessiontoken",
}

FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "is_weekend",
    "is_error", "failed_attempts",
    "unusual_location", "privilege_change",
    "is_iam_event", "is_high_risk_action",
    "event_source_enc", "event_name_enc", "region_enc",
]

TARGET_COLUMN = "label"


# ---------------------------------------------------------------------------
# Rolling failed_attempts re-derivation
# ---------------------------------------------------------------------------

def _rederive_failed_attempts(df: pd.DataFrame) -> pd.Series:
    """
    Re-derive failed_attempts from the event log without using the raw column.

    Logic:
      - Sort events by user and event_time (ascending).
      - For each user, count how many consecutive AccessDenied errors
        occurred BEFORE (not including) the current event.
      - Reset the streak counter when a non-error event is seen.

    This approach only uses information observable at the time of each event,
    preventing future-data leakage.

    Implementation note: uses a direct iterative approach (no groupby.apply)
    to avoid pandas version-specific behaviour with include_groups.
    """
    out = df.copy()
    out["_ts"] = pd.to_datetime(out["event_time"])
    out = out.sort_values(["user_identity", "_ts"]).reset_index(drop=False)
    out["_is_denied"] = (
        out["error_code"].str.lower() == "accessdenied"
    ).astype(int)

    # Build the streak column row by row using a dict to track per-user streak
    streaks = []
    user_streak: dict = {}
    for _, row in out.iterrows():
        user = row["user_identity"]
        current_streak = user_streak.get(user, 0)
        streaks.append(current_streak)
        # Update streak for next row of this user
        if row["_is_denied"]:
            user_streak[user] = current_streak + 1
        else:
            user_streak[user] = 0

    out["failed_attempts_derived"] = streaks

    # Restore original row order
    out = out.sort_values("index").set_index("index")
    out.index.name = None

    return out["failed_attempts_derived"].fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Transforms a cleaned DataFrame into an ML-ready feature matrix.

    Usage
    -----
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(train_df)  # fits encoders
    X_test,  y_test  = fe.transform(test_df)        # applies same encoders
    """

    def __init__(self) -> None:
        self._source_enc: Optional[LabelEncoder] = None
        self._name_enc: Optional[LabelEncoder] = None
        self._region_enc: Optional[LabelEncoder] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour_of_day, day_of_week, is_weekend from event_time."""
        ts = pd.to_datetime(df["event_time"])
        df = df.copy()
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek          # Monday=0, Sunday=6
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        return df

    @staticmethod
    def _error_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive is_error from error_code."""
        df = df.copy()
        df["is_error"] = (df["error_code"].str.lower() != "none").astype(int)
        return df

    @staticmethod
    def _risk_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Derive is_iam_event and is_high_risk_action."""
        df = df.copy()
        df["is_iam_event"] = (
            df["event_source"].str.lower() == "iam.amazonaws.com"
        ).astype(int)
        df["is_high_risk_action"] = (
            df["event_name"].str.lower().isin(HIGH_RISK_ACTIONS)
        ).astype(int)
        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, fit: bool
    ) -> pd.DataFrame:
        """Label-encode event_source, event_name, aws_region.

        Output column names match FEATURE_COLUMNS exactly:
          event_source  -> event_source_enc
          event_name    -> event_name_enc
          aws_region    -> region_enc   (shortened to avoid 'aws_region_enc')
        """
        df = df.copy()

        # (raw_column, encoder_attr, output_column)
        mappings = [
            ("event_source", "_source_enc", "event_source_enc"),
            ("event_name",   "_name_enc",   "event_name_enc"),
            ("aws_region",   "_region_enc", "region_enc"),
        ]

        for col, attr, out_col in mappings:
            enc: LabelEncoder = getattr(self, attr)
            values = df[col].str.lower().fillna("unknown")

            if fit:
                enc.fit(values)
                setattr(self, attr, enc)
                encoded = enc.transform(values)
            else:
                # Handle unseen categories gracefully — map to first known class
                known = set(enc.classes_)
                safe_values = values.apply(
                    lambda v: v if v in known else enc.classes_[0]
                )
                encoded = enc.transform(safe_values)

            df[out_col] = encoded

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit encoders on df and return (feature_df, label_series).
        Call this ONLY on the training split.
        """
        self._source_enc = LabelEncoder()
        self._name_enc = LabelEncoder()
        self._region_enc = LabelEncoder()
        self._fitted = True
        return self._build(df, fit=True)

    def transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply fitted encoders to df and return (feature_df, label_series).
        Call this on validation/test splits after fit_transform.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureEngineer has not been fitted. "
                "Call fit_transform() on the training data first."
            )
        return self._build(df, fit=False)

    def _build(
        self, df: pd.DataFrame, fit: bool
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Internal: apply all feature steps."""
        out = df.copy()

        # Separate label FIRST — never passed to any encoder
        if TARGET_COLUMN not in out.columns:
            raise ValueError(
                f"Column '{TARGET_COLUMN}' not found. "
                "The label column must be present in the input."
            )
        y = out[TARGET_COLUMN].astype(int).copy()

        # Re-derive failed_attempts from event log
        out["failed_attempts"] = _rederive_failed_attempts(out)

        # Feature transformations
        out = self._time_features(out)
        out = self._error_features(out)
        out = self._risk_flags(out)
        out = self._encode_categoricals(out, fit=fit)

        # Keep only feature columns + event_id for traceability
        keep = ["event_id"] + FEATURE_COLUMNS
        X = out[keep].copy()

        logger.info(
            "[features] Built %d features for %d rows (fit=%s).",
            len(FEATURE_COLUMNS), len(X), fit,
        )
        return X, y


# ---------------------------------------------------------------------------
# Convenience function for the pipeline
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, FeatureEngineer]:
    """
    Convenience wrapper: fit and transform the full dataset at once.

    In Phase 3 (ML training), this output is further split into
    train/validation/test sets AFTER feature engineering, which is
    safe because the label encoders are fitted on the full dataset.
    However, for strict leakage prevention, Phase 3 will call
    fit_transform on the training split separately.

    Parameters
    ----------
    df : pd.DataFrame
        A cleaned DataFrame from cleaner.clean().

    Returns
    -------
    (X, y, engineer) :
        X        — feature DataFrame (event_id + FEATURE_COLUMNS)
        y        — label Series
        engineer — fitted FeatureEngineer (for reuse in Phase 3)
    """
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    return X, y, fe
