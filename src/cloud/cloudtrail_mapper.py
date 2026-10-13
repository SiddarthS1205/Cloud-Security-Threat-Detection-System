"""
cloudtrail_mapper.py
====================
Phase 6 — Cloud Engineering: maps an anonymised CloudTrail event dict
to the project's flat schema dict, ready for cleaner.clean() and storage.

SCHEMA PRODUCED
---------------
Every output dict contains exactly:
  event_id, event_time, event_source, event_name, source_ip,
  user_identity, aws_region, error_code, failed_attempts,
  unusual_location, privilege_change, label, data_source, ingested_at

KEY DESIGN DECISIONS
--------------------
1. label = LABEL_UNKNOWN (-1, not 0)
   Unknown labels use the sentinel -1 so they are never mistaken for
   confirmed-normal events. The cleaner preserves -1 unchanged. The
   predictor discards the label value (uses only features for scoring).

2. privilege_change is DERIVED, not guessed
   Derived from event_name using PRIVILEGE_CHANGE_ACTIONS (the same set
   used in generator.py). This is a rule-based approximation — a
   privilege-change-type event_name does not confirm malicious intent.

3. failed_attempts = 0 in the raw record
   The FeatureEngineer re-derives failed_attempts as a rolling per-user
   AccessDenied streak from the event log. Setting 0 in raw is correct.

4. Missing fields raise MappingWarning (not silent defaults)
   If a required field is absent, the record is skipped and logged.
   We do not silently fabricate values for required fields.

5. Optional fields with documented defaults
   aws_region → "us-east-1" (logged as WARNING if missing)
   error_code  → ""          (empty string = no error)
   source_ip   → "REDACTED"  (expected — anonymiser always sets "REDACTED")

6. data_source = "cloudtrail_real"
   Distinguishes real (anonymised) records from synthetic ones in raw_logs.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.cloud.cloudtrail_parser import LABEL_UNKNOWN
from src.data_pipeline.generator import PRIVILEGE_CHANGE_ACTIONS

logger = logging.getLogger(__name__)

DATA_SOURCE_REAL = "cloudtrail_real"


# ---------------------------------------------------------------------------
# Mapping result
# ---------------------------------------------------------------------------

@dataclass
class MappingReport:
    """Summary of mapping a batch of anonymised events."""
    total_input: int = 0
    total_mapped: int = 0
    total_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Mapping report: {self.total_mapped}/{self.total_input} records mapped",
            f"  Skipped : {self.total_skipped}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for r in self.skipped_reasons[:10]:  # show first 10 only
            lines.append(f"  SKIPPED: {r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-event mapper
# ---------------------------------------------------------------------------

def map_event(
    anon_event: dict[str, Any],
    ingested_at: str | None = None,
) -> dict[str, Any] | None:
    """
    Map one anonymised CloudTrail event dict to the project flat schema.

    Parameters
    ----------
    anon_event   : dict from CloudTrailAnonymiser.anonymise()
    ingested_at  : ISO 8601 UTC timestamp string (default: current time)

    Returns
    -------
    dict or None
        Flat schema dict on success, None if a required field is missing.
    """
    if ingested_at is None:
        ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Required fields — skip record if any are absent
    # ------------------------------------------------------------------
    event_time = anon_event.get("eventTime")
    event_source = anon_event.get("eventSource")
    event_name_raw = anon_event.get("eventName")

    if not event_time or not event_source or not event_name_raw:
        missing = [
            f for f, v in [
                ("eventTime", event_time),
                ("eventSource", event_source),
                ("eventName", event_name_raw),
            ]
            if not v
        ]
        logger.warning(
            "[mapper] Skipping event %r — missing required fields: %s",
            anon_event.get("eventID", "unknown"),
            missing,
        )
        return None

    # Lowercase for consistency with synthetic dataset
    event_source_clean = str(event_source).strip().lower()
    event_name_clean = str(event_name_raw).strip().lower()

    # ------------------------------------------------------------------
    # Optional fields with documented defaults
    # ------------------------------------------------------------------
    aws_region = anon_event.get("awsRegion")
    if not aws_region:
        logger.warning(
            "[mapper] Event %r has no awsRegion; defaulting to 'us-east-1'.",
            anon_event.get("eventID", "unknown"),
        )
        aws_region = "us-east-1"
    aws_region = str(aws_region).strip().lower()

    error_code = anon_event.get("errorCode") or ""
    error_code = str(error_code).strip() if error_code else ""

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------

    # privilege_change: rule-based from event_name
    # Documented limitation: not all PRIVILEGE_CHANGE_ACTIONS are
    # necessarily malicious; legitimate admins perform these too.
    privilege_change = 1 if event_name_raw in PRIVILEGE_CHANGE_ACTIONS else 0

    # unusual_location: already computed by anonymiser (0 or 1)
    unusual_location = int(anon_event.get("unusual_location", 0))

    # ------------------------------------------------------------------
    # event_id: use CloudTrail eventID if present, else generate UUID
    # ------------------------------------------------------------------
    event_id = anon_event.get("eventID") or str(uuid.uuid4())

    # ------------------------------------------------------------------
    # label: LABEL_UNKNOWN (-1) — not 0, not 1
    # Unknown = unlabelled real event. The predictor discards this value.
    # Never interpret -1 as "normal".
    # ------------------------------------------------------------------
    label = LABEL_UNKNOWN

    return {
        "event_id": str(event_id),
        "event_time": str(event_time),
        "event_source": event_source_clean,
        "event_name": event_name_clean,
        "source_ip": str(anon_event.get("sourceIPAddress", "REDACTED")),
        "user_identity": str(anon_event.get("userIdentityPseudonym", "user_unknown")),
        "aws_region": aws_region,
        "error_code": error_code,
        "failed_attempts": 0,      # re-derived by FeatureEngineer
        "unusual_location": unusual_location,
        "privilege_change": privilege_change,
        "label": label,             # -1 = unknown, not normal
        "data_source": DATA_SOURCE_REAL,
        "ingested_at": ingested_at,
    }


# ---------------------------------------------------------------------------
# Batch mapper
# ---------------------------------------------------------------------------

def map_events(
    anon_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], MappingReport]:
    """
    Map a list of anonymised CloudTrail events to the project flat schema.

    Parameters
    ----------
    anon_events : list of dicts from CloudTrailAnonymiser.anonymise()

    Returns
    -------
    (mapped_records, report)
        mapped_records : list of flat schema dicts (only successfully mapped)
        report         : MappingReport with counts and warnings
    """
    report = MappingReport(total_input=len(anon_events))
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mapped = []

    for anon_event in anon_events:
        result = map_event(anon_event, ingested_at=ingested_at)
        if result is None:
            report.total_skipped += 1
            report.skipped_reasons.append(
                f"Event {anon_event.get('eventID', 'unknown')}: missing required fields"
            )
        else:
            mapped.append(result)

    report.total_mapped = len(mapped)
    logger.info("[mapper] %s", report.summary())
    return mapped, report
