"""
cloudtrail_parser.py
====================
Phase 6 — Cloud Engineering: reads CloudTrail JSON log files from disk.

PURPOSE
-------
Parses CloudTrail-formatted JSON files (Option B — no AWS SDK, no network).
Supports two formats:
  1. Records array:   {"Records": [...]}          (single-file CloudTrail export)
  2. Line-delimited:  one JSON object per line     (some export tools produce this)

WHAT THIS MODULE DOES NOT DO
-----------------------------
- Does not connect to AWS S3 or any network endpoint.
- Does not validate record content (that is cloudtrail_mapper.py's job).
- Does not anonymise any fields (that is cloudtrail_anonymiser.py's job).

USAGE
-----
    from src.cloud.cloudtrail_parser import parse_file, ParseResult
    result = parse_file("data/cloud_samples/my_trail.json")
    print(result.total_records)   # number of records found
    for event in result.records:  # list[dict]
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LABEL_UNKNOWN: int = -1   # sentinel for unlabelled real CloudTrail events


@dataclass
class ParseResult:
    """Outcome of parsing a CloudTrail JSON file."""
    records: list[dict[str, Any]] = field(default_factory=list)
    total_records: int = 0
    skipped_records: int = 0
    parse_errors: list[str] = field(default_factory=list)
    source_file: str = ""

    @property
    def succeeded(self) -> bool:
        return self.total_records > 0 and not self.parse_errors


def parse_file(path: str | Path) -> ParseResult:
    """
    Parse a CloudTrail JSON log file from disk.

    Parameters
    ----------
    path : str or Path
        Path to the CloudTrail JSON file.

    Returns
    -------
    ParseResult
        Contains the list of raw event dicts and parsing statistics.
        Records are raw CloudTrail dicts — not anonymised or mapped yet.
    """
    result = ParseResult(source_file=str(path))
    path = Path(path)

    if not path.exists():
        result.parse_errors.append(f"File not found: {path}")
        return result

    if path.stat().st_size == 0:
        result.parse_errors.append(f"File is empty: {path}")
        return result

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        result.parse_errors.append(f"Cannot read file: {exc}")
        return result

    # ------------------------------------------------------------------
    # Format 1: Standard CloudTrail export — {"Records": [...]}
    # ------------------------------------------------------------------
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "Records" in data:
            records = data["Records"]
            if not isinstance(records, list):
                result.parse_errors.append(
                    "'Records' key exists but is not a list."
                )
                return result
            result.records = _filter_records(records, result)
            result.total_records = len(result.records)
            logger.info(
                "[parser] Loaded %d records from %s (Records format).",
                result.total_records, path.name,
            )
            return result

        if isinstance(data, list):
            # Some tools export a bare JSON array
            result.records = _filter_records(data, result)
            result.total_records = len(result.records)
            logger.info(
                "[parser] Loaded %d records from %s (array format).",
                result.total_records, path.name,
            )
            return result

    except json.JSONDecodeError:
        pass   # try line-delimited next

    # ------------------------------------------------------------------
    # Format 2: Line-delimited JSON (one object per line)
    # ------------------------------------------------------------------
    raw_records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                raw_records.append(obj)
            else:
                result.skipped_records += 1
                logger.debug("[parser] Line %d: not a dict, skipped.", line_no)
        except json.JSONDecodeError as exc:
            result.skipped_records += 1
            logger.debug("[parser] Line %d: JSON error — %s", line_no, exc)

    if not raw_records:
        result.parse_errors.append(
            "File could not be parsed as Records-format JSON, "
            "bare JSON array, or line-delimited JSON."
        )
        return result

    result.records = _filter_records(raw_records, result)
    result.total_records = len(result.records)
    logger.info(
        "[parser] Loaded %d records from %s (line-delimited format).",
        result.total_records, path.name,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("eventTime", "eventSource", "eventName")


def _filter_records(
    records: list[Any],
    result: ParseResult,
) -> list[dict]:
    """
    Return only records that are dicts with the three required fields.
    Increments result.skipped_records for anything dropped.
    """
    kept = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            result.skipped_records += 1
            logger.debug("[parser] Record %d is not a dict, skipped.", i)
            continue
        missing = [f for f in _REQUIRED_FIELDS if not rec.get(f)]
        if missing:
            result.skipped_records += 1
            logger.debug(
                "[parser] Record %d missing required fields %s, skipped.",
                i, missing,
            )
            continue
        kept.append(rec)
    return kept
