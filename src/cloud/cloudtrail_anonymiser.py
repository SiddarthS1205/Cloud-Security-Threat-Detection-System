"""
cloudtrail_anonymiser.py
========================
Phase 6 — Cloud Engineering: sensitive field pseudonymisation.

PURPOSE
-------
Real AWS CloudTrail events contain sensitive information:
  - IAM usernames and role names (personally identifiable in enterprise context)
  - AWS account IDs (12-digit numbers that identify the account)
  - Resource ARNs (contain account IDs and resource names)
  - Request parameters (may contain bucket names, key IDs, etc.)
  - Response elements (may contain newly created resource identifiers)

This module removes or pseudonymises those fields BEFORE any data
is passed to the mapper, written to SQLite, or used in prediction.

PSEUDONYMISATION DESIGN
-----------------------
User identities are replaced with sequential labels: user_1, user_2, ...
The counter increments per unique identity seen in a single session.

CRITICAL DESIGN CONSTRAINTS
----------------------------
1. The original-to-pseudonym mapping is held in memory only, in the
   Anonymiser instance. It is NEVER written to disk, SQLite, or logs.
2. Each Anonymiser instance maintains its own counter. Create a new
   instance per ingestion session to ensure consistent numbering.
3. Dropped fields: requestParameters, responseElements, resources,
   userIdentity.arn, userIdentity.accountId, userIdentity.sessionContext.
4. The sourceIPAddress is consumed to compute unusual_location BEFORE
   being replaced with "REDACTED". The flag (0 or 1) is the only
   IP-derived value stored.
5. The accountId field in userIdentity is always replaced with "REDACTED".

WHAT IS NOT ANONYMISED
----------------------
- eventSource, eventName, awsRegion — required for feature engineering
- eventTime — required for time-based features
- errorCode — required for is_error feature
- eventID — used as event_id primary key (not sensitive by itself)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CloudTrailAnonymiser:
    """
    Pseudonymises sensitive fields in a CloudTrail event dict.

    Usage
    -----
    anonymiser = CloudTrailAnonymiser(allowlist)
    clean_event = anonymiser.anonymise(raw_event)

    The same instance should be used for all events in one ingestion
    session to ensure consistent user_N numbering within that session.
    """

    def __init__(self, ip_allowlist=None) -> None:
        """
        Parameters
        ----------
        ip_allowlist : IPAllowlist or None
            Used to compute unusual_location before redacting sourceIPAddress.
            If None, unusual_location is always 0.
        """
        self._ip_allowlist = ip_allowlist
        self._user_counter: int = 0
        self._user_map: dict[str, str] = {}  # in-memory only, never persisted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymise(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Return a new dict with all sensitive fields pseudonymised or removed.

        Parameters
        ----------
        event : dict
            A single raw CloudTrail event record (from JSON "Records" list).

        Returns
        -------
        dict
            Anonymised event suitable for schema mapping.
            Contains the derived field 'unusual_location' (int 0/1).
            sourceIPAddress is replaced with "REDACTED".
        """
        out = {}

        # Fields passed through unchanged (not sensitive)
        for field in ("eventVersion", "eventTime", "eventSource",
                      "eventName", "awsRegion", "errorCode", "eventID"):
            out[field] = event.get(field)

        # Compute unusual_location from IP BEFORE redacting it
        raw_ip = event.get("sourceIPAddress", "")
        if self._ip_allowlist is not None:
            out["unusual_location"] = self._ip_allowlist.is_unusual(raw_ip)
        else:
            out["unusual_location"] = 0  # conservative default

        # Always redact the actual IP
        out["sourceIPAddress"] = "REDACTED"

        # Pseudonymise user identity
        out["userIdentityPseudonym"] = self._pseudonymise_user(
            event.get("userIdentity", {})
        )

        # Drop all potentially sensitive nested structures
        # requestParameters, responseElements, resources are not stored
        # (they are not needed by the feature pipeline)

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pseudonymise_user(self, user_identity: dict | None) -> str:
        """
        Map a userIdentity dict to a stable sequential pseudonym.

        The key used for mapping is the userName if present, otherwise
        the principalId, otherwise the arn. Falls back to "unknown_user".

        The mapping exists only in self._user_map (in-memory).
        """
        if not user_identity:
            return self._get_or_create_pseudonym("__unknown__")

        # Try keys in order of preference
        raw_key = (
            user_identity.get("userName")
            or user_identity.get("principalId")
            or user_identity.get("arn")
            or "__unknown__"
        )
        raw_key = str(raw_key).strip()
        if not raw_key:
            raw_key = "__unknown__"

        return self._get_or_create_pseudonym(raw_key)

    def _get_or_create_pseudonym(self, raw_key: str) -> str:
        """Return existing pseudonym or create a new user_N one."""
        if raw_key not in self._user_map:
            self._user_counter += 1
            self._user_map[raw_key] = f"user_{self._user_counter}"
        return self._user_map[raw_key]

    @property
    def unique_users_seen(self) -> int:
        """Number of distinct identities pseudonymised in this session."""
        return self._user_counter
