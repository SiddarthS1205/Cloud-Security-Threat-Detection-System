"""
ip_allowlist.py
===============
Phase 6 — Cloud Engineering: IP allowlist for unusual_location derivation.

PURPOSE
-------
Real CloudTrail logs do not contain an 'unusual_location' flag. This module
computes an approximate binary indicator by comparing an event's source IP
against a configurable list of known-safe network prefixes.

IMPORTANT LIMITATIONS
---------------------
- unusual_location=1 does NOT prove malicious intent.
- unusual_location=0 does NOT confirm the activity is safe.
- The model was trained on synthetic unusual_location flags that do not
  precisely match any real IP allowlist logic.
- If the allowlist file is missing or empty, ALL IPs return unusual_location=0
  (conservative default: no false positives from configuration errors).
- Service endpoints (e.g. "lambda.amazonaws.com") returned as sourceIPAddress
  by AWS are always treated as not-unusual (unusual_location=0).

CONFIG FILE
-----------
Default location: config/ip_allowlist.yaml
Format:
    known_prefixes:
      - "192.0.2."
      - "198.51.100."
      - "10.0."
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config") / "ip_allowlist.yaml"

# AWS service endpoint pattern — when sourceIPAddress is a service domain,
# it is not a real IP and should not be classified as unusual.
_SERVICE_SUFFIXES = (
    ".amazonaws.com",
    ".aws.amazon.com",
)


def _load_prefixes(config_path: Path) -> list[str]:
    """
    Load known-safe IP prefixes from YAML config.
    Returns an empty list if the file is missing or malformed.
    """
    if not config_path.exists():
        logger.debug(
            "[ip_allowlist] Config not found at %s; defaulting to empty allowlist.",
            config_path,
        )
        return []
    try:
        import yaml  # imported here so pyyaml absence only affects cloud modules
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        prefixes = data.get("known_prefixes", []) if data else []
        if not isinstance(prefixes, list):
            logger.warning("[ip_allowlist] 'known_prefixes' must be a list; ignoring.")
            return []
        return [str(p) for p in prefixes if p]
    except Exception as exc:
        logger.warning("[ip_allowlist] Could not load config: %s", exc)
        return []


class IPAllowlist:
    """
    Determines whether a source IP is 'unusual' relative to known-safe prefixes.

    Usage
    -----
    allowlist = IPAllowlist()                          # loads default config
    allowlist = IPAllowlist("config/ip_allowlist.yaml")
    flag = allowlist.is_unusual("203.0.113.99")        # True
    flag = allowlist.is_unusual("192.0.2.10")          # False (known prefix)
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self._prefixes = _load_prefixes(Path(config_path))
        if self._prefixes:
            logger.info(
                "[ip_allowlist] Loaded %d known prefix(es).", len(self._prefixes)
            )
        else:
            logger.info(
                "[ip_allowlist] No known prefixes configured. "
                "All IPs will return unusual_location=0 (conservative default)."
            )

    @property
    def is_configured(self) -> bool:
        """True if at least one prefix is configured."""
        return bool(self._prefixes)

    def is_unusual(self, source_ip: str | None) -> int:
        """
        Return 1 if the IP is unusual (outside the allowlist), else 0.

        Parameters
        ----------
        source_ip : str or None
            The sourceIPAddress value from a CloudTrail event.

        Returns
        -------
        int
            0 — not unusual (or allowlist is empty, or IP is a service endpoint)
            1 — unusual (IP does not match any known-safe prefix)

        Notes
        -----
        - Returns 0 (conservative) when the allowlist is empty.
        - Returns 0 for None, empty string, or service endpoint URLs.
        - Does NOT consider 1 to mean "malicious" — only "outside known prefixes".
        """
        if not source_ip:
            return 0

        ip_str = str(source_ip).strip()

        # Service endpoints (e.g. "lambda.amazonaws.com") are not real IPs
        if any(ip_str.endswith(suf) for suf in _SERVICE_SUFFIXES):
            return 0

        # Empty allowlist → conservative default (not unusual)
        if not self._prefixes:
            return 0

        # If IP matches any known-safe prefix, it is not unusual
        if any(ip_str.startswith(prefix) for prefix in self._prefixes):
            return 0

        return 1
