# cloud package — Phase 6
#
# Modules in this package are used for CloudTrail log ingestion.
# They do NOT connect to AWS in Option B (offline fixture mode).
# AWS connectivity (Option A) requires boto3 and is not active in Phase 4/5.
#
# Modules:
#   cloudtrail_parser.py     — reads CloudTrail JSON from disk
#   cloudtrail_anonymiser.py — pseudonymises PII before storage
#   cloudtrail_mapper.py     — maps CT fields to project flat schema
#   cloudtrail_ingestor.py   — orchestrates the full pipeline
#   cloudtrail_reporter.py   — generates probability-distribution report
#   ip_allowlist.py          — derives unusual_location from config
