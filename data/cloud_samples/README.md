# cloud_samples/

This directory holds real or realistic CloudTrail JSON log files for Phase 6 evaluation.

## ⚠️ IMPORTANT — DO NOT COMMIT REAL AWS LOGS

Real CloudTrail logs may contain:
- AWS account IDs
- IAM usernames (personally identifiable in enterprise settings)
- Resource ARNs with account and resource identifiers
- Source IP addresses

**All `.json` and `.log` files in this directory are git-ignored.**

## What to place here

For **Option B** (offline, no live AWS):
- The AWS sample CloudTrail log from the official documentation
- Hand-crafted fixture files that do NOT contain real account IDs

For **Option A** (future — personal test account):
- JSON exports from your own test AWS account only
- Run anonymisation BEFORE examining any output

## How to use (Option B — offline)

```powershell
# Ingest the test fixture (no AWS connection needed)
venv\Scripts\activate
python -c "
import sys; sys.path.insert(0,'.')
from src.cloud.cloudtrail_ingestor import ingest_file
r = ingest_file('tests/fixtures/sample_cloudtrail.json')
print(r.summary())
"
```

## CloudTrail JSON format reference

https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-log-file-examples.html
