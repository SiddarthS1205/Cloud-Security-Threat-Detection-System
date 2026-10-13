# AWS Setup Guide
## AI-Powered Cloud Security Threat Detection and Analytics System
### Wipro CPPE Project — Phase 6: Cloud Engineering

---

> **Current status (Phase 6, Option B):**  
> The system works completely offline using a hand-crafted CloudTrail fixture.  
> No AWS account, credentials, or network connection are needed.  
> Option A (live AWS) is future work, documented below for reference.

---

## Option B — Offline (currently active)

### What works without AWS

| Feature | Works offline? |
|---|---|
| Parse CloudTrail JSON from local file | ✅ Yes |
| Pseudonymise user identities | ✅ Yes |
| Map CloudTrail fields to project schema | ✅ Yes |
| Compute `unusual_location` from IP allowlist | ✅ Yes |
| Derive `privilege_change` from event_name | ✅ Yes |
| Ingest to SQLite | ✅ Yes |
| Score with trained model | ✅ Yes |
| Generate probability report | ✅ Yes |
| Precision / recall / F1 evaluation | ❌ No — needs ground-truth labels |

### Running the offline pipeline

```powershell
cd d:\Wipro_CPPE_Cloud_Security_AI
venv\Scripts\activate

# Ingest the fixture file (15 sample events)
python -c "
import sys; sys.path.insert(0,'.')
from src.cloud.cloudtrail_ingestor import ingest_and_predict
from src.cloud.cloudtrail_reporter import generate_report

report, preds = ingest_and_predict('tests/fixtures/sample_cloudtrail.json')
print(report.summary())
print()
generate_report(preds, ingestion_summary=report.summary())
"

# View the report
Get-Content reports\cloudtrail_evaluation.txt
```

### Understanding label=-1

Every real CloudTrail event ingested receives `label = -1`.  
This is an explicit "unknown" sentinel — not 0 (normal), not 1 (suspicious).  
The ML predictor discards this value and only uses the 12 engineered features.  
**Do not interpret label=-1 records as confirmed normal events.**

---

## Option A — Live AWS (future work)

> ⚠️ **Do not attempt Option A until you have:**
> - A dedicated personal AWS test account (not your organisation's account)
> - Billing alerts configured
> - Reviewed the AWS Free Tier limits
> - Read the cost estimate below

### AWS Cost Estimate (Option A)

| Service | Usage | Estimated monthly cost |
|---|---|---|
| CloudTrail | First trail per region (free tier) | $0 |
| S3 | < 1 GB log storage | ~$0.02 |
| KMS | 1 customer-managed key (optional) | $1.00 |
| **Total** | | **~$1–2/month** |

### Prerequisites for Option A

1. Create a free-tier AWS account at https://aws.amazon.com/free/
2. Enable MFA on the root account immediately
3. Create an IAM user with least-privilege permissions (do NOT use root)
4. Required IAM permissions:
   - `cloudtrail:DescribeTrails`
   - `cloudtrail:GetTrailStatus`
   - `cloudtrail:LookupEvents`
   - `s3:GetObject` (on the CloudTrail S3 bucket only)
5. Enable CloudTrail in at least one region

### Configuration for Option A

Create a `.env` file in the project root (this file is git-ignored):

```
# .env — NEVER COMMIT THIS FILE
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
CLOUDTRAIL_BUCKET=your-cloudtrail-bucket-name
```

Add entries for known-safe IPs to `config/ip_allowlist.yaml`:

```yaml
known_prefixes:
  - "your.office.ip."
  - "your.vpn.range."
```

Uncomment `boto3` and `python-dotenv` in `requirements.txt` and run:

```powershell
pip install -r requirements.txt
```

### Security reminders

- **Never commit `.env` or any file containing AWS credentials**
- Use the minimum IAM permissions required (principle of least privilege)
- Delete the test account when the project is complete
- Set a billing alert at $5/month to prevent unexpected charges

---

## Schema Mapping Reference

| CloudTrail JSON field | Project schema column | Notes |
|---|---|---|
| `eventTime` | `event_time` | Direct copy |
| `eventSource` | `event_source` | Lowercased |
| `eventName` | `event_name` | Lowercased |
| `sourceIPAddress` | `source_ip` | Set to "REDACTED" after computing `unusual_location` |
| `userIdentity.userName` | `user_identity` | Pseudonymised to `user_N` |
| `awsRegion` | `aws_region` | Lowercased; defaults to "us-east-1" if missing |
| `errorCode` | `error_code` | Empty string if null |
| `eventID` | `event_id` | Direct copy |
| _(derived)_ | `privilege_change` | 1 if `eventName` in `PRIVILEGE_CHANGE_ACTIONS` |
| _(derived)_ | `unusual_location` | 1 if IP not in `ip_allowlist.yaml` prefixes |
| _(sentinel)_ | `label` | Always -1 (unknown — NOT 0 or 1) |
| _(set)_ | `data_source` | Always "cloudtrail_real" |

**Dropped fields** (not stored): `requestParameters`, `responseElements`,
`resources`, `userIdentity.accountId`, `userIdentity.arn`.
