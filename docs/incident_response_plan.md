# Incident Response Plan
## AI-Powered Cloud Security Threat Detection and Analytics System
### Wipro CPPE Project — Phase 5: Security Engineering

> **⚠️ PROJECT DEMONSTRATION NOTICE**
> This document describes a sample incident response procedure for
> the Wipro CPPE college project. It is written to demonstrate knowledge
> of incident response principles. It has **not been tested, reviewed,
> or approved** for use in any real production environment. Do not use
> this procedure to respond to real security incidents without expert review.

---

## 1. Incident Severity Levels

| Level | Description | Example | Response Time |
|---|---|---|---|
| **Critical** | Active, ongoing compromise | Root account brute-force in progress | Immediate |
| **High** | Confirmed suspicious activity | Privilege escalation detected | < 1 hour |
| **Medium** | Probable suspicious activity | Unusual S3 access from new IP | < 4 hours |
| **Low** | Anomalous but unconfirmed | Single off-hours login | < 24 hours |

---

## 2. Incident Lifecycle (Implemented in This Dashboard)

```
Detected
    │
    ▼
  [New]  ──────────────────────────────────────────────────┐
    │                                                       │
    ▼                                                       │
[Investigating]  ─────────────────────────────────────┐   │
    │                                                   │   │
    ▼                                                   ▼   ▼
[Resolved]                                    [False Positive]
```

Status transitions are enforced via allowlist validation in `input_validator.py`.

---

## 3. Response Phases

### Phase 1: Detection and Triage

**Trigger:** The ML model classifies one or more log events as suspicious
(predicted_label = 1, threat_probability ≥ 0.70).

**Steps:**
1. Open the Security Overview page and check the open incident count.
2. Go to Log Upload & Detection. Review the prediction results table.
3. Sort by `threat_probability` descending to identify highest-priority records.
4. For each High-threat event, note:
   - `event_name` — what action was taken
   - `user_identity` — which account was used
   - `source_ip` — where it came from
   - `aws_region` — which region was affected
   - `event_time` — when it occurred
5. Create an incident using the Incident Management page.
   - Set severity to match the threat level.
   - Paste the `event_id` into the incident form for traceability.
6. Update the incident status from **New** to **Investigating**.

---

### Phase 2: Investigation

**Goal:** Determine whether the alert is a true positive or false positive.

**Steps:**
1. Review the event details in `raw_logs` via the Security Overview page.
2. Check the user's activity history:
   - Are there many similar events from the same user?
   - Are there correlated events (e.g., privilege change followed by data exfiltration)?
3. Check the IP address:
   - Is it in the expected office range or an unusual location?
4. For real AWS environments (Phase 6): query CloudTrail for the full event chain.
5. Document your findings in the incident description field.

**Decision criteria:**
- Confirmed malicious → proceed to Containment
- Likely false positive → update status to **False Positive** and close

---

### Phase 3: Containment

> In a real AWS environment (Phase 6), the following would apply.
> In this local demo, no AWS actions are available.

**Short-term containment (real environment):**
- Disable the affected IAM user via `aws iam update-login-profile --no-password-reset-required`
- Revoke temporary credentials: invalidate STS sessions
- Isolate affected EC2 instances from the VPC if compromised
- Block the source IP at the security group level

**Dashboard action (Phase 4/5):**
- Update incident status to **Investigating**
- Add a note in the description: "Containment pending Phase 6 AWS integration"

---

### Phase 4: Eradication and Recovery

> In a real AWS environment (Phase 6).

**Steps:**
1. Identify and remove all malicious changes made by the attacker:
   - Revoke any IAM policies added during the attack
   - Delete any unauthorized users created
   - Restore modified S3 bucket ACLs from known-good state
2. Rotate credentials for any accounts that may have been compromised
3. Re-enable CloudTrail if it was stopped
4. Verify the environment is clean before restoring service

---

### Phase 5: Lessons Learned

**After every incident, document:**
1. Timeline of the attack (from first event to containment)
2. What the ML model detected correctly
3. What the ML model missed (false negatives)
4. Which features were most predictive
5. Recommended updates to detection rules or model training data

**Update this dashboard:**
- Add new attack patterns to the synthetic dataset (retrain model)
- Update the threat model to include the new scenario
- Update the incident response plan

---

## 4. Sample Procedure: Suspicious Cloud Activity Alert

**Scenario:** The dashboard detects an event classified as High threat with
the following properties:

```
event_time    : 2024-06-15T03:17:00Z  (off-hours: 3 AM UTC)
event_source  : iam.amazonaws.com
event_name    : AttachUserPolicy
source_ip     : 203.0.113.99         (unusual location)
user_identity : root
aws_region    : ap-northeast-1
error_code    : AccessDenied
failed_attempts : 8
```

**Step-by-step response:**

1. **Triage (5 minutes)**
   - Open Incident Management → Create incident
   - Title: "Suspicious IAM policy attachment from unusual IP"
   - Severity: High
   - Paste the event_id
   - Status: Investigating

2. **Investigate (15 minutes)**
   - Search `raw_logs` for other events from the same `source_ip` in the same time window
   - Check for prior brute-force pattern: `event_name=GetUser` + `error_code=AccessDenied`
   - Check if `AttachUserPolicy` succeeded (no error) or failed (AccessDenied)
   - In a real environment: use `aws iam list-attached-user-policies` to verify

3. **Contain (if confirmed, real environment)**
   - Disable root account MFA device temporarily to regain control
   - Revoke all root active sessions
   - Change root account password immediately
   - Enable CloudTrail multi-region trail if not already enabled

4. **Recover**
   - Audit all IAM policies modified since the attack timestamp
   - Re-apply known-good policy baseline

5. **Close**
   - Update incident status to **Resolved**
   - Document final findings in incident description
   - Add lessons learned

---

## 5. Dashboard Incident Management Reference

| Action | How |
|---|---|
| Create incident | Incident Management page → "Create a new incident" form |
| View all incidents | Incident Management page → Incident List |
| Filter by status | Use the "Filter by status" dropdown |
| Update status | Incident Management page → "Update Incident Status" section |
| Download predictions | Log Upload & Detection → "Download predictions as CSV" |

---

*This document is a student project demonstration only.*
*Do not use for real incident response without expert review.*
