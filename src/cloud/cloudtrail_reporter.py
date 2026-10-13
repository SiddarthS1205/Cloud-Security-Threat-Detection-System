"""
cloudtrail_reporter.py
======================
Phase 6 — Cloud Engineering: generates evaluation report for ingested
CloudTrail records.

WHAT THIS REPORT CONTAINS
--------------------------
- Event counts by data_source (synthetic vs cloudtrail_real)
- Prediction probability distribution for real events
- Top-10 highest-probability events (for manual review)
- Explicit statement that precision, recall, F1, and accuracy are NOT
  calculated because ground-truth labels are not available

WHAT THIS REPORT DOES NOT CONTAIN
-----------------------------------
- Precision, recall, F1, or accuracy scores
- Any claim that the model performs at a specific detection rate
- Comparison between model performance on synthetic vs real data
  (would require trustworthy ground-truth labels for real data)

The report format is consistent with reports/model_evaluation.txt but
clearly scoped to schema-mapping and prediction integration only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cloud.cloudtrail_parser import LABEL_UNKNOWN

REPORTS_DIR = Path("reports")
CLOUDTRAIL_REPORT_FILE = REPORTS_DIR / "cloudtrail_evaluation.txt"

LABEL_MAP = {0: "Normal (synthetic)", 1: "Suspicious (synthetic)", LABEL_UNKNOWN: "Unknown (unlabelled)"}


def generate_report(
    predictions: pd.DataFrame,
    ingestion_summary: str = "",
    output_path: Path = CLOUDTRAIL_REPORT_FILE,
) -> str:
    """
    Generate a plain-text report for CloudTrail prediction results.

    Parameters
    ----------
    predictions      : DataFrame from predictor.predict() with columns
                       event_id, predicted_label, threat_probability, threat_level
    ingestion_summary: text from IngestionReport.summary() (optional)
    output_path      : where to write the report

    Returns
    -------
    str : the full report text (also written to output_path)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "=" * 70,
        "AI-POWERED CLOUD SECURITY THREAT DETECTION",
        "CloudTrail Evaluation Report — Phase 6",
        f"Generated: {now}",
        "=" * 70,
        "",
        "DATA SOURCE NOTICE",
        "-" * 70,
        "Records in this report come from CloudTrail-formatted JSON files",
        "(Option B: local file, no live AWS connection).",
        "User identities are pseudonymised (user_1, user_2, ...).",
        "Source IP addresses are replaced with 'REDACTED'.",
        "The original-to-pseudonym mapping was never stored.",
        "",
        "MODEL NOTICE",
        "-" * 70,
        "The model (threat_classifier.pkl) was trained on SYNTHETIC data.",
        "It has NOT been retrained on real CloudTrail records.",
        "Predictions reflect synthetic pattern matching only.",
        "",
        "EVALUATION SCOPE",
        "-" * 70,
        "This report covers:",
        "  1. Schema validation: CloudTrail fields mapped to project schema.",
        "  2. Feature derivation: privilege_change and unusual_location computed.",
        "  3. Prediction integration: model produces probabilities for each event.",
        "",
        "This report does NOT cover:",
        "  - Precision, recall, F1-score, or accuracy.",
        "  - False positive or false negative rates.",
        "  - Any claim about real-world threat detection performance.",
        "",
        "REASON: ground-truth labels are not available for these events.",
        "Classification metrics (precision, recall, F1, accuracy) are NOT",
        "CALCULATED because they require trustworthy labels assigned by a",
        "qualified security analyst. Without such labels, metric calculation",
        "would produce meaningless or misleading numbers.",
        "",
    ]

    if ingestion_summary:
        lines += [
            "INGESTION SUMMARY",
            "-" * 70,
            ingestion_summary,
            "",
        ]

    if predictions.empty:
        lines += [
            "PREDICTIONS",
            "-" * 70,
            "No predictions available (ingestion produced 0 records).",
        ]
    else:
        n_total = len(predictions)
        n_high = (predictions["threat_level"] == "High").sum()
        n_medium = (predictions["threat_level"] == "Medium").sum()
        n_low = (predictions["threat_level"] == "Low").sum()

        lines += [
            "PREDICTION SUMMARY",
            "-" * 70,
            f"  Total events scored : {n_total:,}",
            f"  Threat level High   : {n_high:,}  (probability >= 0.70)",
            f"  Threat level Medium : {n_medium:,}  (probability 0.40–0.69)",
            f"  Threat level Low    : {n_low:,}  (probability < 0.40)",
            "",
            "  NOTE: These threat levels are model predictions based on",
            "  synthetic training patterns. They are NOT confirmed detections.",
            "  High threat_level means the model assigned a high probability",
            "  — not that the event is proven malicious.",
            "",
        ]

        # Probability distribution
        probs = predictions["threat_probability"]
        lines += [
            "PROBABILITY DISTRIBUTION",
            "-" * 70,
            f"  Min    : {probs.min():.4f}",
            f"  Median : {probs.median():.4f}",
            f"  Mean   : {probs.mean():.4f}",
            f"  Max    : {probs.max():.4f}",
            "",
        ]

        # Top-10 highest probability events for manual review
        top10 = predictions.nlargest(10, "threat_probability")[
            ["event_id", "threat_probability", "threat_level"]
        ].reset_index(drop=True)

        lines += [
            "TOP-10 HIGHEST PROBABILITY EVENTS (for manual review)",
            "-" * 70,
            "  These events warrant human review — they are NOT confirmed threats.",
            "",
        ]
        lines.append(f"  {'#':>2}  {'event_id':40s}  {'probability':>11}  {'level'}")
        lines.append(f"  {'--':>2}  {'-'*40}  {'-'*11}  -----")
        for i, row in top10.iterrows():
            lines.append(
                f"  {i+1:>2}  {str(row['event_id']):40s}  "
                f"{row['threat_probability']:>11.4f}  {row['threat_level']}"
            )

    lines += [
        "",
        "KNOWN LIMITATIONS",
        "-" * 70,
        "  1. Model trained on synthetic data only (5 event types, 6 AWS services).",
        "     Novel event names and services are mapped to a placeholder category.",
        "  2. unusual_location derives from a configurable IP allowlist.",
        "     An empty allowlist means all IPs return unusual_location=0.",
        "  3. privilege_change is a rule-based approximation from event_name.",
        "     Legitimate admin actions may also set privilege_change=1.",
        "  4. Labels are UNKNOWN (-1) for all real events in this report.",
        "     The value -1 is a sentinel meaning 'not labelled'.",
        "     It does NOT mean normal or safe.",
        "  5. This system is a student project demonstration.",
        "     It is NOT production-ready and has NOT been security-reviewed.",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ]

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    print(f"[reporter] Report written to {output_path}")
    return content
