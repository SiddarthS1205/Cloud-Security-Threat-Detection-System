"""
app.py
======
Phase 4–6 — Streamlit Security Dashboard

AI-Powered Cloud Security Threat Detection and Analytics System
Wipro CPPE College Project

Run with:
    streamlit run dashboard/app.py

DATA NOTICE
-----------
The database contains two types of records:
  - 5,000 synthetic CloudTrail-style records (label=0 normal / label=1 suspicious)
    generated programmatically for development and testing.
  - Up to a small number of anonymised CloudTrail-format sample records
    (label=-1, meaning unlabelled — not confirmed normal or suspicious)
    ingested via the Phase 6 offline pipeline.

No live AWS connection is active. The ML model was trained on synthetic data
and its evaluation metrics do not establish real-world detection performance.
"""

import sys
import os

# Ensure the project root is on the path regardless of how Streamlit is launched
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.db_utils import (
    DB_PATH,
    VALID_STATUSES,
    get_connection,
    get_event_counts,
    get_events_by_source,
    get_events_by_hour,
    get_top_suspicious_users,
    get_recent_events,
    get_incidents,
    get_open_incident_count,
    get_incident_status_counts,
    create_incident,
    update_incident_status,
)
from dashboard.csv_validator import (
    validate_upload,
    prepare_for_prediction,
    REQUIRED_UPLOAD_COLUMNS,
)
from src.ml.predictor import load_model, load_feature_engineer, predict

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cloud Security Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource loaders (loaded once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ML model…")
def _load_clf():
    return load_model()

@st.cache_resource(show_spinner="Loading feature engineer…")
def _load_fe():
    return load_feature_engineer()

@st.cache_data(ttl=30)
def _get_conn_counts():
    conn = get_connection()
    counts = get_event_counts(conn)
    open_inc = get_open_incident_count(conn)
    conn.close()
    return counts, open_inc

# ---------------------------------------------------------------------------
# Colour palette helpers
# ---------------------------------------------------------------------------

COLOUR_NORMAL = "#2ecc71"
COLOUR_SUSPICIOUS = "#e74c3c"
COLOUR_HIGH = "#e74c3c"
COLOUR_MEDIUM = "#f39c12"
COLOUR_LOW = "#2ecc71"

STATUS_COLOURS = {
    "New": "#e74c3c",
    "Investigating": "#f39c12",
    "Resolved": "#2ecc71",
    "False Positive": "#95a5a6",
}

# ---------------------------------------------------------------------------
# Data-source banners — shown on each page, accurate to what's in the DB
# ---------------------------------------------------------------------------

def _data_notice_banner() -> None:
    """
    Display an accurate data-source notice.

    - If the DB contains only synthetic records, says so clearly.
    - If the DB also contains CloudTrail sample records, names both sources
      and states that CloudTrail records are unlabelled, not confirmed threats.
    - Always states there is no live AWS connection.
    """
    try:
        counts, _ = _get_conn_counts()
        unknown = counts.get("unknown", 0)
    except Exception:
        unknown = 0

    if unknown > 0:
        st.info(
            f"ℹ️ **Data sources in this database:**  \n"
            f"• **5,000 synthetic records** — generated programmatically for the "
            f"Wipro CPPE project (label: 0 = normal, 1 = suspicious).  \n"
            f"• **{unknown:,} CloudTrail-format sample record"
            f"{'s' if unknown != 1 else ''}** — anonymised, ingested via the "
            f"Phase 6 offline pipeline (label: −1 = unlabelled, "
            f"not confirmed normal or suspicious).  \n"
            f"**No live AWS connection is active.**",
            icon="ℹ️",
        )
    else:
        st.warning(
            "⚠️ **Synthetic Data** — All records in this database were generated "
            "programmatically for the Wipro CPPE project. "
            "This dashboard does not connect to AWS and does not display real cloud activity.",
            icon="⚠️",
        )


def _prediction_notice() -> None:
    """Banner for pages that show ML predictions from uploaded files."""
    st.warning(
        "⚠️ **Synthetic Training Data** — The model was trained on synthetic data. "
        "Threat scores shown below are **model predictions**, not confirmed attacks. "
        "This dashboard does not connect to AWS.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Page 1: Security Overview
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.title("🛡️ Security Overview")
    _data_notice_banner()

    counts, open_inc = _get_conn_counts()

    # --- KPI row ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Events", f"{counts['total']:,}")
    c2.metric("Normal Events", f"{counts['normal']:,}",
              help="Confirmed normal (synthetic, label=0)")
    c3.metric("Suspicious Events", f"{counts['suspicious']:,}",
              delta=f"{counts['suspicious']} detected",
              delta_color="inverse")
    c4.metric("Unknown (Real CT)", f"{counts.get('unknown', 0):,}",
              help="Unlabelled real CloudTrail events (label=-1). "
                   "Not confirmed normal or suspicious.")
    c5.metric("Open Incidents", f"{open_inc}",
              delta="requires attention" if open_inc > 0 else "all clear",
              delta_color="inverse" if open_inc > 0 else "normal")

    st.divider()

    # --- Charts row ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Events by Class")
        unknown_count = counts.get("unknown", 0)
        pie_names   = ["Normal", "Suspicious"]
        pie_values  = [counts["normal"], counts["suspicious"]]
        pie_colours = {"Normal": COLOUR_NORMAL, "Suspicious": COLOUR_SUSPICIOUS}
        if unknown_count > 0:
            pie_names.append("Unknown (Real CT)")
            pie_values.append(unknown_count)
            pie_colours["Unknown (Real CT)"] = "#95a5a6"
        fig_pie = px.pie(
            names=pie_names,
            values=pie_values,
            color=pie_names,
            color_discrete_map=pie_colours,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Events by Hour of Day")
        conn = get_connection()
        df_hour = get_events_by_hour(conn)
        conn.close()
        if not df_hour.empty:
            df_hour["class"] = df_hour["label"].map(
                {0: "Normal", 1: "Suspicious"}
            )
            fig_hour = px.bar(
                df_hour, x="hour", y="count", color="class",
                color_discrete_map={
                    "Normal": COLOUR_NORMAL,
                    "Suspicious": COLOUR_SUSPICIOUS,
                },
                labels={"hour": "Hour of Day (UTC)", "count": "Event Count"},
                barmode="stack",
            )
            fig_hour.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_hour, use_container_width=True)

    st.divider()

    # --- Events by service ---
    st.subheader("Events by AWS Service")
    conn = get_connection()
    df_src = get_events_by_source(conn)
    conn.close()
    if not df_src.empty:
        df_src["class"] = df_src["label"].map({0: "Normal", 1: "Suspicious"})
        fig_src = px.bar(
            df_src, x="event_source", y="count", color="class",
            color_discrete_map={
                "Normal": COLOUR_NORMAL,
                "Suspicious": COLOUR_SUSPICIOUS,
            },
            labels={
                "event_source": "AWS Service",
                "count": "Event Count",
            },
            barmode="group",
        )
        fig_src.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_src, use_container_width=True)

    st.divider()

    # --- Top suspicious users ---
    col_users, col_recent = st.columns(2)
    with col_users:
        st.subheader("Top Suspicious Users")
        conn = get_connection()
        df_users = get_top_suspicious_users(conn)
        conn.close()
        if not df_users.empty:
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("No suspicious users in the dataset.")

    with col_recent:
        st.subheader("Recent Events")
        conn = get_connection()
        df_recent = get_recent_events(conn, n=15)
        conn.close()
        if not df_recent.empty:
            df_recent["label"] = df_recent["label"].map(
                {0: "✅ Normal", 1: "🚨 Suspicious", -1: "❓ Unknown (CT sample)"}
            ).fillna("❓ Unknown")
            st.dataframe(df_recent, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 2: Log Upload & Threat Detection
# ---------------------------------------------------------------------------

def page_detection() -> None:
    st.title("🔍 Log Upload & Threat Detection")
    _prediction_notice()

    st.markdown(
        "Upload a CSV file of CloudTrail-style log records to score them "
        "against the trained model. The model was trained on synthetic data "
        "and **will not be retrained** during this process. "
        "Scores are **model predictions** — not confirmed attack detections."
    )

    # --- Expected schema hint ---
    with st.expander("📋 Expected CSV columns"):
        st.markdown(
            "Your CSV must contain these columns (case-sensitive):\n\n"
            + "\n".join(f"- `{c}`" for c in REQUIRED_UPLOAD_COLUMNS)
            + "\n\n**Optional columns** (added automatically if missing): "
            "`event_id`, `data_source`, `ingested_at`"
        )
        st.markdown(
            "**Example row:**\n"
            "```\n"
            "event_time,event_source,event_name,source_ip,user_identity,"
            "aws_region,error_code,failed_attempts,unusual_location,privilege_change\n"
            "2024-06-15T10:23:00Z,s3.amazonaws.com,GetObject,192.0.2.10,"
            "alice@example-corp.com,us-east-1,,0,0,0\n"
            "```"
        )

    uploaded_file = st.file_uploader(
        "Upload a CSV log file",
        type=["csv"],
        help="Maximum 5,000 rows per file.",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file above to start threat detection.")
        return

    # --- Load and validate ---
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    result = validate_upload(df_raw)

    if result.errors:
        for err in result.errors:
            st.error(f"❌ {err}")
        return

    for warn in result.warnings:
        st.warning(f"⚠️ {warn}")

    st.success(f"✅ File validated: {result.row_count} rows, "
               f"{len(REQUIRED_UPLOAD_COLUMNS)} required columns found.")

    # --- Run prediction ---
    try:
        clf = _load_clf()
        fe = _load_fe()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    df_prep = prepare_for_prediction(df_raw)

    with st.spinner("Running threat detection…"):
        try:
            pred_df = predict(df_prep, clf=clf, fe=fe)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

    # Merge predictions back with the original columns for display
    if "event_id" in df_prep.columns:
        display_df = df_prep.merge(pred_df, on="event_id", how="left")
    else:
        display_df = pd.concat(
            [df_prep.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1,
        )

    # --- Summary metrics ---
    n_suspicious = (pred_df["predicted_label"] == 1).sum()
    n_high = (pred_df["threat_level"] == "High").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Records Scored", f"{len(pred_df):,}")
    c2.metric(
        "Predicted Suspicious",
        f"{n_suspicious:,}",
        delta_color="inverse",
        help="Model predicted label=1. Not a confirmed attack — requires human review.",
    )
    c3.metric(
        "Predicted High Threat",
        f"{n_high:,}",
        delta_color="inverse",
        help="Threat probability ≥ 0.70. Not a confirmed attack — requires human review.",
    )
    if n_suspicious > 0 or n_high > 0:
        st.caption(
            "ℹ️ **Prediction disclaimer:** 'Predicted Suspicious' and "
            "'Predicted High Threat' are outputs of the ML model trained on "
            "synthetic data. They are **not confirmed attacks**. "
            "Each flagged record should be reviewed by a qualified analyst."
        )

    st.divider()

    # --- Threat level breakdown chart ---
    st.subheader("Threat Level Distribution")
    level_counts = pred_df["threat_level"].value_counts().reset_index()
    level_counts.columns = ["threat_level", "count"]
    fig_lvl = px.bar(
        level_counts, x="threat_level", y="count",
        color="threat_level",
        color_discrete_map={
            "High": COLOUR_HIGH, "Medium": COLOUR_MEDIUM, "Low": COLOUR_LOW
        },
        labels={"threat_level": "Threat Level", "count": "Records"},
    )
    fig_lvl.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_lvl, use_container_width=True)

    # --- Results table ---
    st.subheader("Prediction Results")
    show_cols = [
        c for c in [
            "event_id", "event_time", "event_source", "event_name",
            "user_identity", "aws_region", "unusual_location",
            "privilege_change", "error_code",
            "predicted_label", "threat_probability", "threat_level",
        ]
        if c in display_df.columns
    ]
    result_display = display_df[show_cols].copy()
    result_display["predicted_label"] = result_display["predicted_label"].map(
        {0: "✅ Normal", 1: "🚨 Suspicious"}
    )
    st.dataframe(result_display, use_container_width=True, hide_index=True)

    # --- Download results ---
    csv_out = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download predictions as CSV",
        data=csv_out,
        file_name="threat_predictions.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Page 3: Model Performance
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    st.title("📊 Model Performance")
    _data_notice_banner()

    # --- Load metadata ---
    meta_path = Path("models") / "training_metadata.json"
    report_path = Path("reports") / "model_evaluation.txt"

    if not meta_path.exists():
        st.error("training_metadata.json not found. Run the trainer first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    st.info(
        "ℹ️ **Source:** All metrics on this page are loaded from "
        "`reports/model_evaluation.txt`, which was generated during Phase 3 "
        "training on the **synthetic test set only**. "
        "These figures reflect synthetic pattern learning. "
        "They do **not** demonstrate real-world cloud security detection performance. "
        "No metrics are recalculated from uploaded prediction results."
    )

    # --- Model info ---
    col_m, col_d = st.columns(2)
    with col_m:
        st.subheader("Model Configuration")
        params = meta.get("model_params", {})
        st.table(
            pd.DataFrame(
                list(params.items()), columns=["Parameter", "Value"]
            )
        )

    with col_d:
        st.subheader("Training Data Summary")
        st.table(
            pd.DataFrame(
                [
                    ["Total records", "5,000"],
                    ["Training rows", f"{meta.get('train_rows', 'N/A'):,}"],
                    ["Test rows", f"{meta.get('test_rows', 'N/A'):,}"],
                    ["Train suspicious", f"{meta.get('train_suspicious', 'N/A'):,}"],
                    ["Test suspicious", f"{meta.get('test_suspicious', 'N/A'):,}"],
                    ["Random seed", str(meta.get("random_seed", "N/A"))],
                    ["Data source", meta.get("data_source", "synthetic")],
                ],
                columns=["Item", "Value"],
            )
        )

    st.divider()

    # --- Confusion matrix from report ---
    st.subheader("Confusion Matrix (test set)")
    st.markdown(
        "Each cell shows how many records were predicted vs. their actual class."
    )

    # Parse confusion matrix from the text report
    cm_data = None
    if report_path.exists():
        txt = report_path.read_text(encoding="utf-8")
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            if "Actual Normal" in line:
                try:
                    parts_n = lines[i].split()
                    parts_s = lines[i + 1].split()
                    tn = int(parts_n[-2])
                    fp = int(parts_n[-1])
                    fn = int(parts_s[-2])
                    tp = int(parts_s[-1])
                    cm_data = [[tn, fp], [fn, tp]]
                except (ValueError, IndexError):
                    pass
                break

    if cm_data:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted Normal", "Predicted Suspicious"],
            y=["Actual Normal", "Actual Suspicious"],
            text=[[str(v) for v in row] for row in cm_data],
            texttemplate="%{text}",
            textfont={"size": 18},
            colorscale="Blues",
            showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            margin=dict(t=30, b=30),
            height=300,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Positives",  f"{cm_data[1][1]}",
                  help="Suspicious correctly identified")
        c2.metric("True Negatives",  f"{cm_data[0][0]}",
                  help="Normal correctly identified")
        c3.metric("False Positives", f"{cm_data[0][1]}",
                  help="Normal wrongly flagged as suspicious")
        c4.metric("False Negatives", f"{cm_data[1][0]}",
                  help="Suspicious missed by the model")
        st.caption(
            "Values loaded from `reports/model_evaluation.txt` "
            "(Phase 3 synthetic test set, 1,000 held-out rows, seed=42). "
            "Not recalculated from any uploaded file."
        )
    else:
        st.warning("Could not parse confusion matrix from the evaluation report.")

    st.divider()

    # --- Feature importances ---
    st.subheader("Feature Importances")
    st.markdown(
        "How much each feature contributed to the model's decisions "
        "(mean decrease in impurity across all trees)."
    )
    try:
        clf = _load_clf()
        from src.data_pipeline.features import FEATURE_COLUMNS
        imp_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": clf.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Mean Decrease in Impurity"},
        )
        fig_imp.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Load the model to see feature importances.")

    st.divider()

    # --- Full text report ---
    if report_path.exists():
        with st.expander("📄 Full evaluation report (text)"):
            st.code(
                report_path.read_text(encoding="utf-8"),
                language="text",
            )


# ---------------------------------------------------------------------------
# Page 4: Incident Management
# ---------------------------------------------------------------------------

def page_incidents() -> None:
    st.title("🚨 Incident Management")
    _data_notice_banner()

    conn = get_connection()

    # --- Status summary ---
    status_counts = get_incident_status_counts(conn)
    c1, c2, c3, c4 = st.columns(4)
    for col_widget, status in zip([c1, c2, c3, c4], VALID_STATUSES):
        col_widget.metric(status, status_counts.get(status, 0))

    st.divider()

    # --- Create new incident ---
    with st.expander("➕ Create a new incident"):
        with st.form("create_incident_form", clear_on_submit=True):
            title = st.text_input("Incident title *", max_chars=200)
            event_id = st.text_input(
                "Event ID (optional — paste from a prediction result)"
            )
            description = st.text_area("Description", height=100)
            severity = st.selectbox(
                "Severity", ["Low", "Medium", "High", "Critical"],
                index=1,
            )
            submitted = st.form_submit_button("Create Incident")

        if submitted:
            if not title.strip():
                st.error("Incident title is required.")
            else:
                result = create_incident(
                    conn,
                    event_id=event_id.strip() or "manual",
                    title=title.strip(),
                    description=description.strip(),
                    severity=severity,
                )
                if result["already_existed"]:
                    st.warning(
                        f"An incident with this title and event ID already "
                        f"exists (ID: {result['incident_id']})."
                    )
                else:
                    st.success(
                        f"✅ Incident #{result['incident_id']} created."
                    )
                _get_conn_counts.clear()

    st.divider()

    # --- Filter and display ---
    st.subheader("Incident List")
    status_filter = st.selectbox(
        "Filter by status",
        ["All"] + list(VALID_STATUSES),
        index=0,
    )
    df_inc = get_incidents(
        conn,
        status_filter=None if status_filter == "All" else status_filter,
    )

    if df_inc.empty:
        st.info("No incidents found. Create one using the form above.")
        conn.close()
        return

    # Display with status colour highlighting
    def _colour_status(val: str) -> str:
        colour = STATUS_COLOURS.get(val, "#ffffff")
        return f"background-color: {colour}22; color: {colour}; font-weight: bold"

    styled = df_inc.style.applymap(
        _colour_status, subset=["status"]
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # --- Update status ---
    st.subheader("Update Incident Status")
    if not df_inc.empty:
        inc_options = {
            f"#{row['incident_id']} — {row['title']} [{row['status']}]": row["incident_id"]
            for _, row in df_inc.iterrows()
        }
        selected_label = st.selectbox("Select incident", list(inc_options.keys()))
        selected_id = inc_options[selected_label]
        new_status = st.selectbox("New status", VALID_STATUSES, key="new_status_select")

        if st.button("Update Status"):
            ok = update_incident_status(conn, selected_id, new_status)
            if ok:
                st.success(f"Incident #{selected_id} status updated to **{new_status}**.")
                _get_conn_counts.clear()
            else:
                st.error("Update failed. Invalid incident ID or status.")

    conn.close()


# ---------------------------------------------------------------------------
# Page 5: CPPE Module Overview
# ---------------------------------------------------------------------------

def page_cppe() -> None:
    st.title("📚 CPPE Module Overview")

    st.markdown(
        "This project covers all four Wipro CPPE modules. "
        "The table below maps each feature to its module and shows "
        "whether it works locally or requires AWS."
    )

    st.divider()

    modules = {
        "🤖 AI Engineering": {
            "colour": "#3498db",
            "features": [
                ("Synthetic data labelling strategy", "Local ✅"),
                ("RandomForestClassifier training", "Local ✅"),
                ("Stratified train/test split (80/20)", "Local ✅"),
                ("Precision / Recall / F1 / Confusion Matrix", "Local ✅"),
                ("Feature importance ranking", "Local ✅"),
                ("Batch and single-record prediction", "Local ✅"),
                ("Streamlit prediction interface", "Local ✅"),
            ],
        },
        "🗄️ Data Engineering": {
            "colour": "#2ecc71",
            "features": [
                ("Synthetic CloudTrail-style log generator", "Local ✅"),
                ("Schema validation pipeline", "Local ✅"),
                ("Data cleaning and normalisation", "Local ✅"),
                ("Feature engineering (12 ML features)", "Local ✅"),
                ("SQLite storage with upsert safety", "Local ✅"),
                ("CSV upload and validation", "Local ✅"),
                ("Pipeline re-run safety (idempotent)", "Local ✅"),
            ],
        },
        "☁️ Cloud Engineering": {
            "colour": "#9b59b6",
            "features": [
                ("AWS architecture design document", "Local ✅"),
                ("CloudTrail-compatible schema", "Local ✅"),
                ("CloudTrail JSON parser (offline, Option B)", "Local ✅"),
                ("CloudTrail anonymiser (pseudonymisation)", "Local ✅"),
                ("CloudTrail schema mapper + ingestor", "Local ✅"),
                ("AWS S3 log ingestion (live)", "Phase 6 Option A — AWS required 🔜"),
                ("AWS CloudTrail live reading", "Phase 6 Option A — AWS required 🔜"),
                ("AWS CloudWatch alerting", "Phase 6 Option A — AWS required 🔜"),
                ("AWS KMS encryption", "Phase 6 Option A — AWS required 🔜"),
                ("IAM least-privilege role design", "Phase 6 Option A — AWS required 🔜"),
            ],
        },
        "🔒 Security Engineering": {
            "colour": "#e74c3c",
            "features": [
                ("STRIDE threat model document", "Phase 5 — Planned 📋"),
                ("Incident response workflow", "Phase 5 — Planned 📋"),
                ("Input validation and sanitisation", "Local ✅"),
                ("Incident tracking (New/Investigating/Resolved)", "Local ✅"),
                ("Audit trail (pipeline_runs table)", "Local ✅"),
                ("No hardcoded credentials policy", "Local ✅"),
                ("Synthetic data clearly labelled", "Local ✅"),
            ],
        },
    }

    for module_name, module_data in modules.items():
        st.subheader(module_name)
        rows = []
        for feature, status in module_data["features"]:
            rows.append({"Feature": feature, "Status": status})
        df_mod = pd.DataFrame(rows)

        def _colour_status(val: str) -> str:
            if "✅" in val:
                return "color: #2ecc71; font-weight: bold"
            if "🔜" in val:
                return "color: #9b59b6"
            return "color: #f39c12"

        styled = df_mod.style.applymap(_colour_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown("")

    st.divider()
    st.subheader("AWS Cost & Permission Summary (Phase 6)")
    st.markdown("""
| AWS Service | Usage | Estimated Cost |
|---|---|---|
| S3 | < 1 GB of log files | ~\\$0.02/month |
| CloudTrail | First trail per region | \\$0 (free tier) |
| CloudWatch | Basic monitoring | \\$0 (free tier) |
| KMS | 1 customer-managed key | \\$1.00/month |
| **Total** | | **~\\$1–2/month** |

> **Phase 6 Option B (offline) is active.** The CloudTrail parser, anonymiser,
> and schema mapper work without AWS credentials using local JSON files.
> Live AWS ingestion (Option A) requires `boto3` and IAM credentials,
> isolated in `src/cloud/` and not imported by this dashboard.
    """)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def main() -> None:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
        st.title("Cloud Security")
        st.caption("Wipro CPPE Project")
        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "🏠 Security Overview",
                "🔍 Log Upload & Detection",
                "📊 Model Performance",
                "🚨 Incident Management",
                "📚 CPPE Module Overview",
            ],
            index=0,
        )

        st.divider()
        # Dynamic data source summary — reflects what is actually in the database
        try:
            _counts, _ = _get_conn_counts()
            _unknown = _counts.get("unknown", 0)
            if _unknown > 0:
                _data_line = (
                    f"**Data:** 5,000 synthetic records + "
                    f"{_unknown:,} unlabelled CloudTrail sample record"
                    f"{'s' if _unknown != 1 else ''}\n\n"
                    "*(CloudTrail records: label unknown, not confirmed normal)*"
                )
            else:
                _data_line = "**Data:** 5,000 synthetic records"
        except Exception:
            _data_line = "**Data:** Synthetic + optional CloudTrail sample"
        st.caption(
            f"{_data_line}\n\n"
            "**AWS:** Not connected (Phase 6)\n\n"
            "**Model:** RandomForest (local)"
        )

    if page == "🏠 Security Overview":
        page_overview()
    elif page == "🔍 Log Upload & Detection":
        page_detection()
    elif page == "📊 Model Performance":
        page_model_performance()
    elif page == "🚨 Incident Management":
        page_incidents()
    elif page == "📚 CPPE Module Overview":
        page_cppe()


if __name__ == "__main__":
    main()
