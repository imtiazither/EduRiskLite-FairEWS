"""Streamlit dashboard for educator-facing student risk review."""

from __future__ import annotations

import os
import pandas as pd
from pathlib import Path
import subprocess
import streamlit as st
import sys

from src.config import MODEL_PATH, SCORED_STUDENTS_PATH
from src.data_utils import format_feature_name, load_dataset
from src.modeling import explain_student_prediction, load_model_bundle, score_students


TOP_FEATURE_COUNT = 3
ROW_PREVIEW_FIELDS = [
    "age_at_enrollment",
    "admission_grade",
    "debtor",
    "tuition_fees_up_to_date",
    "scholarship_holder",
    "curricular_units_1st_sem_approved",
    "curricular_units_2nd_sem_approved",
]


def running_under_streamlit() -> bool:
    """Detect whether the script is already executing inside a Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False

    return get_script_run_ctx() is not None


def relaunch_with_streamlit() -> None:
    """Re-run the file with Streamlit so VS Code's Run button opens the dashboard correctly."""
    launch_environment = os.environ.copy()
    launch_environment["STREAMLIT_SERVER_SHOW_EMAIL_PROMPT"] = "false"
    launch_environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    launch_command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(subprocess.run(launch_command, check=False, env=launch_environment).returncode)


def load_dashboard_assets() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Load the trained model and a sample processed student dataset for the dashboard."""
    model_bundle = load_model_bundle(MODEL_PATH)

    # The cleaned feature frame acts as a sample processed dataset for row selection and display.
    raw_dataset = load_dataset()
    processed_students = raw_dataset[["student_id", *model_bundle["feature_columns"]]].copy()

    if SCORED_STUDENTS_PATH.exists():
        scored_students = pd.read_csv(SCORED_STUDENTS_PATH)
    else:
        scored_students = score_students(model_bundle, raw_dataset)

    return model_bundle, processed_students, scored_students


def inject_styles() -> None:
    """Apply a lightweight visual treatment for a cleaner presentation."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .eyebrow {
            color: #5b6b7a;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .hero-copy {
            color: #304255;
            font-size: 1.02rem;
            margin-bottom: 0.25rem;
        }
        .section-note {
            color: #5b6b7a;
            font-size: 0.92rem;
        }
        .feature-card {
            background: #f8fafc;
            border: 1px solid #d9e2ec;
            border-radius: 14px;
            padding: 1rem;
            min-height: 180px;
        }
        .feature-rank {
            color: #5b6b7a;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .feature-name {
            color: #102a43;
            font-size: 1rem;
            font-weight: 700;
            margin: 0.45rem 0;
        }
        .feature-direction-up {
            color: #b42318;
            font-weight: 600;
        }
        .feature-direction-down {
            color: #0f766e;
            font-weight: 600;
        }
        .feature-impact {
            color: #486581;
            font-size: 0.92rem;
            margin-top: 0.65rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_predicted_outcome(predicted_flag: int) -> str:
    """Map the binary model output into a portfolio-friendly outcome label."""
    return "Dropout" if int(predicted_flag) == 1 else "Graduate / Enrolled"


def render_support_action(risk_level: str, support_action: str) -> None:
    """Display the support suggestion with a severity that matches the risk level."""
    if risk_level == "High":
        st.error(support_action)
    elif risk_level == "Medium":
        st.warning(support_action)
    else:
        st.success(support_action)


def render_feature_cards(explanation: pd.DataFrame) -> None:
    """Show the top local drivers behind the selected prediction."""
    feature_columns = st.columns(TOP_FEATURE_COUNT)

    for position, (_, row) in enumerate(explanation.iterrows()):
        direction_class = "feature-direction-up" if row["contribution"] >= 0 else "feature-direction-down"
        impact_label = f"Impact score: {row['abs_contribution']:.3f}"

        with feature_columns[position]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-rank">Top Factor {position + 1}</div>
                    <div class="feature-name">{row['factor']}</div>
                    <div class="{direction_class}">{row['direction']}</div>
                    <div class="feature-impact">{impact_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def build_student_row_preview(selected_student_row: pd.DataFrame) -> pd.DataFrame:
    """Prepare a compact preview of the selected sample row."""
    preview_fields = [
        column_name for column_name in ROW_PREVIEW_FIELDS if column_name in selected_student_row.columns
    ]
    preview_frame = selected_student_row[preview_fields].iloc[0].reset_index()
    preview_frame.columns = ["feature", "value"]
    preview_frame["feature"] = preview_frame["feature"].apply(format_feature_name)
    return preview_frame


def main() -> None:
    """Render the educator-facing decision-support dashboard."""
    st.set_page_config(page_title="EduRiskLite-FairEWS", page_icon="🎓", layout="wide")
    inject_styles()

    st.markdown('<div class="eyebrow">Educational Data Mining • Learning Analytics • Responsible AI</div>', unsafe_allow_html=True)
    st.title("EduRiskLite-FairEWS")
    st.markdown(
        '<div class="hero-copy">A lightweight educator-facing prototype for reviewing student risk signals, model outputs, and suggested support actions.</div>',
        unsafe_allow_html=True,
    )
    st.warning(
        "This dashboard is a decision-support prototype, not an automated disciplinary or diagnostic system. "
        "Predictions should be reviewed alongside educator judgment, policy, and student context."
    )

    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run `python train.py` before launching the dashboard.")
        st.stop()

    model_bundle, processed_students, scored_students = load_dashboard_assets()

    with st.sidebar:
        st.header("Student Row")
        st.caption("Select one anonymized sample row from the processed student dataset.")
        selected_student_id = st.selectbox(
            "Choose a student record",
            processed_students["student_id"].tolist(),
            format_func=lambda student_id: f"Student {int(student_id):04d}",
        )
        st.divider()
        st.caption(f"Loaded model: {model_bundle['model_name']}")
        st.caption(f"Sample rows available: {len(processed_students):,}")

    selected_student_row = processed_students.loc[
        processed_students["student_id"] == selected_student_id
    ].copy()
    selected_student_score = scored_students.loc[
        scored_students["student_id"] == selected_student_id
    ].iloc[0]

    explanation = explain_student_prediction(
        model_bundle=model_bundle,
        student_features=selected_student_row,
        top_n=TOP_FEATURE_COUNT,
    )

    predicted_outcome = get_predicted_outcome(selected_student_score["predicted_dropout_flag"])
    predicted_probability = float(selected_student_score["predicted_dropout_probability"])
    risk_level = str(selected_student_score["risk_category"])
    support_action = str(selected_student_score["support_action"])

    summary_column, action_column = st.columns([2.2, 1.2])

    with summary_column:
        with st.container(border=True):
            st.subheader(f"Prediction Summary: Student {int(selected_student_id):04d}")
            metric_one, metric_two, metric_three = st.columns(3)
            metric_one.metric("Predicted Outcome", predicted_outcome)
            metric_two.metric("Risk Level", risk_level)
            metric_three.metric("Predicted Risk Probability", f"{predicted_probability:.1%}")

            st.markdown(
                '<div class="section-note">The risk probability reflects the model estimate for dropout risk in this binary early-support classification setup.</div>',
                unsafe_allow_html=True,
            )

    with action_column:
        with st.container(border=True):
            st.subheader("Suggested Support Action")
            render_support_action(risk_level, support_action)

    with st.container(border=True):
        st.subheader("Top 3 Important Features")
        st.caption(
            "These are local coefficient-based signals for the selected prediction. "
            "They indicate association with the model score, not causation."
        )
        render_feature_cards(explanation)

    detail_column, preview_column = st.columns([1.4, 1.1])

    with detail_column:
        with st.container(border=True):
            st.subheader("Interpretation Notes")
            st.write(
                "A higher risk probability means the model sees a stronger pattern associated with dropout in the training data. "
                "The risk level groups that probability into an educator-friendly band for quick review."
            )
            st.write(
                "The top features help explain why this single row received its score. "
                "Positive directions raise predicted dropout risk, while negative directions lower it."
            )

    with preview_column:
        with st.container(border=True):
            st.subheader("Selected Student Row")
            st.caption("Compact preview from the processed sample feature set used by the dashboard.")
            st.dataframe(
                build_student_row_preview(selected_student_row),
                hide_index=True,
                width="stretch",
            )


if __name__ == "__main__":
    # VS Code often starts Python files directly. Relaunch through Streamlit so the app opens in a browser.
    if not running_under_streamlit() and os.environ.get("EDURISKLITE_SKIP_STREAMLIT_RELAUNCH") != "1":
        relaunch_with_streamlit()
    main()
