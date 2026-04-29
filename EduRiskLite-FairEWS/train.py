"""Train the baseline dropout-risk model and write project outputs."""

from __future__ import annotations

import pandas as pd

from src.config import (
    FAIRNESS_CHART_PATH,
    FAIRNESS_REPORT_PATH,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    MODEL_PATH,
    MODEL_REPORT_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    POSITIVE_CLASS,
    PREDICTION_THRESHOLD,
    SCORED_STUDENTS_PATH,
    SUBGROUP_COLUMNS,
)
from src.data_utils import load_dataset, split_features_target
from src.fairness import create_false_negative_rate_chart, evaluate_fairness
from src.modeling import save_model_bundle, score_students, train_baseline_model


def main() -> None:
    """Train the project model and create the required CSV outputs."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the public UCI dataset and create the binary dropout target.
    dataset = load_dataset()
    features, target, feature_columns = split_features_target(dataset)

    model_bundle, evaluation_artifacts = train_baseline_model(features, target)
    model_bundle["feature_columns"] = feature_columns
    save_model_bundle(model_bundle, MODEL_PATH)

    # Score the full dataset so the Streamlit app can display student views immediately.
    scored_students = score_students(model_bundle, dataset)
    scored_students.to_csv(SCORED_STUDENTS_PATH, index=False)

    model_report = pd.DataFrame(
        [
            {
                "model_name": model_bundle["model_name"],
                "task": "Binary dropout risk classification",
                "positive_class": POSITIVE_CLASS,
                "train_rows": len(evaluation_artifacts["X_train"]),
                "test_rows": len(evaluation_artifacts["X_test"]),
                "decision_threshold": PREDICTION_THRESHOLD,
                "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
                "high_risk_threshold": HIGH_RISK_THRESHOLD,
                **evaluation_artifacts["metrics"],
            }
        ]
    )
    model_report.to_csv(MODEL_REPORT_PATH, index=False)

    available_subgroups = [
        subgroup_name
        for subgroup_name in SUBGROUP_COLUMNS
        if subgroup_name in evaluation_artifacts["X_test"].columns
    ]
    fairness_report = evaluate_fairness(
        y_true=evaluation_artifacts["y_test"],
        y_pred=evaluation_artifacts["y_pred"],
        y_prob=evaluation_artifacts["y_prob"],
        subgroup_frame=evaluation_artifacts["X_test"][available_subgroups].copy(),
        subgroup_columns=available_subgroups,
    )
    fairness_report.to_csv(FAIRNESS_REPORT_PATH, index=False)
    create_false_negative_rate_chart(fairness_report, FAIRNESS_CHART_PATH)

    print(f"Saved trained model to: {MODEL_PATH}")
    print(f"Saved model report to: {MODEL_REPORT_PATH}")
    print(f"Saved fairness report to: {FAIRNESS_REPORT_PATH}")
    print(f"Saved fairness chart to: {FAIRNESS_CHART_PATH}")
    print(f"Saved scored student records to: {SCORED_STUDENTS_PATH}")


if __name__ == "__main__":
    main()
