"""Subgroup fairness reporting for the dropout-risk classifier."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SUBGROUP_COLUMNS
from src.modeling import compute_classification_metrics


def evaluate_fairness(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subgroup_frame: pd.DataFrame,
    subgroup_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compare model performance across available subgroup variables."""
    selected_subgroups = subgroup_columns or SUBGROUP_COLUMNS
    overall_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    fairness_rows: list[dict[str, object]] = [
        {
            "subgroup": "overall",
            "group_value": "all",
            "small_sample_warning": False,
            "accuracy_gap_vs_overall": 0.0,
            "recall_gap_vs_overall": 0.0,
            "precision_gap_vs_overall": 0.0,
            "false_positive_rate_gap_vs_overall": 0.0,
            "false_negative_rate_gap_vs_overall": 0.0,
            "f1_score": float(overall_metrics["f1"]),
            **overall_metrics,
        }
    ]

    for subgroup_name in selected_subgroups:
        if subgroup_name not in subgroup_frame.columns:
            continue

        subgroup_series = subgroup_frame[subgroup_name].fillna("Missing")
        for group_value in sorted(subgroup_series.unique(), key=lambda item: str(item)):
            group_mask = subgroup_series == group_value
            group_metrics = compute_classification_metrics(
                y_true[group_mask],
                y_pred[group_mask],
                y_prob[group_mask],
            )
            fairness_rows.append(
                {
                    "subgroup": subgroup_name,
                    "group_value": str(group_value),
                    "small_sample_warning": int(group_metrics["sample_count"]) < 50,
                    "accuracy_gap_vs_overall": float(group_metrics["accuracy"] - overall_metrics["accuracy"]),
                    "recall_gap_vs_overall": float(group_metrics["recall"] - overall_metrics["recall"]),
                    "precision_gap_vs_overall": float(group_metrics["precision"] - overall_metrics["precision"]),
                    "false_positive_rate_gap_vs_overall": float(
                        group_metrics["false_positive_rate"] - overall_metrics["false_positive_rate"]
                    )
                    if not np.isnan(group_metrics["false_positive_rate"])
                    and not np.isnan(overall_metrics["false_positive_rate"])
                    else np.nan,
                    "false_negative_rate_gap_vs_overall": float(
                        group_metrics["false_negative_rate"] - overall_metrics["false_negative_rate"]
                    )
                    if not np.isnan(group_metrics["false_negative_rate"])
                    and not np.isnan(overall_metrics["false_negative_rate"])
                    else np.nan,
                    "f1_score": float(group_metrics["f1"]),
                    **group_metrics,
                }
            )

    return pd.DataFrame(fairness_rows)


def create_false_negative_rate_chart(
    fairness_report: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Save a simple bar chart comparing false negative rates across subgroup rows."""
    output_path = Path(output_path)
    chart_frame = fairness_report.loc[
        fairness_report["subgroup"] != "overall",
        ["subgroup", "group_value", "false_negative_rate", "small_sample_warning"],
    ].copy()
    chart_frame = chart_frame.dropna(subset=["false_negative_rate"])
    chart_frame["group_label"] = chart_frame.apply(
        lambda row: f"{row['subgroup']}={row['group_value']}",
        axis=1,
    )
    chart_frame = chart_frame.sort_values("false_negative_rate", ascending=False)

    figure_height = max(4, 0.45 * len(chart_frame))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    bar_colors = [
        "#d64545" if not small_sample_warning else "#9aa5b1"
        for small_sample_warning in chart_frame["small_sample_warning"]
    ]
    axis.barh(chart_frame["group_label"], chart_frame["false_negative_rate"], color=bar_colors)
    axis.invert_yaxis()
    axis.set_title("False Negative Rate by Subgroup")
    axis.set_xlabel("False Negative Rate")
    axis.set_ylabel("Subgroup = Group Value")
    axis.set_xlim(0, min(1.0, max(chart_frame["false_negative_rate"].max() * 1.15, 0.1)))

    for index, value in enumerate(chart_frame["false_negative_rate"]):
        axis.text(
            value + 0.01,
            index,
            f"{value:.1%}",
            va="center",
            fontsize=9,
            color="#243b53",
        )

    axis.grid(axis="x", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path
