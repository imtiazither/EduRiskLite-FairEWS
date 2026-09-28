"""Model training, scoring, and lightweight explanation helpers."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    RISK_ACTIONS,
    TEST_SIZE,
)
from src.data_utils import format_feature_name


def build_baseline_pipeline(feature_frame: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    """Create a simple preprocessing + logistic regression pipeline."""
    numeric_features = [feature_name for feature_name in NUMERIC_FEATURES if feature_name in feature_frame.columns]
    categorical_features = [feature_name for feature_name in feature_frame.columns if feature_name not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    return pipeline, numeric_features, categorical_features


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | int]:
    """Compute the core metrics used in both model and fairness reporting."""
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_prob_array = np.asarray(y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_array, labels=[0, 1]).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true_array, y_prob_array))
    except ValueError:
        roc_auc = np.nan

    return {
        "sample_count": int(len(y_true_array)),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred_array, zero_division=0)),
        "roc_auc": roc_auc,
        "observed_dropout_rate": float(np.mean(y_true_array)),
        "predicted_dropout_rate": float(np.mean(y_pred_array)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
    }


def train_baseline_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, object], dict[str, object]]:
    """Train the baseline classifier and return the fitted bundle plus evaluation artifacts."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    pipeline, numeric_features, categorical_features = build_baseline_pipeline(features)
    pipeline.fit(x_train, y_train)

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
    evaluation_metrics = compute_classification_metrics(y_test, y_pred, y_prob)

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": list(features.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "medium_risk_threshold": MEDIUM_RISK_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "risk_actions": RISK_ACTIONS,
        "model_name": "LogisticRegression",
    }

    evaluation_artifacts = {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": evaluation_metrics,
    }

    return model_bundle, evaluation_artifacts


def save_model_bundle(model_bundle: dict[str, object], model_path=MODEL_PATH) -> None:
    """Persist the trained model bundle to disk."""
    joblib.dump(model_bundle, model_path)


def load_model_bundle(model_path=MODEL_PATH) -> dict[str, object]:
    """Load a previously trained model bundle."""
    return joblib.load(model_path)


def map_risk_level(probability: float, model_bundle: dict[str, object]) -> str:
    """Map a predicted dropout probability into a simple risk band."""
    if probability >= float(model_bundle["high_risk_threshold"]):
        return "High"
    if probability >= float(model_bundle["medium_risk_threshold"]):
        return "Medium"
    return "Low"


def score_students(model_bundle: dict[str, object], dataset: pd.DataFrame) -> pd.DataFrame:
    """Score every student row and attach risk labels and suggested actions."""
    feature_columns = model_bundle["feature_columns"]
    pipeline: Pipeline = model_bundle["pipeline"]

    scoring_frame = dataset[feature_columns].copy()
    probabilities = pipeline.predict_proba(scoring_frame)[:, 1]
    predicted_flags = (probabilities >= float(model_bundle["prediction_threshold"])).astype(int)
    risk_categories = [map_risk_level(probability, model_bundle) for probability in probabilities]

    scored_students = pd.DataFrame(
        {
            "student_id": dataset["student_id"],
            "predicted_dropout_probability": probabilities,
            "predicted_dropout_flag": predicted_flags,
            "risk_category": risk_categories,
            "support_action": [model_bundle["risk_actions"][risk] for risk in risk_categories],
        }
    )

    if "target" in dataset.columns:
        scored_students["observed_outcome"] = dataset["target"]

    return scored_students


def format_encoded_feature_name(encoded_feature_name: str, categorical_features: list[str]) -> str:
    """Convert transformed feature names into readable explanation labels."""
    if encoded_feature_name.startswith("num__"):
        return format_feature_name(encoded_feature_name.replace("num__", ""))

    cleaned_name = encoded_feature_name.replace("cat__", "")
    for raw_feature_name in sorted(categorical_features, key=len, reverse=True):
        prefix = f"{raw_feature_name}_"
        if cleaned_name.startswith(prefix):
            group_value = cleaned_name[len(prefix) :]
            return f"{format_feature_name(raw_feature_name)} = {group_value}"

    return format_feature_name(cleaned_name)


def explain_student_prediction(
    model_bundle: dict[str, object],
    student_features: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return the strongest positive and negative feature contributions for one student."""
    feature_columns = model_bundle["feature_columns"]
    categorical_features = model_bundle["categorical_features"]
    pipeline: Pipeline = model_bundle["pipeline"]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]

    transformed_row = preprocessor.transform(student_features[feature_columns])
    if hasattr(transformed_row, "toarray"):
        transformed_values = transformed_row.toarray().ravel()
    else:
        transformed_values = np.asarray(transformed_row).ravel()

    transformed_feature_names = preprocessor.get_feature_names_out()
    contributions = transformed_values * classifier.coef_[0]

    explanation = pd.DataFrame(
        {
            "encoded_feature": transformed_feature_names,
            "contribution": contributions,
        }
    )
    explanation["abs_contribution"] = explanation["contribution"].abs()
    explanation = explanation[explanation["abs_contribution"] > 0]
    explanation = explanation.sort_values("abs_contribution", ascending=False).head(top_n).copy()
    explanation["factor"] = explanation["encoded_feature"].apply(
        lambda feature_name: format_encoded_feature_name(feature_name, categorical_features)
    )
    explanation["direction"] = np.where(
        explanation["contribution"] >= 0,
        "Higher dropout risk",
        "Lower dropout risk",
    )
    

    return explanation[["factor", "direction", "contribution", "abs_contribution"]]
