"""Project-wide configuration values."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH_CANDIDATES = [
    PROJECT_ROOT / "data.csv",
    PROJECT_ROOT / "data" / "data.csv",
]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = MODELS_DIR / "edurisklite_fairews_model.joblib"
MODEL_REPORT_PATH = OUTPUTS_DIR / "model_report.csv"
FAIRNESS_REPORT_PATH = OUTPUTS_DIR / "fairness_report.csv"
FAIRNESS_CHART_PATH = OUTPUTS_DIR / "fairness_false_negative_rate.png"
SCORED_STUDENTS_PATH = OUTPUTS_DIR / "scored_students.csv"

TARGET_COLUMN = "target"
POSITIVE_CLASS = "Dropout"
RANDOM_STATE = 42
TEST_SIZE = 0.20
PREDICTION_THRESHOLD = 0.50
MEDIUM_RISK_THRESHOLD = 0.33
HIGH_RISK_THRESHOLD = 0.66
TOP_FACTOR_COUNT = 5

RISK_ACTIONS = {
    "Low": "Continue routine monitoring and reinforce the student's current supports.",
    "Medium": "Schedule a check-in within two weeks and review attendance, coursework, and advising needs.",
    "High": "Start proactive outreach within 72 hours and connect the student to academic, advising, and financial support.",
}

# These variables are treated as numeric magnitudes rather than code-based categories.
NUMERIC_FEATURES = [
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
]

# Available subgroup variables used by the fairness module.
SUBGROUP_COLUMNS = [
    "gender",
    "scholarship_holder",
    "debtor",
    "tuition_fees_up_to_date",
    "displaced",
    "educational_special_needs",
    "international",
]
