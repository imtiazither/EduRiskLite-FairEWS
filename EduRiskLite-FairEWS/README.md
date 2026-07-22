# EduRiskLite-FairEWS

EduRiskLite-FairEWS is a lightweight, educator-facing early-support analytics prototype built with Python, scikit-learn, and Streamlit. Using the UCI Predict Students' Dropout and Academic Success dataset, the project demonstrates an end-to-end workflow for educational risk modeling: public data ingestion, preprocessing, baseline classification, interpretable risk communication, subgroup fairness evaluation, and dashboard delivery for practitioner review. The goal is not to automate decisions about students, but to show how learning analytics and responsible AI methods can be combined in a transparent, reproducible portfolio project.

## Why This Project Matters

Student attrition is rarely the result of a single factor. Academic progression, financial strain, enrollment context, and institutional conditions can interact in ways that make timely support difficult without structured analytics. This project matters because it frames predictive modeling as an early-support tool rather than a surveillance or exclusion mechanism. It is designed to help educators identify students who may benefit from outreach, advising, or additional resources while keeping human judgment, fairness checks, and responsible-use constraints central to the workflow.

## Early-Support Analytics and Fairness-Aware Educational AI

EduRiskLite-FairEWS sits at the intersection of Educational Data Mining, learning analytics, and responsible AI. From an early-support analytics perspective, it focuses on actionable risk signals that can inform intervention planning. From a fairness-aware AI perspective, it goes beyond reporting a single accuracy score by comparing model behavior across subgroup variables available in the dataset. The project is intentionally simple and interpretable so that questions of transparency, usability, and equity remain visible alongside predictive performance.

## Dataset Description

The project uses the public **UCI Predict Students' Dropout and Academic Success** dataset, which contains anonymized student-level records and outcome labels. The data includes demographic, academic, and macroeconomic variables such as admission grades, curricular unit completion, scholarship status, tuition status, and labor-market context. The original target labels are `Dropout`, `Graduate`, and `Enrolled`. For this prototype, the task is reframed as a binary early-support classification problem:

- Positive class: `Dropout`
- Negative class: `Graduate` or `Enrolled`

This formulation supports a practical early-warning use case while preserving the public, non-private nature of the source data.

## Methodology

The modeling pipeline is intentionally lightweight, reproducible, and appropriate for a baseline educational analytics prototype.

1. **Data loading and cleaning**  
   The dataset is read from the included CSV file, headers are standardized into `snake_case`, and a synthetic `student_id` is added for dashboard browsing.

2. **Preprocessing**  
   Numeric features are median-imputed and standardized. Categorical features are imputed with the most frequent value and one-hot encoded.

3. **Baseline modeling**  
   A `LogisticRegression` classifier with balanced class weights is trained using a stratified train/test split. This keeps the model easy to interpret and fast to run while still providing a credible baseline for portfolio and research demonstration purposes.

4. **Risk-level mapping**  
   Predicted dropout probabilities are translated into educator-friendly categories:
   - `Low`
   - `Medium`
   - `High`

5. **Local explanation support**  
   For each selected student record, the dashboard surfaces the top model contributions derived from transformed feature values and model coefficients. These are presented as directional signals, not causal claims.

6. **Evaluation and reporting**  
   The training script writes a model performance report, a fairness report, a scored student output, and a serialized model artifact for reuse in the dashboard.

## Dashboard Features

The Streamlit application is designed for an educator-facing review experience. It allows a user to:

- browse anonymized student records from the public dataset
- view each student's predicted dropout probability
- see a corresponding risk category: `Low`, `Medium`, or `High`
- inspect the top contributing factors behind the score
- review a suggested support action tied to the risk level
- view a compact student snapshot with selected academic and support-relevant fields
- review a simple risk distribution summary across the scored dataset

## Fairness Evaluation Features

The fairness module evaluates holdout-set performance across subgroup variables available in the source data, including:

- `gender`
- `scholarship_holder`
- `debtor`
- `tuition_fees_up_to_date`
- `displaced`
- `educational_special_needs`
- `international`

The generated `fairness_report.csv` includes:

- sample counts by subgroup
- accuracy, precision, recall, F1, and ROC-AUC where applicable
- observed and predicted positive rates
- false positive rate and false negative rate
- gaps versus overall model performance
- warnings for small subgroup sample sizes

This does not make the system "fair by default," but it does provide a practical starting point for auditing subgroup disparities in an educational AI workflow.

## Repository Outputs

Running the training pipeline produces the following artifacts:

- `models/edurisklite_fairews_model.joblib`
- `outputs/model_report.csv`
- `outputs/fairness_report.csv`
- `outputs/scored_students.csv`

These files support reproducibility, dashboard loading, and downstream review.

## Responsible-Use Disclaimer

This project is a **decision-support prototype**, not a production intervention system. Its outputs should be used to inform supportive educator review, not to deny access, label students permanently, or automate high-stakes decisions. The model reflects patterns in a public historical dataset and may not generalize to other institutions, populations, or policy settings. Fairness metrics should be interpreted as diagnostic signals rather than guarantees of equity. Any real-world deployment would require institutional governance, human oversight, privacy review, and context-specific validation.

Additional guidance is documented in [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## How To Run Locally

Clone or open the repository, then run the following commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

What these steps do:

- `pip install -r requirements.txt` installs the required libraries
- `python train.py` trains the baseline model and generates the CSV outputs
- `streamlit run app.py` launches the educator-facing dashboard

## Example Screenshots

This section is intended for portfolio presentation once dashboard images are captured. Recommended screenshots to include are:

| Suggested file | What it should show |
| --- | --- |
| `dashboard-overview.png` | Risk distribution, summary metrics, and highest-risk records |
| `student-detail.png` | Student risk card, predicted probability, and top contributing factors |
| `fairness-panel.png` | Subgroup performance comparison table from the fairness section |

Suggested placement:

- Save screenshots in `docs/screenshots/`
- Reference them from this README after exporting images from the running dashboard

Example Markdown once screenshots are available: Copy and Paste these links to your browser.

```markdown
![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![Student Detail View](docs/screenshots/student-detail.png)
![Fairness Evaluation Panel](docs/screenshots/fairness-panel.png)
```

## Future Work

There are several meaningful ways to extend this prototype:

- compare multiple baseline models such as decision trees, random forests, or calibrated linear models
- add threshold analysis to examine intervention trade-offs between recall and false positives
- incorporate model calibration plots and probability reliability checks
- expand fairness analysis with additional parity measures and uncertainty summaries
- add downloadable case summaries for advising or student-support workflows
- replace the sample-record selector with a secure upload workflow for institution-approved data
- evaluate longitudinal or semester-specific prediction settings for earlier intervention timing

## Portfolio Relevance

For a portfolio in Educational Data Mining, learning analytics, and responsible AI, EduRiskLite-FairEWS demonstrates the ability to connect technical modeling with applied educational decision support. It shows competence in data preprocessing, interpretable machine learning, educator-centered dashboard design, subgroup fairness analysis, and research-aligned documentation. Just as importantly, it treats predictive performance, transparency, and responsible use as part of the same system rather than separate concerns.
