# Responsible Use

## Purpose

EduRiskLite-FairEWS is a prototype for early-support analytics. Its goal is to help educators identify patterns that may justify outreach or additional support. It is not a final production system and should not be used as the sole basis for high-stakes decisions.

## Appropriate use

- Use the model as a decision-support aid, not as an automated decision-maker.
- Combine model outputs with educator judgment, student context, and current institutional policy.
- Treat `High` risk scores as a prompt for supportive outreach, not as proof that a student will disengage or drop out.

## Inappropriate use

- Do not use this prototype to deny services, remove opportunities, or impose punitive action.
- Do not interpret model outputs as causal explanations.
- Do not assume the fairness report proves the system is bias-free.

## Data and modeling limitations

- The training data comes from a public UCI dataset and may not reflect the local population where the tool is deployed.
- Several predictors are coded variables and may not capture the full context of student experience.
- The model is a simple baseline logistic regression. Its explanations are approximate coefficient-based contributions, not definitive causal drivers.
- Small subgroup sizes can make fairness comparisons unstable.

## Fairness guidance

- Review `outputs/fairness_report.csv` before sharing or acting on results.
- Pay attention to subgroup differences in recall, false positive rate, and false negative rate.
- Reassess the model if subgroup gaps are large or if the local deployment context differs from the original training data.

## Privacy guidance

- This repository should only be used with anonymized or appropriately governed student data.
- Do not add private student information to the repository without proper approval, security controls, and data governance.
- The included sample dataset is public and anonymized.

## Operational guidance

- Retrain and re-evaluate the model whenever the data source changes.
- Keep a human in the loop for all interventions.
- Document any policy or threshold changes made after initial deployment.
