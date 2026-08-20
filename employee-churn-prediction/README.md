# Employee Churn Prediction

**Notebook:** [`Employee Churn Predictive Analysis.ipynb`](./Employee%20Churn%20Predictive%20Analysis.ipynb)
**Tools:** Python · pandas · scikit-learn · XGBoost · seaborn · matplotlib

## Objective

Build a classification model that predicts whether an employee will leave the company, so HR
can proactively identify at-risk employees and understand the key drivers of attrition.
Dataset: 15,000 employee records (satisfaction score, last evaluation, project count, monthly
hours, tenure, department, salary band, and whether the employee left).

## Approach

1. **EDA & data cleaning** — checked data types, removed duplicate records, screened for
   outliers using the IQR method, and reviewed class balance on the target (`left`).
2. **Feature engineering** — one-hot encoded categorical fields (department, salary) with
   `pd.get_dummies`, and checked for multicollinearity using Variance Inflation Factor (VIF).
3. **Modeling** — compared Logistic Regression, Decision Tree, and Random Forest classifiers.
   The Decision Tree and Random Forest were tuned with `GridSearchCV` (5-fold CV) scored across
   accuracy, F1, precision, recall, and ROC-AUC, refitting on ROC-AUC.
4. **Best model** — a tuned Decision Tree (`max_depth=4, min_samples_leaf=5`) achieved a
   cross-validated **ROC-AUC of ~0.975**. On the held-out test set it correctly classified
   2,748 of 2,792 employees (2,312 correct stays, 436 correct departures; 9 false positives,
   35 false negatives).
5. **Feature importance** — `satisfaction_level` was by far the strongest predictor (58.7%
   Gini importance), followed by tenure (15.3%), last evaluation score (14.5%), and number of
   projects (10.3%).

## Skills Demonstrated

Classification modeling, hyperparameter tuning via `GridSearchCV`, multicollinearity/VIF
analysis, feature importance interpretation, and model evaluation with ROC-AUC and confusion
matrices.
