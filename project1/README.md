# Project 1 — 2026 PG Free Agent Salary Prediction

Project 1 trains a salary prediction model from advanced player stats and historical salary data, then generates projected 2026 salaries for point guard free agents.

## Run

```bash
cd project1
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib
python model.py
python salary_prediction.py
```

## Output files

- `salary_model.pkl`
- `salary_transformer.pkl`
- `feature_names.pkl`
- `2026 PG Salary Predictions.csv`

