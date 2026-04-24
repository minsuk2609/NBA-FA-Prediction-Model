# NBA Free Agency Prediction Model

This repository contains two separate Python projects focused on NBA analytics:

- **Project 1 (`project1`)** predicts 2026 point guard free-agent salaries using advanced player stats and historical adjusted salary data.
- **Project 2 (`project2`)** predicts 2025–26 team win totals by modeling the error between Vegas over/under lines and actual wins.

## Project 1: Free Agent Salary Prediction

Small description:  
Project 1 trains an XGBoost regression model on player-level advanced stats + salary history, then uses it to estimate 2026 salaries for point guard free agents.

### Run Project 1

```bash
cd project1
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib
python model.py
python salary_prediction.py
```

### Project 1 outputs

- `salary_model.pkl`
- `salary_transformer.pkl`
- `feature_names.pkl`
- `2026 PG Salary Predictions.csv`

---

## Project 2: Team Wins Prediction

Small description:  
Project 2 builds team-level features (including injuries, roster continuity, lagged stats, and Vegas lines) and predicts 2025–26 NBA win totals.

### Run Project 2

```bash
cd project2
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib seaborn matplotlib
python run.py
```

### Project 2 outputs

- `saved_models/xgb_model.pkl`
- `results/feature_correlation_heatmap.png`
- `results/wins_feature_correlation.png`
- `results/validate_2024_25.png`

