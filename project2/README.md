# Project 2 — 2025–26 NBA Team Wins Prediction

Project 2 predicts team win totals by modeling the gap between Vegas over/under expectations and actual performance using team, injury, and roster continuity features.

## Run

```bash
cd project2
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib seaborn matplotlib
python run.py
```

## Output files

- `saved_models/xgb_model.pkl`
- `results/feature_correlation_heatmap.png`
- `results/wins_feature_correlation.png`
- `results/validate_2024_25.png`

