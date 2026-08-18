# Project 2 — Two-Stage NBA Wins Prediction

Project 2 now uses a two-stage modeling approach for 2025–26 NBA win totals:

1. A player-impact model projects each player's next-season impact from age, minutes, LEBRON WAR, BPM, VORP, recent growth, and age-curve indicators.
2. The team wins model aggregates those player projections into team-level upside/risk features and combines them with Vegas odds, injuries, roster continuity, and team-strength metrics.

The goal is to capture jumps like young OKC improving from a mid-tier win team to a contender because Shai Gilgeous-Alexander and the surrounding core improved individually, not just because the previous team-level stats were strong.

## New pipeline pieces

- `model/player_impact_model.py` trains the individual player-impact model.
- `data/player_impact_projections.csv` is generated when `run.py` executes and contains player-level projected impact rows for inspection.
- Team-level features produced by the player model include:
  - `Projected_Player_Impact`
  - `Projected_Player_Impact_Delta`
  - `Projected_Young_Upside`
  - `Projected_Aging_Drag`
  - `Projected_Rotation_Minutes`
  - `Projected_Rotation_Count`
- The wins pipeline converts those projections into:
  - `Development_Upside`
  - `Age_Adjusted_Net`
  - `Injury_Regression_Upside`
  - `Injury_Risk_Adjusted_Net`

## Setup

```bash
cd project2
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib seaborn matplotlib
```

## Run everything

```bash
python run.py
```

## Output files

- `data/player_impact_projections.csv` — individual player impact projections used by the team model.
- `saved_models/xgb_model.pkl` — trained XGBoost team wins model.
- `results/feature_correlation_heatmap.png` — strongest feature correlations.
- `results/wins_feature_correlation.png` — features most correlated with wins.
- `results/validate_2024_25.png` — 2024–25 backtest plot.

## Modeling notes

The team model still predicts `actual wins - Vegas_OU`, which keeps Vegas as the baseline and lets the machine learning model focus on where the market may be too low or too high. Player development and injury features are intentionally separated so you can inspect whether a team is being boosted by young-player growth, injury rebound, or both.
