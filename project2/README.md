# Project 2 — Player Growth & Injury-Adjusted NBA Wins Prediction

This project forecasts each team's upcoming win total. Vegas over/under is the starting point; the model learns the expected miss from that market using prior team performance, continuity, injuries, and a player-level projection layer.

The player layer is designed to capture the kind of jump seen in Oklahoma City: a young, high-minute player with a positive multi-year impact trend is not treated the same as an older player with the same current value.

## New pipeline pieces

- `model/player_impact_model.py` trains the individual player-impact and availability model.
- `data/player_impact_projections.csv` is generated when `run.py` executes and contains player-level projected impact rows for inspection.
- Team-level features produced by the player model include:
  - `Projected_Player_Impact`
  - `Projected_Player_Impact_Delta`
  - `Projected_Young_Upside`
  - `Projected_Aging_Drag`
  - `Projected_Rotation_Minutes`
  - `Projected_Rotation_Count`
  - `Projected_Healthy_Player_Impact`
  - `Projected_Rotation_Injury_Risk`
- The wins pipeline converts those projections into:
  - `Development_Upside`
  - `Age_Adjusted_Net`
  - `Injury_Regression_Upside`
  - `Injury_Risk_Adjusted_Net`
  - `Projected_Health_Adjusted_Impact`
  - `Rotation_Health_Risk`

## What changed in this version

- **Player growth momentum.** The player model uses one- and two-season impact lags, a two-year trend, and an acceleration term. Young-player trend is explicitly modeled, so a rising 21-year-old can receive more upside than an equally productive veteran.
- **Availability-adjusted rotations.** The project does not contain player-by-player injury histories. Instead, it transparently uses prior team injury burden as an exposure proxy, weights it by projected rotation role, and produces a separate healthy-impact and rotation-risk output. This avoids fabricating individual injury data while still penalizing fragile rotations.
- **Leakage-free team features.** Injury and continuity features now use *previous-season* net rating. Earlier code mixed a season's team rating into that season's target, which made backtests look better than a real forecast.
- **Vegas stays a baseline.** The model predicts `actual wins - Vegas_OU`, rather than trying to replace the market from a small historical sample.

## Data limits

This is a preseason team-wins model, not a game-by-game injury predictor. The injury adjustment is a team-context proxy because the included injury CSV has team-season totals only. To model a player's exact injury probability, add a player-season availability dataset with player name, season, games played, and games missed; the `PlayerKey` column in `player_impact_model.py` is the intended join key.

## Setup

From the repository root:

```powershell
cd project2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Run everything

```bash
python run.py
```

The first run trains both models and creates the output folders if needed. It may take a few minutes because the player model is retrained in rolling historical windows.

## Output files

- `data/player_impact_projections.csv` — individual player impact projections used by the team model.
- `saved_models/xgb_model.pkl` — trained XGBoost team wins model.
- `results/feature_correlation_heatmap.png` — strongest feature correlations.
- `results/wins_feature_correlation.png` — features most correlated with wins.
- `results/validate_2024_25.png` — 2024–25 backtest plot.

## Modeling notes

The printed table shows the projected 2025–26 wins after league-wide win-total normalization. Inspect `data/player_impact_projections.csv` to see whether a team's adjustment comes from player development, aging drag, or health risk.

