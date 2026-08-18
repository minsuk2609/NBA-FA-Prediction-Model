import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TEAM_ABBREVIATIONS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets', 'BKN': 'Brooklyn Nets',
    'CHO': 'Charlotte Hornets', 'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers', 'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NOH': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHO': 'Phoenix Suns', 'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

PLAYER_FEATURES = [
    'Age', 'Age2', 'MP', 'MinutesWeight', 'Impact', 'Impact_Lag1', 'Impact_Delta1',
    'Impact_Roll2', 'LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON', 'BPM', 'VORP',
    'WS/48', 'AgeCurve', 'YoungUpsideFlag', 'PrimeFlag', 'AgingFlag', 'Impact_x_AgeCurve',
]


def _normalize_player_name(value):
    return str(value).strip().lower().replace('.', '').replace("'", '')


def _age_curve(age):
    age = pd.to_numeric(age, errors='coerce')
    return np.select(
        [age <= 20, age <= 23, age <= 26, age <= 30, age <= 33, age > 33],
        [1.35, 1.20, 1.08, 1.00, 0.92, 0.82],
        default=1.00,
    )


def _next_season_label(season):
    start = int(str(season)[:4]) + 1
    return f'{start}-{str(start + 1)[-2:]}'


def _load_player_frame(data_dir):
    lebron = pd.read_csv(f'{data_dir}/nba_2014_2025_LEBRON.csv')
    advanced = pd.read_csv(f'{data_dir}/nba_advanced_stats_2015_2025.csv')

    lebron['PlayerKey'] = lebron['Player'].map(_normalize_player_name)
    advanced['PlayerKey'] = advanced['Player'].map(_normalize_player_name)
    advanced = advanced[['Season', 'PlayerKey', 'MP', 'WS/48', 'BPM', 'VORP']]

    players = lebron.merge(advanced, on=['Season', 'PlayerKey'], how='left')
    for col in ['Age', 'LEBRON WAR', 'LEBRON', 'O-LEBRON', 'D-LEBRON', 'MP', 'WS/48', 'BPM', 'VORP']:
        players[col] = pd.to_numeric(players[col], errors='coerce')

    players['Team'] = players['Team(s)'].map(TEAM_ABBREVIATIONS)
    players = players[players['Team'].notna()].copy()
    players['Season_Start'] = players['Season'].str[:4].astype(int)
    players = players.sort_values(['PlayerKey', 'Season_Start'])

    players['MinutesWeight'] = players['MP'].fillna(players['MP'].median()).clip(lower=250, upper=3000) / 2000.0
    players['Impact'] = (
        players['LEBRON WAR'].fillna(0)
        + (0.75 * players['VORP'].fillna(0))
        + (0.20 * players['BPM'].fillna(0))
    )
    players['Impact_Lag1'] = players.groupby('PlayerKey')['Impact'].shift(1)
    players['Impact_Delta1'] = (players['Impact'] - players['Impact_Lag1']).clip(-5, 5)
    players['Impact_Roll2'] = players.groupby('PlayerKey')['Impact'].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    players['Age2'] = players['Age'] ** 2
    players['AgeCurve'] = _age_curve(players['Age'])
    players['YoungUpsideFlag'] = (players['Age'] <= 24).astype(int)
    players['PrimeFlag'] = players['Age'].between(25, 30).astype(int)
    players['AgingFlag'] = (players['Age'] >= 32).astype(int)
    players['Impact_x_AgeCurve'] = players['Impact'] * players['AgeCurve']
    players['Next_Impact'] = players.groupby('PlayerKey')['Impact'].shift(-1)
    players['Projection_Season'] = players['Season'].map(_next_season_label)
    return players


def _training_rows(players, max_training_season=None):
    train = players[players['Next_Impact'].notna()].copy()
    if max_training_season is not None:
        train = train[train['Season_Start'] <= max_training_season]
    train = train[train['MP'].fillna(0) >= 250]
    return train


def _fit_player_models(train):
    x = train[PLAYER_FEATURES]
    y = train['Next_Impact']
    sample_weight = train['MinutesWeight'].clip(0.2, 1.5)

    growth_model = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        HistGradientBoostingRegressor(
            loss='absolute_error',
            learning_rate=0.06,
            max_leaf_nodes=12,
            min_samples_leaf=18,
            l2_regularization=0.25,
            random_state=42,
        ),
    )
    stability_model = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        ),
    )
    growth_model.fit(x, y, histgradientboostingregressor__sample_weight=sample_weight)
    stability_model.fit(x, y, randomforestregressor__sample_weight=sample_weight)
    return growth_model, stability_model


def _predict_player_impact(models, frame):
    growth_model, stability_model = models
    pred_growth = growth_model.predict(frame[PLAYER_FEATURES])
    pred_stability = stability_model.predict(frame[PLAYER_FEATURES])
    frame = frame.copy()
    frame['Projected_Impact'] = (0.65 * pred_growth) + (0.35 * pred_stability)
    frame['Projected_Impact'] = frame['Projected_Impact'].clip(-6, 14)
    frame['Projected_Impact_Delta'] = frame['Projected_Impact'] - frame['Impact'].fillna(0)
    frame['Projected_Young_Upside'] = np.where(frame['Age'] <= 24, frame['Projected_Impact_Delta'].clip(lower=0), 0)
    frame['Projected_Aging_Drag'] = np.where(frame['Age'] >= 32, (-frame['Projected_Impact_Delta']).clip(lower=0), 0)
    return frame


def _aggregate_team_projection(player_predictions):
    player_predictions = player_predictions.copy()
    player_predictions['RotationWeight'] = player_predictions['MinutesWeight'].clip(0.15, 1.4)
    player_predictions['Weighted_Projected_Impact'] = player_predictions['Projected_Impact'] * player_predictions['RotationWeight']
    player_predictions['Weighted_Impact_Delta'] = player_predictions['Projected_Impact_Delta'] * player_predictions['RotationWeight']
    player_predictions['Weighted_Young_Upside'] = player_predictions['Projected_Young_Upside'] * player_predictions['RotationWeight']
    player_predictions['Weighted_Aging_Drag'] = player_predictions['Projected_Aging_Drag'] * player_predictions['RotationWeight']

    top_players = (
        player_predictions.sort_values(['Projection_Season', 'Team', 'Weighted_Projected_Impact'], ascending=[True, True, False])
        .groupby(['Projection_Season', 'Team'])
        .head(10)
    )
    team = top_players.groupby(['Projection_Season', 'Team'], as_index=False).agg(
        Projected_Player_Impact=('Weighted_Projected_Impact', 'sum'),
        Projected_Player_Impact_Delta=('Weighted_Impact_Delta', 'sum'),
        Projected_Young_Upside=('Weighted_Young_Upside', 'sum'),
        Projected_Aging_Drag=('Weighted_Aging_Drag', 'sum'),
        Projected_Rotation_Minutes=('MP', 'sum'),
        Projected_Rotation_Count=('Player', 'count'),
    )
    return team.rename(columns={'Projection_Season': 'Season'})


def build_projected_team_impact(data_dir='data', save_player_predictions=False):
    """Train a player-impact model and aggregate its projections to team-season features."""
    players = _load_player_frame(data_dir)
    seasons = sorted(players['Season_Start'].dropna().unique())
    team_frames = []
    player_frames = []

    for target_start in seasons[2:] + [max(seasons) + 1]:
        train = _training_rows(players, max_training_season=target_start - 2)
        if len(train) < 100:
            continue
        models = _fit_player_models(train)

        if target_start == max(seasons) + 1:
            latest = players.sort_values('Season_Start').groupby('PlayerKey').tail(1).copy()
            roster = pd.read_csv(f'{data_dir}/nba_roster_2025_26_bgm.csv')
            roster['PlayerKey'] = roster['Player'].map(_normalize_player_name)
            frame = roster.merge(latest.drop(columns=['Team', 'Player']), on='PlayerKey', how='left')
            frame['Projection_Season'] = '2025-26'
            frame['Age'] = frame['Age'] + 1
            frame['Age2'] = frame['Age'] ** 2
            frame['AgeCurve'] = _age_curve(frame['Age'])
            frame['YoungUpsideFlag'] = (frame['Age'] <= 24).astype(int)
            frame['PrimeFlag'] = frame['Age'].between(25, 30).astype(int)
            frame['AgingFlag'] = (frame['Age'] >= 32).astype(int)
            frame['Impact_x_AgeCurve'] = frame['Impact'].fillna(0) * frame['AgeCurve']
            frame['MinutesWeight'] = frame['MinutesWeight'].fillna(0.35)
        else:
            frame = players[players['Season_Start'] == target_start - 1].copy()

        projected = _predict_player_impact(models, frame)
        team_frames.append(_aggregate_team_projection(projected))
        player_frames.append(projected)

    team_features = pd.concat(team_frames, ignore_index=True)
    if save_player_predictions and player_frames:
        players_out = pd.concat(player_frames, ignore_index=True)
        players_out.to_csv(f'{data_dir}/player_impact_projections.csv', index=False)
    return team_features
