import pandas as pd
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURE_COLS = [
    'avg_fantasy_pts_last5', 'avg_runs_last5', 'avg_strike_rate_last5',
    'avg_fours_last5', 'avg_sixes_last5', 'avg_wickets_last5',
    'avg_economy_last5', 'avg_catches_last5', 'recent_form_trend', 'pts_std_last5',
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fantasy_model.pkl')


def train_model(features_df: pd.DataFrame, model_type: str = 'xgboost') -> dict:
    df = features_df.dropna(subset=FEATURE_COLS + ['actual_fantasy_pts'])
    X = df[FEATURE_COLS]
    y = df['actual_fantasy_pts']
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if model_type == 'xgboost' and HAS_XGB:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=8,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 2)
    mae = round(float(mean_absolute_error(y_test, preds)), 2)

    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': FEATURE_COLS}, f)

    return {'model': model, 'rmse': rmse, 'mae': mae,
            'feature_importance': importance,
            'train_size': len(X_train), 'test_size': len(X_test),
            'model_type': model_type}


def predict_xi(features_df, team1, team2, model, top_n=11):
    teams_df = features_df[features_df['team'].isin([team1, team2])].copy()
    if teams_df.empty:
        return pd.DataFrame()
    latest = teams_df.sort_values('date').groupby('player').last().reset_index()
    X = latest[FEATURE_COLS].fillna(0)
    latest['predicted_pts'] = model.predict(X).clip(0).round(1)
    result = latest[['player', 'team', 'predicted_pts',
                      'avg_fantasy_pts_last5', 'avg_runs_last5',
                      'avg_wickets_last5', 'recent_form_trend']].copy()
    result.columns = ['Player', 'Team', 'Predicted Points',
                      'Avg Pts (Last 5)', 'Avg Runs (Last 5)',
                      'Avg Wickets (Last 5)', 'Form Trend']
    return result.sort_values('Predicted Points', ascending=False).reset_index(drop=True)


def get_player_profile(features_df, player_name):
    df = features_df[features_df['player'] == player_name].sort_values('date')
    if df.empty:
        return {}
    latest = df.iloc[-1]
    history = df.tail(10)
    return {
        'name': player_name,
        'team': latest['team'],
        'avg_pts_last5': round(float(latest['avg_fantasy_pts_last5']), 1),
        'form_trend': round(float(latest['recent_form_trend']), 1),
        'pts_history': history['actual_fantasy_pts'].tolist(),
        'dates': history['date'].tolist(),
    }
