import pandas as pd
import numpy as np
import glob
import os

FANTASY_SCORING = {
    'run': 1, 'four_bonus': 1, 'six_bonus': 2,
    'half_century_bonus': 8, 'century_bonus': 16,
    'thirty_bonus': 4, 'duck_penalty': -2,
    'wicket': 25, 'lbw_bowled_bonus': 8,
    'three_wicket_bonus': 4, 'four_wicket_bonus': 8,
    'five_wicket_bonus': 16, 'maiden_over': 4,
    'catch': 8, 'three_catch_bonus': 4,
    'stumping': 12, 'run_out_direct': 12, 'run_out_indirect': 6,
}


def load_cricsheet_data(data_folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_folder}.\n"
            "Download IPL data from https://cricsheet.org/downloads/ "
            "(choose IPL → CSV format) and extract into the data/ folder."
        )
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df['match_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True)


def compute_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (match_file, innings), group in df.groupby(['match_file', 'innings']):
        match_id = match_file.replace('.csv', '')
        date = group['start_date'].iloc[0] if 'start_date' in group.columns else None
        batting_team = group['batting_team'].iloc[0]
        bowling_team = group['bowling_team'].iloc[0]

        # Batting
        for batter, bat_df in group.groupby('striker'):
            runs = bat_df['runs_off_bat'].sum()
            balls = len(bat_df)
            fours = (bat_df['runs_off_bat'] == 4).sum()
            sixes = (bat_df['runs_off_bat'] == 6).sum()
            dismissed = bat_df['wicket_type'].notna().any()
            pts = runs + fours + sixes * 2
            if runs >= 100: pts += 16
            elif runs >= 50: pts += 8
            elif runs >= 30: pts += 4
            if runs == 0 and dismissed: pts -= 2
            records.append({
                'match_id': match_id, 'date': date, 'player': batter,
                'team': batting_team, 'opponent': bowling_team, 'innings': innings,
                'runs': runs, 'balls_faced': balls, 'fours': fours, 'sixes': sixes,
                'strike_rate': round(runs / balls * 100, 2) if balls > 0 else 0,
                'dismissed': int(dismissed), 'wickets': 0, 'economy': 0,
                'catches': 0, 'batting_pts': pts, 'bowling_pts': 0,
                'fielding_pts': 0, 'total_fantasy_pts': pts,
            })

        # Bowling
        for bowler, bowl_df in group.groupby('bowler'):
            wickets = bowl_df['wicket_type'].notna().sum()
            wickets -= (bowl_df['wicket_type'] == 'run out').sum()
            wickets = max(int(wickets), 0)
            runs_c = bowl_df['runs_off_bat'].sum()
            legal = len(bowl_df[bowl_df['wides'].isna()])
            overs = legal / 6
            lbw_b = bowl_df['wicket_type'].isin(['lbw', 'bowled']).sum()
            pts = wickets * 25 + lbw_b * 8
            if wickets >= 5: pts += 16
            elif wickets >= 4: pts += 8
            elif wickets >= 3: pts += 4
            records.append({
                'match_id': match_id, 'date': date, 'player': bowler,
                'team': bowling_team, 'opponent': batting_team, 'innings': innings,
                'runs': 0, 'balls_faced': 0, 'fours': 0, 'sixes': 0,
                'strike_rate': 0, 'dismissed': 0, 'wickets': wickets,
                'economy': round(runs_c / overs, 2) if overs > 0 else 0,
                'catches': 0, 'batting_pts': 0, 'bowling_pts': pts,
                'fielding_pts': 0, 'total_fantasy_pts': pts,
            })

        # Fielding
        if 'fielder' in group.columns:
            for fielder, f_df in group[group['fielder'].notna()].groupby('fielder'):
                catches = (f_df['wicket_type'] == 'caught').sum()
                stumpings = (f_df['wicket_type'] == 'stumped').sum()
                run_outs = (f_df['wicket_type'] == 'run out').sum()
                pts = catches * 8 + stumpings * 12 + run_outs * 12
                if catches >= 3: pts += 4
                records.append({
                    'match_id': match_id, 'date': date, 'player': fielder,
                    'team': bowling_team, 'opponent': batting_team, 'innings': innings,
                    'runs': 0, 'balls_faced': 0, 'fours': 0, 'sixes': 0,
                    'strike_rate': 0, 'dismissed': 0, 'wickets': 0, 'economy': 0,
                    'catches': int(catches), 'batting_pts': 0, 'bowling_pts': 0,
                    'fielding_pts': pts, 'total_fantasy_pts': pts,
                })

    stats = pd.DataFrame(records).fillna(0)
    stats['total_fantasy_pts'] = stats['batting_pts'] + stats['bowling_pts'] + stats['fielding_pts']
    return stats


def engineer_features(stats: pd.DataFrame, n_recent: int = 5) -> pd.DataFrame:
    stats = stats.sort_values('date').reset_index(drop=True)
    feature_rows = []
    for player, player_df in stats.groupby('player'):
        player_df = player_df.sort_values('date').reset_index(drop=True)
        for i in range(n_recent, len(player_df)):
            cur = player_df.iloc[i]
            rec = player_df.iloc[i - n_recent:i]
            feature_rows.append({
                'player': player,
                'match_id': cur['match_id'],
                'date': cur['date'],
                'team': cur['team'],
                'opponent': cur['opponent'],
                'actual_fantasy_pts': cur['total_fantasy_pts'],
                'avg_fantasy_pts_last5': rec['total_fantasy_pts'].mean(),
                'avg_runs_last5': rec['runs'].mean(),
                'avg_strike_rate_last5': rec['strike_rate'].mean(),
                'avg_fours_last5': rec['fours'].mean(),
                'avg_sixes_last5': rec['sixes'].mean(),
                'avg_wickets_last5': rec['wickets'].mean(),
                'avg_economy_last5': rec['economy'].mean(),
                'avg_catches_last5': rec['catches'].mean(),
                'recent_form_trend': (
                    rec['total_fantasy_pts'].iloc[-3:].mean() -
                    rec['total_fantasy_pts'].iloc[:3].mean()
                ) if len(rec) >= 6 else 0,
                'pts_std_last5': rec['total_fantasy_pts'].std(),
            })
    return pd.DataFrame(feature_rows).fillna(0)
