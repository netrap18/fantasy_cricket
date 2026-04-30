import pandas as pd
import numpy as np

IPL_TEAMS = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'
]

PLAYERS = {
    'Mumbai Indians': ['Rohit Sharma', 'Ishan Kishan', 'Suryakumar Yadav',
                       'Hardik Pandya', 'Tim David', 'Jasprit Bumrah',
                       'Tilak Varma', 'Piyush Chawla', 'Kieron Pollard',
                       'Hrithik Shokeen', 'Jofra Archer'],
    'Chennai Super Kings': ['MS Dhoni', 'Ruturaj Gaikwad', 'Devon Conway',
                            'Moeen Ali', 'Ravindra Jadeja', 'Deepak Chahar',
                            'Shivam Dube', 'Ben Stokes', 'Matheesha Pathirana',
                            'Tushar Deshpande', 'Ambati Rayudu'],
    'Royal Challengers Bangalore': ['Virat Kohli', 'Faf du Plessis', 'Glenn Maxwell',
                                    'Dinesh Karthik', 'Mohammed Siraj', 'Wanindu Hasaranga',
                                    'Josh Hazlewood', 'Shahbaz Ahmed', 'Rajat Patidar',
                                    'Mahipal Lomror', 'Reece Topley'],
    'Kolkata Knight Riders': ['Shreyas Iyer', 'Nitish Rana', 'Andre Russell',
                              'Sunil Narine', 'Rinku Singh', 'Varun Chakaravarthy',
                              'Shardul Thakur', 'Tim Southee', 'Venkatesh Iyer',
                              'Jason Roy', 'Kuldeep Nair'],
    'Delhi Capitals': ['David Warner', 'Prithvi Shaw', 'Rishabh Pant',
                       'Axar Patel', 'Anrich Nortje', 'Kuldeep Yadav',
                       'Mitchell Marsh', 'Rovman Powell', 'Lalit Yadav',
                       'Phil Salt', 'Mukesh Kumar'],
    'Sunrisers Hyderabad': ['Abhishek Sharma', 'Heinrich Klaasen', 'Aiden Markram',
                            'Bhuvneshwar Kumar', 'T Natarajan', 'Washington Sundar',
                            'Harry Brook', 'Mayank Agarwal', 'Marco Jansen',
                            'Umran Malik', 'Kane Williamson'],
    'Rajasthan Royals': ['Sanju Samson', 'Jos Buttler', 'Yashasvi Jaiswal',
                         'Shimron Hetmyer', 'Trent Boult', 'Yuzvendra Chahal',
                         'Ravichandran Ashwin', 'Dhruv Jurel', 'Devdutt Padikkal',
                         'Jason Holder', 'Sandeep Sharma'],
    'Punjab Kings': ['Shikhar Dhawan', 'Jonny Bairstow', 'Liam Livingstone',
                     'Sam Curran', 'Arshdeep Singh', 'Kagiso Rabada',
                     'Jitesh Sharma', 'Harpreet Brar', 'Nathan Ellis',
                     'Rishi Dhawan', 'Bhanuka Rajapaksa'],
    'Gujarat Titans': ['Shubman Gill', 'Wriddhiman Saha', 'David Miller',
                       'Mohammad Shami', 'Rashid Khan', 'Lockie Ferguson',
                       'Vijay Shankar', 'Abhinav Manohar', 'Rahul Tewatia',
                       'Noor Ahmad', 'Hardik Pandya'],
    'Lucknow Super Giants': ['KL Rahul', 'Quinton de Kock', 'Marcus Stoinis',
                             'Deepak Hooda', 'Avesh Khan', 'Ravi Bishnoi',
                             'Krunal Pandya', 'Kyle Mayers', 'Ayush Badoni',
                             'Prerak Mankad', 'Mohsin Khan'],
}

RATINGS = {
    'Rohit Sharma': 85, 'Virat Kohli': 92, 'MS Dhoni': 80, 'Jos Buttler': 88,
    'Suryakumar Yadav': 90, 'Rashid Khan': 87, 'Jasprit Bumrah': 89,
    'Glenn Maxwell': 83, 'Andre Russell': 85, 'Yuzvendra Chahal': 82,
    'Ruturaj Gaikwad': 84, 'Shubman Gill': 86, 'Yashasvi Jaiswal': 85,
    'KL Rahul': 83, 'Heinrich Klaasen': 84, 'David Warner': 82,
}


def _sim_match(player, seed_offset=0):
    seed = (abs(hash(player)) + seed_offset) % (2**31)
    rng = np.random.default_rng(seed)
    skill = RATINGS.get(player, 70) / 100.0
    runs = int(rng.negative_binomial(skill * 4, 0.18))
    runs = min(runs, 120)
    balls = max(runs + int(rng.normal(10, 8)), 1)
    fours = int(runs * rng.uniform(0.05, 0.15))
    sixes = int(runs * rng.uniform(0.02, 0.10))
    dismissed = rng.random() > 0.3
    wickets = int(rng.negative_binomial(skill * 2, 0.6)) if rng.random() < 0.5 else 0
    wickets = min(wickets, 5)
    economy = float(max(5, rng.normal(8 - skill * 2, 1.5)))
    catches = int(rng.random() < 0.2)
    pts = runs + fours + sixes * 2
    if runs >= 100: pts += 16
    elif runs >= 50: pts += 8
    elif runs >= 30: pts += 4
    if runs == 0 and dismissed: pts -= 2
    pts += wickets * 25 + catches * 8
    return {
        'runs': runs, 'balls_faced': balls, 'fours': fours, 'sixes': sixes,
        'dismissed': int(dismissed), 'wickets': wickets,
        'economy': round(economy, 2), 'catches': catches,
        'strike_rate': round(runs / balls * 100, 1) if balls > 0 else 0,
        'batting_pts': runs + fours + sixes * 2,
        'bowling_pts': wickets * 25,
        'fielding_pts': catches * 8,
        'total_fantasy_pts': pts,
    }


def generate_sample_dataset(n_matches: int = 300) -> pd.DataFrame:
    records = []
    dates = pd.date_range('2019-03-01', periods=n_matches, freq='3D')
    teams = IPL_TEAMS.copy()
    for i, date in enumerate(dates):
        np.random.shuffle(teams)
        t1, t2 = teams[0], teams[1]
        mid = f"match_{i:04d}"
        for team, opp in [(t1, t2), (t2, t1)]:
            for player in PLAYERS.get(team, []):
                s = _sim_match(player, seed_offset=i)
                records.append({
                    'match_id': mid,
                    'date': date.strftime('%Y-%m-%d'),
                    'player': player, 'team': team, 'opponent': opp,
                    **s,
                })
    return pd.DataFrame(records)
