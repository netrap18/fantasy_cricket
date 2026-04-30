# 🏏 IPL Fantasy XI Predictor

An ML-powered web app that predicts the best Dream11 fantasy cricket XI 
for an IPL match using historical player performance data.

## Live Demo
*[Add your Streamlit Cloud link here after deployment]*

## Features
- Predicts fantasy points for each player using rolling 5-match averages
- Recommends Captain and Vice Captain picks
- Player form tracker and comparison tool
- Feature importance analysis
- Supports real Cricsheet IPL data OR built-in demo mode

## Tech Stack
Python · Streamlit · XGBoost · Scikit-learn · Pandas · Matplotlib

## How to run locally

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ipl-fantasy-predictor.git
cd ipl-fantasy-predictor

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the app
streamlit run app.py

### 4. Use real data (optional)
Download IPL ball-by-ball CSVs from https://cricsheet.org/downloads/
Extract into the data/ folder, then switch to "Real data" mode in the sidebar.

## Model
- Algorithm: XGBoost (default), Random Forest, Gradient Boosting
- Features: rolling 5-match averages for runs, wickets, strike rate, economy, form trend
- Train/test split: time-based (80/20) to simulate real-world prediction
- Target variable: Dream11 fantasy points (calculated using official scoring)

## Dream11 Scoring Used
| Action | Points |
|--------|--------|
| Run scored | +1 |
| Four | +1 bonus |
| Six | +2 bonus |
| 50 runs | +8 bonus |
| 100 runs | +16 bonus |
| Wicket | +25 |
| Catch | +8 |
| Stumping | +12 |

## Author
Netra Pawar — [LinkedIn](https://linkedin.com/in/netrap18) · [GitHub](https://github.com/netrap18)
