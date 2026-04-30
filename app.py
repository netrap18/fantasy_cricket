import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from utils.sample_data import generate_sample_dataset, IPL_TEAMS
from utils.data_processor import engineer_features, load_cricsheet_data, compute_match_stats
from utils.model import train_model, predict_xi, get_player_profile, FEATURE_COLS

st.set_page_config(page_title="IPL Fantasy XI Predictor", page_icon="🏏", layout="wide")

st.markdown("""
<style>
.stButton > button {
    background: #7c3aed; color: white; border: none;
    border-radius: 8px; font-weight: 600; width: 100%;
}
.stButton > button:hover { background: #6d28d9; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data(mode, data_folder=None):
    if mode == "sample":
        stats = generate_sample_dataset(n_matches=300)
    else:
        raw = load_cricsheet_data(data_folder)
        stats = compute_match_stats(raw)
    features = engineer_features(stats, n_recent=5)
    return stats, features


@st.cache_resource(show_spinner=False)
def get_model(key, features_df, model_type):
    return train_model(features_df, model_type=model_type)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 Fantasy XI Predictor")
    st.markdown("---")
    data_mode = st.radio("Data source", ["Demo (sample data)", "Real data (Cricsheet)"])
    data_folder = None
    if data_mode == "Real data (Cricsheet)":
        st.info("Download IPL CSVs from cricsheet.org → IPL → CSV format. Extract to data/ folder.")
        data_folder = st.text_input("Folder path", value="data/")
    st.markdown("---")
    model_type = st.selectbox("ML Algorithm", ["xgboost", "random_forest", "gradient_boosting"])
    st.markdown("---")
    st.markdown("""
**How it works**
1. Loads IPL ball-by-ball data  
2. Calculates Dream11 points per player  
3. Engineers rolling 5-match features  
4. Trains ML regression model  
5. Predicts your best XI
    """)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("# 🏏 IPL Fantasy XI Predictor")
st.markdown("*ML-powered Dream11 team selector — built by Netra Pawar*")
st.markdown("---")

mode_key = "sample" if "sample" in data_mode else "real"

with st.spinner("Loading data & training model — takes ~20 seconds first time..."):
    try:
        stats_df, features_df = load_data(mode_key, data_folder)
        results = get_model(f"{mode_key}_{model_type}", features_df, model_type)
        model = results['model']
        loaded = True
    except Exception as e:
        st.error(str(e))
        loaded = False

if loaded:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", results['model_type'].replace('_', ' ').title())
    c2.metric("RMSE", f"{results['rmse']} pts")
    c3.metric("MAE", f"{results['mae']} pts")
    c4.metric("Training records", f"{results['train_size']:,}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🎯 Predict My XI", "📊 Player Analysis", "🔬 Model Insights"])

    # ── TAB 1 ────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Select a Match")
        all_teams = sorted(features_df['team'].unique().tolist())
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", all_teams, index=0)
        with col2:
            opts = [t for t in all_teams if t != team1]
            team2 = st.selectbox("Team 2", opts, index=0)

        if st.button("⚡ Predict Best XI"):
            with st.spinner("Running predictions..."):
                preds = predict_xi(features_df, team1, team2, model)

            if preds.empty:
                st.warning("Not enough data for these teams.")
            else:
                st.subheader(f"Recommended XI — {team1} vs {team2}")
                top11 = preds.head(11).copy()
                top11.index = range(1, 12)

                def label(i, name):
                    if i == 1: return f"{name} 👑 C"
                    if i == 2: return f"{name} ⭐ VC"
                    return name

                top11['Player'] = [label(i, p) for i, p in enumerate(top11['Player'], 1)]
                top11['Form Trend'] = top11['Form Trend'].apply(
                    lambda x: f"↑ {abs(x):.1f}" if x > 0 else f"↓ {abs(x):.1f}")
                st.dataframe(top11, use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                raw = preds.head(11)
                colors = ['#7c3aed' if i == 0 else '#a78bfa' if i == 1 else '#c4b5fd'
                          for i in range(len(raw))]
                ax.barh(raw['Player'].tolist()[::-1],
                        raw['Predicted Points'].tolist()[::-1],
                        color=colors[::-1], edgecolor='none')
                ax.set_xlabel("Predicted Fantasy Points")
                ax.set_title(f"Predicted XI — {team1} vs {team2}", fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                handles = [mpatches.Patch(color='#7c3aed', label='Captain'),
                           mpatches.Patch(color='#a78bfa', label='Vice Captain'),
                           mpatches.Patch(color='#c4b5fd', label='Player')]
                ax.legend(handles=handles, loc='lower right')
                st.pyplot(fig)
                plt.close()
                st.success(f"👑 Captain: {preds['Player'].iloc[0]}  |  ⭐ VC: {preds['Player'].iloc[1]}")

    # ── TAB 2 ────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Player Form")
        all_players = sorted(features_df['player'].unique().tolist())
        sel = st.selectbox("Select player", all_players)
        profile = get_player_profile(features_df, sel)
        if profile:
            m1, m2, m3 = st.columns(3)
            m1.metric("Team", profile['team'])
            m2.metric("Avg Pts (Last 5)", profile['avg_pts_last5'])
            arrow = "↑" if profile['form_trend'] > 0 else "↓"
            m3.metric("Form Trend", f"{arrow} {abs(profile['form_trend']):.1f}")
            if profile['pts_history']:
                fig2, ax2 = plt.subplots(figsize=(9, 3.5))
                x = range(len(profile['pts_history']))
                ax2.plot(x, profile['pts_history'], marker='o', color='#7c3aed',
                         lw=2, markersize=6, markerfacecolor='white', markeredgewidth=2)
                ax2.fill_between(x, profile['pts_history'], alpha=0.1, color='#7c3aed')
                ax2.set_title(f"{sel} — Fantasy Points (Last 10 matches)", fontweight='bold')
                ax2.set_ylabel("Fantasy Points")
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.yaxis.grid(True, linestyle='--', alpha=0.4)
                st.pyplot(fig2)
                plt.close()

        st.markdown("---")
        st.subheader("Compare Two Players")
        cc1, cc2 = st.columns(2)
        with cc1: pa = st.selectbox("Player A", all_players, key='pa')
        with cc2: pb = st.selectbox("Player B", all_players, index=1, key='pb')
        if pa != pb:
            fig3, ax3 = plt.subplots(figsize=(9, 4))
            for name, color, marker in [(pa, '#7c3aed', 'o'), (pb, '#f59e0b', 's')]:
                d = features_df[features_df['player'] == name].sort_values('date')
                if not d.empty:
                    ax3.plot(d['actual_fantasy_pts'].tail(10).tolist(),
                             marker=marker, label=name, color=color, lw=2, markersize=5)
            ax3.set_title("Fantasy Points — Last 10 Matches", fontweight='bold')
            ax3.set_ylabel("Fantasy Points")
            ax3.legend()
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.yaxis.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig3)
            plt.close()

    # ── TAB 3 ────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Feature Importance — What drives predictions?")
        fi = results['feature_importance']
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.barh(fi['feature'].tolist()[::-1],
                 fi['importance'].tolist()[::-1], color='#7c3aed', edgecolor='none')
        ax4.set_title("Feature Importance", fontweight='bold')
        ax4.set_xlabel("Importance score")
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        st.pyplot(fig4)
        plt.close()

        st.markdown("---")
        st.subheader("Model Performance")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| RMSE | **{results['rmse']} pts** |
| MAE | **{results['mae']} pts** |
| Training records | **{results['train_size']:,}** |
| Test records | **{results['test_size']:,}** |
        """)
        if mode_key == "sample":
            st.warning("You are using demo data. For real predictions, download IPL CSVs from cricsheet.org and switch to 'Real data' mode.")
