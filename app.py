import streamlit as st
import pandas as pd
import json
from ml_model import train_models, predict_spread, ensemble_prediction
from data_fetch import fetch_latest_data
from utils import log_prediction, backtest

st.set_page_config(page_title="NFL Spread Predictor", layout="wide")

st.title("🏈 NFL Spread Predictor")

# Sidebar options
st.sidebar.header("Options")
mode = st.sidebar.radio("Choose mode:", ["Predict Upcoming Game", "Backtest Models"])

# Load historical dataset
@st.cache_data
def load_data():
    teams = pd.read_csv("data/nfl_teams.csv")
    scores = pd.read_csv("data/spreadspoke_scores.csv")
    return teams, scores

teams, scores = load_data()

# Train models on startup
models = train_models(scores)

if mode == "Predict Upcoming Game":
    st.subheader("Upcoming NFL Game Prediction")

    # Fetch real-time data
    latest_games = fetch_latest_data()
    st.write("Latest Games from API:", latest_games)

    selected_game = st.selectbox("Select Game:", latest_games["game"].tolist())

    if st.button("Predict Spread Outcome"):
        game_data = latest_games[latest_games["game"] == selected_game].iloc[0].to_dict()

        # JSON prompt
        json_prompt = json.dumps(game_data, indent=4)
        st.json(game_data)

        preds = predict_spread(models, game_data)
        final_pred = ensemble_prediction(preds)

        st.metric("Predicted Spread", f"{final_pred['spread']:.2f}")
        st.write(f"Recommendation: **{final_pred['recommendation']}** (Confidence: {final_pred['confidence']}%)")

        log_prediction(final_pred, "logs/predictions_log.csv")

elif mode == "Backtest Models":
    st.subheader("Backtest on Historical Data")
    results = backtest(models, scores)
    st.dataframe(results)
    st.line_chart(results["accuracy"])
