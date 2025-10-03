import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score

def train_models(scores):
    scores = scores.dropna(subset=["spread_favorite", "score_home", "score_away"])
    scores["spread_result"] = (scores["score_home"] - scores["score_away"]) - scores["spread_favorite"]

    scores["cover"] = (scores["spread_result"] > 0).astype(int)

    features = ["spread_favorite"]
    X = scores[features]
    y = scores["cover"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "logreg": LogisticRegression().fit(X_train, y_train),
        "rf": RandomForestClassifier().fit(X_train, y_train),
        "xgb": xgb.XGBClassifier().fit(X_train, y_train)
    }
    return models

def predict_spread(models, game_data):
    X_new = pd.DataFrame([[game_data["spread_favorite"]]], columns=["spread_favorite"])
    preds = {name: model.predict_proba(X_new)[0][1] for name, model in models.items()}
    return preds

def ensemble_prediction(preds):
    avg_conf = np.mean(list(preds.values())) * 100
    recommendation = "Favorite likely covers" if avg_conf > 50 else "Underdog likely covers"
    return {"spread": avg_conf, "recommendation": recommendation, "confidence": round(avg_conf, 2)}
