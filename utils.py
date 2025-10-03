import pandas as pd
from datetime import datetime

def log_prediction(pred, log_file):
    df = pd.DataFrame([{
        "timestamp": datetime.now(),
        "spread": pred["spread"],
        "recommendation": pred["recommendation"],
        "confidence": pred["confidence"]
    }])
    df.to_csv(log_file, mode="a", header=not pd.io.common.file_exists(log_file), index=False)

def backtest(models, scores):
    results = []
    for idx, row in scores.dropna(subset=["spread_favorite"]).iterrows():
        game_data = {"spread_favorite": row["spread_favorite"]}
        preds = {name: model.predict_proba([[row["spread_favorite"]]])[0][1] for name, model in models.items()}
        avg_conf = sum(preds.values()) / len(preds)
        actual = 1 if (row["score_home"] - row["score_away"]) > row["spread_favorite"] else 0
        results.append({"game_id": row["game_id"], "prediction": avg_conf, "actual": actual})
    return pd.DataFrame(results)
