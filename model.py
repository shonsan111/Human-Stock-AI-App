import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_on_user_data():
    try:
        df = pd.read_csv("user_predictions.csv")
    except:
        return None, None

    if len(df) < 10:
        return None, None

    X = df[["MA10", "MA50", "Return"]]
    y = df["UserPrediction"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    accuracy = model.score(X, y)
    return model, accuracy
