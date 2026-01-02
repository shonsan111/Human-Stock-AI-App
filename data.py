import yfinance as yf
import pandas as pd
import os

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

def save_user_prediction(row):
    file_exists = os.path.isfile("user_predictions.csv")
    row.to_csv(
        "user_predictions.csv",
        mode="a",
        header=not file_exists,
        index=False
    )
