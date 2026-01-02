import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go

from data import add_indicators, save_user_prediction
from model import train_on_user_data

st.set_page_config(page_title="Human-Based Stock AI")
st.title("🧠 Human-Trained Stock Predictor")
st.caption("AI trained only on YOUR predictions and notes")

# -------------------
# Stock input
# -------------------
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")

# -------------------
# TradingView Embed for price chart
# -------------------
st.subheader(f"{ticker} Price Chart (TradingView)")
st.markdown(f"""
<iframe src="https://s.tradingview.com/widgetembed/?symbol={ticker}&interval=60&hidesidetoolbar=1&symboledit=1&saveimage=0&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1" 
width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""", unsafe_allow_html=True)

# -------------------
# Fetch daily data for indicators
# -------------------
def fetch_daily_for_indicators(symbol):
    df = yf.download(symbol, period="1y", interval="1d")
    if df.empty:
        st.error("Could not fetch daily data for indicators.")
    df.index = pd.to_datetime(df.index)
    return df

df = fetch_daily_for_indicators(ticker)
if df.empty:
    st.stop()

df = add_indicators(df)

# Ensure numeric columns
for col in ["Close","MA10","MA50","Return"]:
    if col in df.columns:
        val = df[col]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:,0]
        if isinstance(val, np.ndarray) and val.ndim==2 and val.shape[1]==1:
            val = val.flatten()
        df[col] = pd.to_numeric(val, errors="coerce")
    else:
        st.error(f"Missing column: {col}")
        st.stop()

df_plot = df[["Close","MA10","MA50","Return"]].fillna(method="bfill")
df_plot.index = pd.to_datetime(df_plot.index)

# -------------------
# Info box for trends
# -------------------
st.subheader("Trend Indicators Info")
st.info("""
**Short-Term Trend (MA10):** Shows the 10-day moving average of closing prices.  
Indicates recent price direction and short-term momentum.  

**Medium-Term Trend (MA50):** Shows the 50-day moving average of closing prices.  
Indicates the overall trend over the past 2–3 months.
""")

# -------------------
# Plot function
# -------------------
def plot_chart(y,title,color,name):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:,0]
    elif isinstance(y, np.ndarray) and y.ndim==2 and y.shape[1]==1:
        y = y.flatten()
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_datetime(df_plot.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=color,width=2), name=name))
    fig.update_layout(template="plotly_dark", title=title, xaxis_title="Date", yaxis_title=name)
    st.plotly_chart(fig, use_container_width=True)

# -------------------
# Tabs for indicators
# -------------------
tab1, tab2, tab3 = st.tabs(["Short-Term Trend","Medium-Term Trend","Returns"])

with tab1:
    plot_chart(df_plot["MA10"], f"{ticker} Short-Term Trend","orange","MA10")

with tab2:
    plot_chart(df_plot["MA50"], f"{ticker} Medium-Term Trend","green","MA50")

with tab3:
    plot_chart(df_plot["Return"], f"{ticker} Returns","blue","Return")

# -------------------
# User prediction
# -------------------
st.subheader("Your Prediction")
user_choice = st.radio("What do you think the next move will be?", ["Up 📈","Down 📉"])
notes = st.text_area("Why are you making this prediction?", placeholder="Example: price bouncing, volume increasing...")

if st.button("Save My Prediction", key="save_prediction"):
    latest = df_plot.iloc[-1]
    row = pd.DataFrame([{
        "Date": latest.name,
        "Ticker": ticker,
        "MA10": latest["MA10"],
        "MA50": latest["MA50"],
        "Return": latest["Return"],
        "UserPrediction": 1 if "Up" in user_choice else 0,
        "Notes": notes
    }])
    save_user_prediction(row)
    st.success("Prediction saved!")

# -------------------
# AI learning
# -------------------
st.divider()
st.subheader("AI Learning From You")
st.info("""
The AI predicts **what you would likely say** based on your past predictions.  
It learns patterns from your choices and trends (MA10, MA50, Return).  
**Note:** This AI does not predict real market moves — it predicts your behavior.
""")

model, accuracy = train_on_user_data()
if model is None:
    st.info("Add at least 10 predictions before the AI activates.")
else:
    latest_features = df_plot.iloc[-1][["MA10","MA50","Return"]]
    ai_prediction = model.predict([latest_features])[0]
    st.write("AI prediction based on YOUR past behavior:")
    st.write("📈 UP" if ai_prediction==1 else "📉 DOWN")
    st.write(f"Accuracy: {accuracy:.2%}")

# -------------------
# Ask the AI for a prediction
# -------------------
st.subheader("Ask the AI for a Prediction")
if model is None:
    st.info("Add at least 10 predictions before the AI can give predictions.")
else:
    if st.button("Predict Next Move", key="predict_button"):
        latest_features = df_plot.iloc[-1][["MA10","MA50","Return"]]
        ai_prediction = model.predict([latest_features])[0]
        st.success(f"The AI predicts: {'📈 UP' if ai_prediction==1 else '📉 DOWN'}")
        st.write(f"Based on your past behavior. Accuracy: {accuracy:.2%}")
