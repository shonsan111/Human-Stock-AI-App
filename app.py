import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Human-Based Stock AI")
st.title("🧠 Human-Trained Stock Predictor")
st.caption("AI trained only on YOUR predictions and notes")

# ------------------------------
# Sidebar: Stock Ticker & Range
# ------------------------------
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
range_option = st.sidebar.selectbox("Select Time Range", ["Yesterday", "Today", "1 Week", "1 Month", "1 Year"])

# ------------------------------
# Determine interval & period
# ------------------------------
now = datetime.datetime.now()
if range_option == "Today":
    interval = "1h"
    start = now.replace(hour=0, minute=0)
    end = now
elif range_option == "Yesterday":
    interval = "5m"
    yesterday = now - datetime.timedelta(days=1)
    start = yesterday.replace(hour=9, minute=30)
    end = yesterday.replace(hour=16, minute=0)
elif range_option == "1 Week":
    interval = "1h"
    start = now - datetime.timedelta(days=7)
    end = now
elif range_option == "1 Month":
    interval = "1d"
    start = now - datetime.timedelta(days=30)
    end = now
else:  # 1 Year
    interval = "1d"
    start = now - datetime.timedelta(days=365)
    end = now

# ------------------------------
# Fetch stock data using yfinance
# ------------------------------
df = yf.download(ticker, start=start, end=end, interval=interval)
if df.empty:
    st.error("No data available for this range.")
    st.stop()

df.index = pd.to_datetime(df.index)

# ------------------------------
# Add simple moving averages
# ------------------------------
df['MA10'] = df['Close'].rolling(10).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['Return'] = df['Close'].pct_change()

# ------------------------------
# Tabs for visualization
# ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Price Chart","Short-Term Trend","Medium-Term Trend","Macro Info"])

def plot_chart(y, title, color, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=y,
        mode='lines',
        line=dict(color=color, width=2),
        name=name,
        hovertemplate=f'<b>{name}: $%{{y:.2f}}</b><br>Date: %{{x|%Y-%m-%d %H:%M}}</extra>'
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showline=True, linecolor='gray', zeroline=False),
        yaxis=dict(showgrid=False, showline=True, linecolor='gray', zeroline=False)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab1:
    plot_chart(df["Close"], f"{ticker} Price","white","Price")
with tab2:
    plot_chart(df["MA10"], f"{ticker} Short-Term Trend","orange","MA10")
with tab3:
    plot_chart(df["MA50"], f"{ticker} Medium-Term Trend","green","MA50")
with tab4:
    st.write("📊 Macro Info: Example US Economic Variables")
    st.write("- Inflation, Interest Rates, GDP growth")
    st.info("This can later be updated with live macro data from a reliable source.")

# ------------------------------
# User prediction section
# ------------------------------
st.subheader("Your Prediction")
user_choice = st.radio("What do you think the next move will be?", ["Up 📈","Down 📉"])
notes = st.text_area("Why are you making this prediction?", placeholder="Example: price bouncing, volume increasing...")

if st.button("Save My Prediction"):
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    st.session_state.predictions.append({
        "MA10": df['MA10'].iloc[-1],
        "MA50": df['MA50'].iloc[-1],
        "Return": df['Return'].iloc[-1],
        "Prediction": 1 if "Up" in user_choice else 0,
        "Notes": notes
    })
    st.success("Prediction saved!")

# ------------------------------
# AI Prediction based on your past inputs
# ------------------------------
st.divider()
st.subheader("AI Learning From You")

if "predictions" in st.session_state and len(st.session_state.predictions) >= 3:
    data = pd.DataFrame(st.session_state.predictions)
    X = data[['MA10','MA50','Return']].fillna(0)
    y = data['Prediction']
    model = LogisticRegression()
    model.fit(X, y)
    latest_features = np.array([[df['MA10'].iloc[-1], df['MA50'].iloc[-1], df['Return'].iloc[-1]]])
    ai_prediction = model.predict(latest_features)[0]
    st.write("AI prediction based on YOUR past behavior:")
    st.write("📈 UP" if ai_prediction == 1 else "📉 DOWN")
else:
    st.info("Add at least 3 predictions before the AI activates.")
