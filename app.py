import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
import plotly.graph_objects as go
from data import add_indicators, save_user_prediction
from model import train_on_user_data
import pandas_datareader.data as web

st.set_page_config(page_title="Human-Based Stock AI + Macro Dashboard")
st.title("🧠 Human-Trained Stock Predictor")
st.caption("AI trained only on YOUR predictions and notes")

# --- Sidebar Tabs ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Stock Predictor", "Macro Dashboard"])

# -------------------
# --- STOCK PAGE ----
# -------------------
if page == "Stock Predictor":
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")

    # Fetch stock data (daily only)
    df = yf.download(ticker, period="1y", interval="1d")
    if df.empty:
        st.error("No stock data available.")
        st.stop()

    df.index = pd.to_datetime(df.index)

    # Flatten multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col]).strip('_') for col in df.columns]

    # Ensure Close exists
    if "Close" not in df.columns:
        close_candidates = [c for c in df.columns if "Close" in c]
        if not close_candidates:
            st.error("No Close column found.")
            st.stop()
        df["Close"] = df[close_candidates[0]]

    # Add indicators
    df = add_indicators(df)

    # Ensure numeric safely
    for col in ["Close","MA10","MA50","Return"]:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:,0]
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            st.error(f"Missing column: {col}")
            st.stop()

    df_plot = df[["Close","MA10","MA50","Return"]].fillna(method="bfill")

    # Tabs for charts
    tab1, tab2, tab3 = st.tabs(["Price Chart","Short-Term Trend","Medium-Term Trend"])

    def plot_chart(y, title, color, name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=y,
            mode='lines',
            line=dict(color=color, width=2),
            name=name
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
        plot_chart(df_plot["Close"], f"{ticker} Price","white","Price")
        st.info("Price Chart: Shows daily closing prices over the last year.")

    with tab2:
        plot_chart(df_plot["MA10"], f"{ticker} Short-Term Trend","orange","MA10")
        st.info("Short-Term Trend: 10-day moving average shows recent price momentum.")

    with tab3:
        plot_chart(df_plot["MA50"], f"{ticker} Medium-Term Trend","green","MA50")
        st.info("Medium-Term Trend: 50-day moving average shows longer-term trend direction.")

    # --- User prediction ---
    st.subheader("Your Prediction")
    user_choice = st.radio("What do you think the next move will be?", ["Up 📈","Down 📉"])
    notes = st.text_area("Why are you making this prediction?", placeholder="Example: price bouncing, volume increasing...")

    if st.button("Save My Prediction"):
        latest = df.iloc[-1]
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

    # --- AI prediction ---
    st.divider()
    st.subheader("AI Learning From You")
    model, accuracy = train_on_user_data()
    if model is None:
        st.info("Add at least 10 predictions before the AI activates.")
    else:
        latest_features = df.iloc[-1][["MA10","MA50","Return"]]
        ai_prediction = model.predict([latest_features])[0]
        st.write("AI prediction based on YOUR past behavior:")
        st.write("📈 UP" if ai_prediction == 1 else "📉 DOWN")
        st.write(f"Accuracy: {accuracy:.2%}")

        # Input for asking AI
        st.subheader("Ask AI for a prediction")
        if st.button("What will the AI predict next?"):
            st.write("AI predicts:")
            st.write("📈 UP" if ai_prediction == 1 else "📉 DOWN")

# -------------------
# --- MACRO PAGE ----
# -------------------
else:
    st.subheader("U.S. Macroeconomic Dashboard")
    macro_choice = st.radio("Select Variable", ["Inflation (CPI)", "Unemployment Rate", "GDP", "Interest Rate"])

    start = datetime.datetime(2015, 1, 1)
    end = datetime.datetime.today()

    try:
        if macro_choice == "Inflation (CPI)":
            df_macro = web.DataReader("CPIAUCSL", "fred", start, end)
            title = "U.S. Consumer Price Index (CPI)"
        elif macro_choice == "Unemployment Rate":
            df_macro = web.DataReader("UNRATE", "fred", start, end)
            title = "U.S. Unemployment Rate"
        elif macro_choice == "GDP":
            df_macro = web.DataReader("GDP", "fred", start, end)
            title = "U.S. GDP"
        elif macro_choice == "Interest Rate":
            df_macro = web.DataReader("FEDFUNDS", "fred", start, end)
            title = "U.S. Federal Funds Rate"

        fig_macro = go.Figure()
        fig_macro.add_trace(go.Scatter(
            x=df_macro.index,
            y=df_macro[df_macro.columns[0]],
            mode='lines',
            line=dict(color='cyan', width=2),
            name=macro_choice
        ))
        fig_macro.update_layout(
            title=title,
            template="plotly_dark",
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white")
        )

        st.plotly_chart(fig_macro, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load data for {macro_choice}. Error: {e}")
