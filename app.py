# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from textblob import TextBlob
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from prophet import Prophet

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Cryptocurrency Analytics Dashboard")

# ================================
# CUSTOM CSS (UI BOOST)
# ================================
st.markdown("""
<style>
.metric-box {
    background-color: #1e1e1e;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR (UI IMPROVED)
# ================================
st.sidebar.header("⚙️ Controls")

coin_dict = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Cardano (ADA)": "cardano",
    "Solana (SOL)": "solana",
    "Ripple (XRP)": "ripple"
}

mode = st.sidebar.radio("Choose Input Method", ["Select Coin", "Enter Manually"])

if mode == "Select Coin":
    selected_coin = st.sidebar.selectbox("Select Cryptocurrency", list(coin_dict.keys()))
    coin = coin_dict[selected_coin]
else:
    coin = st.sidebar.text_input("Enter Coin ID", "bitcoin")

days = st.sidebar.slider("Days of history", 30, 365, 180)

st.sidebar.markdown("Get API key from https://newsapi.org")
NEWS_API_KEY = st.sidebar.text_input("Enter NewsAPI Key", type="password")

# ================================
# DATA COLLECTION
# ================================
@st.cache_data
def get_crypto_data(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": "usd", "days": days}

    r = requests.get(url, params=params)
    data = r.json()

    if "prices" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["date", "price"]]

df = get_crypto_data(coin, days)

if df.empty:
    st.error("Invalid coin")
    st.stop()

# ================================
# PREPROCESSING
# ================================
df["return"] = df["price"].pct_change()
df["volatility"] = df["return"].rolling(10).std()
df["MA20"] = df["price"].rolling(20).mean()
df.dropna(inplace=True)

# ================================
# KPI METRICS (NEW UI)
# ================================
st.subheader("Key Metrics")

sentiment_score = 0  # placeholder (will update later)

col1, col2, col3 = st.columns(3)
col1.metric("Latest Price", round(df['price'].iloc[-1], 2))
col2.metric("Volatility", f"{round(df['volatility'].iloc[-1],4)}")

# ================================
# SENTIMENT
# ================================
def get_news_sentiment():
    if not NEWS_API_KEY:
        return 0

    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": coin,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "pageSize": 20
        }

        r = requests.get(url, params=params).json()
        sentiments = []

        for a in r.get("articles", []):
            text = (a.get("title") or "") + " " + (a.get("description") or "")
            sentiments.append(TextBlob(text).sentiment.polarity)

        return np.mean(sentiments) if sentiments else 0

    except Exception:
        return 0

sentiment_score = get_news_sentiment()
col3.metric("Sentiment", round(sentiment_score, 3))

# ================================
# PRICE + VOLATILITY (SIDE BY SIDE)
# ================================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name="Price"))
fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Trend")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Volatility")
    st.line_chart(df.set_index("date")["volatility"])

# ================================
# RETURNS DISTRIBUTION
# ================================
st.subheader("Market Distribution")
hist = go.Figure(data=[go.Histogram(x=df["return"])])
st.plotly_chart(hist, use_container_width=True)

# ================================
# TRAIN TEST SPLIT
# ================================
train_size = int(len(df) * 0.8)
train = df["price"][:train_size]
test = df["price"][train_size:]

# ================================
# ARIMA
# ================================
rmse_arima = None
fig_arima = None

try:
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    rmse_arima = np.sqrt(mean_squared_error(test, forecast))

    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(y=test.values, name="Actual"))
    fig_arima.add_trace(go.Scatter(y=forecast.values, name="Forecast"))

except Exception as e:
    st.error(f"ARIMA error: {e}")

# ================================
# LSTM
# ================================
rmse_lstm = None
fig_lstm = None

try:
    if len(df) >= 50:
        data = df["price"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(10, len(scaled)):
            X.append(scaled[i-10:i, 0])
            y.append(scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(50, input_shape=(X.shape[1], 1)),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        preds = model.predict(X_test)
        preds = scaler.inverse_transform(preds)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse_lstm = np.sqrt(mean_squared_error(actual, preds))

        fig_lstm = go.Figure()
        fig_lstm.add_trace(go.Scatter(y=actual.flatten(), name="Actual"))
        fig_lstm.add_trace(go.Scatter(y=preds.flatten(), name="Prediction"))

except Exception as e:
    st.error(f"LSTM Error: {e}")

# ================================
# PROPHET
# ================================
fig_prophet = None

try:
    prophet_df = df.rename(columns={"date": "ds", "price": "y"})
    m = Prophet()
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual"))
    fig_prophet.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))

except Exception as e:
    st.error(f"Prophet error: {e}")

# ================================
# MODEL TABS (UI UPGRADE)
# ================================
st.subheader("Forecast Models")

tab1, tab2, tab3 = st.tabs(["ARIMA", "LSTM", "Prophet"])

with tab1:
    if fig_arima:
        st.plotly_chart(fig_arima)
        st.write(f"RMSE: {round(rmse_arima,2)}")

with tab2:
    if fig_lstm:
        st.plotly_chart(fig_lstm)
        st.write(f"RMSE: {round(rmse_lstm,2)}")

with tab3:
    if fig_prophet:
        st.plotly_chart(fig_prophet)

# ================================
# MODEL COMPARISON
# ================================
st.subheader(" Model Comparison")

if rmse_arima and rmse_lstm:
    better = "ARIMA" if rmse_arima < rmse_lstm else "LSTM"
    st.success(f"Better Model: {better}")

# ================================
# TRADING SIGNAL (UI IMPROVED)
# ================================
st.subheader("Trading Insight")

if sentiment_score > 0.1:
    st.success("BUY ")
elif sentiment_score < -0.1:
    st.error("SELL")
else:
    st.warning("HOLD")
