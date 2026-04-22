
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


# CONFIG
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Cryptocurrency Analytics Dashboard")


# SIDEBAR
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
NEWS_API_KEY = st.sidebar.text_input("NewsAPI Key", type="password")


# DATA
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


# FEATURES
df["return"] = df["price"].pct_change()
df["volatility"] = df["return"].rolling(10).std()
df["MA20"] = df["price"].rolling(20).mean()
df.dropna(inplace=True)


# SENTIMENT
def get_sentiment():
    if not NEWS_API_KEY:
        return 0

    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": coin, "apiKey": NEWS_API_KEY, "pageSize": 20}
        news = requests.get(url, params=params).json()

        scores = []
        for a in news.get("articles", []):
            text = (a.get("title") or "") + " " + (a.get("description") or "")
            scores.append(TextBlob(text).sentiment.polarity)

        return np.mean(scores) if scores else 0
    except:
        return 0

sentiment_score = get_sentiment()


# METRICS
st.subheader("Key Metrics")
c1, c2, c3 = st.columns(3)

c1.metric("Price", round(df["price"].iloc[-1], 2))
c2.metric("Volatility", round(df["volatility"].iloc[-1], 4))
c3.metric("Sentiment", round(sentiment_score, 3))


# VISUALS
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name="Price"))
fig.add_trace(go.Scatter(x=df["date"], y=df["MA20"], name="MA20"))
st.plotly_chart(fig, use_container_width=True)


# SPLIT
train_size = int(len(df) * 0.8)
train = df["price"][:train_size]
test = df["price"][train_size:]


# ARIMA
rmse_arima = None
try:
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast_arima = model_fit.forecast(steps=len(test))
    rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))
except:
    pass

# LSTM
rmse_lstm = None
try:
    data = df["price"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    actual = scaler.inverse_transform(y_test)

    rmse_lstm = np.sqrt(mean_squared_error(actual, preds))
except:
    pass


# PROPHET
rmse_prophet = None
try:
    prophet_df = df.rename(columns={"date": "ds", "price": "y"})
    m = Prophet()
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    actual = prophet_df["y"][-30:]
    predicted = forecast["yhat"][-30:]

    rmse_prophet = np.sqrt(mean_squared_error(actual, predicted))
except:
    pass

# MODEL COMPARISON
st.subheader("Model Comparison")

rmse_dict = {
    "ARIMA": rmse_arima,
    "LSTM": rmse_lstm,
    "Prophet": rmse_prophet
}

rmse_dict = {k: v for k, v in rmse_dict.items() if v is not None}

if rmse_dict:
    best_model = min(rmse_dict, key=rmse_dict.get)
    st.success(f"Best Model: {best_model}")
    st.write(rmse_dict)
else:
    st.warning("No models available")


# TRADING SIGNAL
st.subheader("Trading Insight")

final_score = (
    0.5 * sentiment_score +
    0.3 * df["return"].iloc[-1] +
    0.2 * df["volatility"].iloc[-1]
)

if final_score > 0.05:
    st.success("BUY Signal")
elif final_score < -0.05:
    st.error("SELL Signal")
else:
    st.warning("HOLD")
