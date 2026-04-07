import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from prophet import Prophet

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("Crypto Analytics Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crypto.csv")  
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# -------------------------------
# SELECT COIN
# -------------------------------

selected_coin = "Bitcoin"

coin_df = df[['Date', 'Close']].copy()

# Prophet format
coin_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Convert date properly
coin_df['ds'] = pd.to_datetime(coin_df['ds'], errors='coerce')

# Remove invalid rows
coin_df = coin_df.dropna(subset=['ds'])

# Convert price to numeric
coin_df['y'] = coin_df['y'].replace(',', '', regex=True)
coin_df['y'] = pd.to_numeric(coin_df['y'], errors='coerce')

# Remove NaN in price
coin_df = coin_df.dropna(subset=['y'])

# Sort values
coin_df = coin_df.sort_values('ds')


# -------------------------------
# LIVE PRICE (API)
# -------------------------------
def get_live_price(coin):
    coin_map = {
        "Bitcoin": "bitcoin",
        "Ethereum": "ethereum",
        "Tether": "tether"
    }
    
    coin_id = coin_map.get(coin, "bitcoin")
    
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    
    try:
        data = requests.get(url).json()
        return data[coin_id]['usd']
    except:
        return None

live_price = get_live_price(selected_coin)

# -------------------------------
# DISPLAY LIVE PRICE
# -------------------------------
if live_price:
    st.metric(f"💰 Live {selected_coin} Price (USD)", live_price)
else:
    st.warning("Live price not available")

# -------------------------------
# PRICE TREND
# -------------------------------
st.subheader("Historical Price Trend")
fig1 = px.line(coin_df, x='ds', y='y', title=f"{selected_coin} Price")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# VOLATILITY
# -------------------------------
st.subheader("Volatility")

# FIX: convert to numeric
coin_df['y'] = coin_df['y'].replace(',', '', regex=True)
coin_df['y'] = pd.to_numeric(coin_df['y'], errors='coerce')

coin_df['returns'] = coin_df['y'].pct_change()
coin_df['volatility'] = coin_df['returns'].rolling(7).std()

fig2 = px.line(coin_df, x='ds', y='volatility', title="7-Day Volatility")
st.plotly_chart(fig2, use_container_width=True)


# -------------------------------
# FORECASTING
# -------------------------------
st.subheader("Forecast (Next 30 Days)")

model = Prophet()
model.fit(coin_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig3 = px.line(forecast, x='ds', y='yhat', title="Predicted Prices")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# INSIGHT
# -------------------------------
st.subheader("Market Insight")

if live_price:
    latest_prediction = forecast.iloc[-1]['yhat']
    
    if live_price > latest_prediction:
        st.success("Market is ABOVE predicted trend (Bullish)")
    else:
        st.error("Market is BELOW predicted trend (Bearish)")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built using Streamlit + Prophet + CoinGecko API")
