# AI-Powered Cryptocurrency Analytics Dashboard

## Overview
This project was developed as part of my internship at Amdocs, where I applied data analytics, machine learning, and time-series forecasting techniques to real-world cryptocurrency data.

It is an end-to-end cryptocurrency analytics and forecasting dashboard that integrates real-time data, sentiment analysis, and multiple AI models to generate insights.

---

## Objective
To analyze cryptocurrency trends, measure market volatility, and predict future price movements using historical data.

---

## Features

### Data Analysis
- Real-time cryptocurrency data using CoinGecko API
- Historical trend visualization
- Returns and volatility analysis
- Moving averages for trend detection

### Forecasting Models
- ARIMA (Statistical time-series model)
- LSTM (Deep learning model)
- Prophet (Trend and seasonality forecasting)

### Model Evaluation
- Performance comparison using RMSE (Root Mean Squared Error)
- Automatic selection of best-performing model

### Sentiment Analysis
- News data fetched using NewsAPI
- Sentiment scoring using TextBlob
- Captures overall market sentiment

### Trading Insight
- Multi-factor signal based on:
  - Sentiment
  - Returns
  - Volatility

- Generates:
  - BUY
  - SELL
  - HOLD signals

### Interactive Dashboard
- Built using Streamlit
- Visualizations using Plotly
- User-friendly interface

---

## Tech Stack
- Python
- Pandas, NumPy
- Streamlit
- Plotly
- ARIMA (statsmodels)
- LSTM (TensorFlow/Keras)
- Prophet
- TextBlob
- APIs: CoinGecko, NewsAPI


---

## Installation & Setup

1. Clone the repository
git clone https://github.com/your-username/crypto-dashboard.git

2. Navigate to the folder
cd crypto-dashboard

4. Run the app
streamlit run app.py

---

## Workflow
1. Collect historical and live crypto data
2. Preprocess and clean data
3. Perform exploratory data analysis
4. Apply forecasting models (ARIMA, LSTM, Prophet)
5. Evaluate models using RMSE
6. Generate insights and trading signals

---

## Limitations
- Sentiment analysis is not finance-specific
- Limited historical data
- Not intended for real trading decisions

---

## Future Improvements
- Integrate FinBERT for better sentiment analysis
- Add XGBoost or ensemble models
- Implement backtesting system
- Deploy on Streamlit Cloud
- Improve feature engineering

---
