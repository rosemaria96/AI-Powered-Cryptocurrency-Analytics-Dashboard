# AI-Powered-Cryptocurrency-Analytics-Dashboard

##  Overview

This project is a **real-time cryptocurrency analytics and forecasting dashboard** that integrates historical data with live API data to analyze trends, measure volatility, and predict future prices.

The entire system is implemented in a single application using Streamlit.

---

## Features

*  **Real-Time Price Fetching**

  * Uses CoinGecko API to get live cryptocurrency prices

*  **Historical Trend Analysis**

  * Visualizes past price movements

*  **Volatility Analysis**

  * Calculates market risk using rolling standard deviation

*  **Price Forecasting**

  * Uses Facebook Prophet for time series prediction

*  **Market Insight**

  * Compares live price with predicted trend
  * Displays Bullish / Bearish signals

*  **Interactive Dashboard**

  * Built using Streamlit and Plotly

---

##  Tech Stack

* Python
* Pandas
* Streamlit
* Plotly
* Prophet
* Requests

##  Project Structure

crypto-dashboard/
│
├── crypto.csv
├── app.py
└── README.md


##  Installation & Setup

## 1. Clone the repository

```
git clone https://github.com/your-username/crypto-dashboard.git
cd crypto-dashboard
```

## 2. Install dependencies

```
pip install streamlit pandas plotly prophet requests
```

### 3. Run the application

```
streamlit run app.py
```

##  How It Works

1. Loads historical cryptocurrency data
2. Fetches real-time price using API
3. Cleans and preprocesses data
4. Performs volatility analysis
5. Applies forecasting model
6. Displays insights via dashboard


##  Future Improvements

* Modularize code into separate components
* Add multiple cryptocurrency support
* Integrate sentiment analysis
* Deploy to cloud

