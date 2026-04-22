AI-Powered Cryptocurrency Analytics Dashboard

Project Description
This project is developed as part of my internship at Amdocs, focusing on analyzing cryptocurrency price trends using advanced time-series forecasting techniques.

The system integrates data analytics, statistical modeling, machine learning, and deep learning to predict future price movements based on historical data.

It includes a complete pipeline of:

Data collection
Preprocessing & feature engineering
Exploratory data analysis
Forecasting using multiple models

Objective
To build an intelligent analytics system that provides data-driven insights into cryptocurrency markets, helping users understand trends, volatility, and potential future movements.

Key Features
Data Analysis
Real-time cryptocurrency data using CoinGecko API
Historical trend visualization
Volatility and return analysis
Moving averages and pattern detection

Forecasting Models
ARIMA → Statistical time-series modeling
LSTM → Deep learning for sequential prediction
Prophet → Trend and seasonality forecasting

Model Evaluation
Performance comparison using RMSE
Automatic identification of the best-performing model

Sentiment Analysis
News data fetched using NewsAPI
Sentiment scoring using NLP (TextBlob)
Captures market mood and external influence

Intelligent Trading Insight
Multi-factor signal combining:
Sentiment
Returns
Volatility
Generates:
BUY
SELL
HOLD

Interactive Dashboard
Built with Streamlit
Real-time visualization using Plotly
User-friendly interface for financial analysis

Tech Stack
Python
Pandas, NumPy
Streamlit
Plotly
ARIMA (statsmodels)
LSTM (TensorFlow/Keras)
Prophet
TextBlob (NLP)
APIs: CoinGecko, NewsAPI

System Workflow
Collect historical and live crypto data
Clean and preprocess data
Perform exploratory data analysis
Apply forecasting models (ARIMA, LSTM, Prophet)
Evaluate model performance (RMSE)
Generate insights and trading signals

How to Run
git clone https://github.com/your-username/crypto-dashboard.git
cd crypto-dashboard
pip install -r requirements.txt
streamlit run app.py

Limitations
Sentiment analysis uses general NLP (not finance-specific)
Limited historical data affects prediction accuracy
Not intended for real-world trading decisions

Future Improvements
Integrate FinBERT for financial sentiment analysis
Add XGBoost / ensemble models
Implement backtesting system
Deploy on Streamlit Cloud
Improve feature engineering and model tuning
