AI-Powered Cryptocurrency Analytics Dashboard
Overview

This project was developed as part of my internship at Amdocs, where I applied data analytics, machine learning, and time-series forecasting techniques to real-world cryptocurrency data.

The system is an end-to-end cryptocurrency analytics and forecasting dashboard that integrates real-time data, sentiment analysis, and multiple AI models to generate actionable insights.

Objective

To build a data-driven system that analyzes cryptocurrency trends, measures market volatility, and predicts future price movements using historical data.
Features
📊 Data Analysis
Real-time cryptocurrency data using CoinGecko API
Historical trend visualization
Returns and volatility analysis
Moving averages for trend detection
🤖 Forecasting Models
ARIMA – Statistical time-series model
LSTM – Deep learning model for sequential prediction
Prophet – Trend and seasonality forecasting
📈 Model Evaluation
Performance comparison using RMSE (Root Mean Squared Error)
Automatic selection of the best-performing model
🧠 Sentiment Analysis
Fetches crypto-related news using NewsAPI
Performs sentiment analysis using TextBlob
Captures overall market sentiment
💡 Trading Insight
Multi-factor signal based on:
Sentiment
Recent returns
Market volatility
Generates:
✅ BUY
❌ SELL
⚠️ HOLD
🖥️ Interactive Dashboard
Built using Streamlit
Interactive charts with Plotly
User-friendly interface for analysis
🛠️ Tech Stack
Python
Pandas, NumPy
Streamlit
Plotly
ARIMA (statsmodels)
LSTM (TensorFlow / Keras)
Prophet
TextBlob
APIs: CoinGecko, NewsAPI
📂 Project Structure
crypto-dashboard/
│
├── app.py
├── requirements.txt
└── README.md

🚀 Installation & Setup
git clone https://github.com/your-username/crypto-dashboard.git
cd crypto-dashboard
pip install -r requirements.txt
streamlit run app.py

🔄 Workflow
Collect historical and live cryptocurrency data
Preprocess and clean data
Perform exploratory data analysis
Apply forecasting models (ARIMA, LSTM, Prophet)
Evaluate model performance using RMSE
Generate trading insights
⚠️ Limitations
Sentiment analysis is not finance-specific
Limited historical data affects prediction accuracy
Not intended for real trading decisions
🔥 Future Improvements
Integrate FinBERT for advanced sentiment analysis
Add XGBoost / ensemble models
Implement backtesting system
Deploy on Streamlit Cloud
Improve feature engineering and model tuning
👩‍💻 Author
