Cryptocurrency Analytics Dashboard (AI-Powered)

An end-to-end AI-driven cryptocurrency analytics system that integrates real-time market data, sentiment analysis, and multiple forecasting models to generate actionable trading insights.

Overview

This project combines data analytics + machine learning + deep learning + NLP to analyze cryptocurrency trends and predict future prices.

It is designed to simulate a real-world financial analytics pipeline, including:

Data collection from APIs
Feature engineering
Model training & evaluation
Insight generation
⚙️ Features
📊 Data & Visualization
Real-time cryptocurrency data using CoinGecko API
Interactive price trend visualization (Plotly)
Moving averages (MA20)
Volatility analysis
Returns distribution histogram
🤖 Machine Learning Models
1. ARIMA (Statistical Model)
Time-series forecasting
Captures linear trends
2. LSTM (Deep Learning)
Sequential neural network for time-series prediction
Learns complex temporal patterns
3. Prophet (Facebook)
Handles seasonality and trend decomposition
Robust for business time-series forecasting
📈 Model Evaluation
Performance comparison using RMSE (Root Mean Squared Error)
Automatic selection of best-performing model
🧠 Sentiment Analysis (NLP)
News data fetched via NewsAPI
Sentiment scoring using TextBlob
Measures market mood (positive/negative/neutral)
💡 Trading Insight System
Multi-factor decision model combining:
Sentiment score
Recent returns
Market volatility
Generates:
✅ BUY
❌ SELL
⚠️ HOLD signals
🛠️ Tech Stack
Frontend: Streamlit
Data Processing: Pandas, NumPy
Visualization: Plotly
Machine Learning:
ARIMA (statsmodels)
LSTM (TensorFlow / Keras)
Prophet
NLP: TextBlob
APIs: CoinGecko, NewsAPI
📂 Project Structure
├── app.py              # Main Streamlit application
├── requirements.txt   # Dependencies
└── README.md          # Project documentation

🚀 How to Run
1. Clone Repository
git clone https://github.com/your-username/crypto-dashboard.git
cd crypto-dashboard

2. Install Dependencies
pip install -r requirements.txt

3. Run Application
streamlit run app.py

🔑 API Setup
Get free API key from: https://newsapi.org
Enter it in the sidebar to enable sentiment analysis
📊 Sample Workflow
Select cryptocurrency (BTC, ETH, etc.)
Fetch historical data
Analyze trends & volatility
Run forecasting models
Compare model performance
Generate trading signals
⚠️ Limitations
Sentiment analysis uses generic NLP (not finance-specific)
Models are trained on limited historical data
Not intended for real trading decisions
🔥 Future Improvements
Replace TextBlob with FinBERT (financial NLP)
Add XGBoost / Random Forest models
Implement backtesting system
Deploy using Streamlit Cloud
Add real-time trading signals
👩‍💻 Author

Rose Maria Jose
BCA (Cloud Technology & Information Security)
Aspiring Data Analyst / AI Engineer
