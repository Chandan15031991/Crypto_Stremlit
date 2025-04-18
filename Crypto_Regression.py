import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from binance.client import Client
from PIL import Image

# Initialize Binance Client
API_KEY = 'vi8jUaBge48S8qMbnpOZFg9hiMHh3F5AyxDyBD2AX6YYnUMS1DBdCix34ctaVw0W'
API_SECRET = 'ihrEeK0lQTSWvpVdGb2M4eQnTHlHca50X5Xv9onuTxCLKNIeiNc5H7iuokDYm69i'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# List of pairs
crypto_pairs = ['USDTTRY', 'BTCUSDT', 'USDTARS', 'USDTCOP', 'ETHUSDT', 'USDCUSDT',
                 'FDUSDUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'PENGUUSDT',
                 'BNBUSDT', 'MOVEUSDT', 'PEPEUSDT', 'USUALUSDT', 'FDUSDTRY',
                 'ZENUSDT', 'SUIUSDT', 'USDTBRL', 'HBARUSDT', 'TRXUSDT', 'VANAUSDT',
                 'PHAUSDT', 'ENAUSDT', 'ADAUSDT', 'MEUSDT', 'LINKUSDT', 'VIBUSDT',
                 'AAVEUSDT', 'AVAXUSDT', 'PNUTUSDT', 'STGUSDT', 'CRVUSDT',
                 'FTTUSDT', 'SHIBUSDT', 'LTCUSDT', 'AGLDUSDT', 'FTMUSDT',
                 'FLOKIUSDT', 'COWUSDT', 'GMTUSDT', 'WLDUSDT', 'WIFUSDT', 'LPTUSDT',
                 'BONKUSDT', 'UNIUSDT', 'EIGENUSDT', 'NEIROUSDT', 'NEARUSDT',
                 'DOTUSDT']

# Function to pull data from Binance API
def get_binance_data(symbol, interval, lookback):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback)
    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S"),
                                          end_time.strftime("%d %b %Y %H:%M:%S"))
    df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'Close time', 'Quote asset volume', 'Number of trades',
                                       'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    return df[['Open time', 'Close']]

# Function to run regression and categorize trends
def categorize_trends(crypto_pairs, interval, lookback):
    results = []

    for pair in crypto_pairs:
        try:
            df = get_binance_data(pair, interval, lookback)
            df['time_index'] = range(len(df))
            X = sm.add_constant(df['time_index'])
            y = df['Close']
            model = sm.OLS(y, X).fit()
            trend = model.params['time_index']

            if trend > 0.001:
                category = 'Positive Trend'
            elif trend < -0.001:
                category = 'Negative Trend'
            else:
                category = 'Flat'

            results.append({
                'Symbol': pair,
                'Trend Coefficient': trend,
                'Category': category,
                'R-squared': model.rsquared
            })
        except Exception as e:
            results.append({
                'Symbol': pair,
                'Trend Coefficient': None,
                'Category': 'Error',
                'R-squared': None
            })

    return pd.DataFrame(results)

# Streamlit App
def main():
    st.set_page_config(layout="wide")

    # Sidebar setup
    st.sidebar.image("Pic1.png", use_column_width=True)
    interval = st.sidebar.selectbox("Select Interval", ['1m', '5m', '15m', '1h', '1d'])
    lookback = st.sidebar.slider("Lookback (days)", min_value=1, max_value=30, step=1)
    symbol_to_visualize = st.sidebar.selectbox("Symbol to visualize", crypto_pairs)

    if st.sidebar.button("Run Regressions"):
        df_results = categorize_trends(crypto_pairs, interval, lookback)

        # Main page
        st.image("Pic2.png", use_column_width=True)
        st.markdown("<h1 style='text-align: center;'>Binance API - Crypto Regression Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<style>.block-container {padding-top: 0;}</style>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h2 style='text-align: center;'>Regression Results</h2>", unsafe_allow_html=True)
            st.dataframe(df_results)

        with col2:
            st.markdown(f"<h2 style='text-align: center;'>Scatterplot for {symbol_to_visualize}</h2>", unsafe_allow_html=True)
            data = get_binance_data(symbol_to_visualize, interval, lookback)
            X = sm.add_constant(range(len(data)))
            y = data['Close']
            model = sm.OLS(y, X).fit()
            data['Regression'] = model.predict(X)

            plt.figure(figsize=(10, 6))
            plt.scatter(data['Open time'], data['Close'], label='Close Prices', alpha=0.6)
            plt.plot(data['Open time'], data['Regression'], color='red', label='Regression Line')
            plt.xlabel("Time")
            plt.ylabel("Close Price")
            plt.title(f"Regression Analysis for {symbol_to_visualize}")
            plt.legend()
            st.pyplot(plt)

if __name__ == "__main__":
    main()


# Streamlit run Crypto_Regression.py