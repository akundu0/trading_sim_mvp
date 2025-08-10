import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Trading Strategy Simulator")

ticker = st.text_input("Ticker", "AAPL")
period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"])
interval = st.selectbox("Interval", ["1d", "1h", "15m"])

if st.button("Run Simulation"):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    data['SMA_5'] = data['Close'].rolling(5).mean()
    data['SMA_10'] = data['Close'].rolling(10).mean()

    # Plot
    st.line_chart(data[['Close', 'SMA_5', 'SMA_10']])
