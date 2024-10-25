import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stock_data import StockData

st.title('Bloomberg Capstone Project: time series anomaly detection')

ticker_info = pd.read_csv('sp_400_midcap.csv')['Symbol'].tolist()

with st.sidebar:
    st.header("Input information")
    default_index = ticker_info.index('AA') if 'AA' in ticker_info else 0
    ticker = st.selectbox('Ticker Name', tuple(ticker_info), index=default_index)

    start_date = st.date_input('Start Date', value=datetime(2022, 1, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 8, 30))
    end_date = st.date_input("End Date", value= datetime(2022, 1, 30), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 8, 30))

    if start_date > end_date:
        st.error("End date must be after start date")


st.write(f"Selected Time Range: {ticker}")
st.write(f"Selected Time Range: {start_date} to {end_date}")

stock_data = StockData('sp_400_midcap.csv', '662166cb8e3d13.57537943')

with st.expander("View Close Price Time Series"):
    price_df = stock_data.fetch_stock(ticker=ticker, period='d', start=start_date, end=end_date)
    if price_df.empty:
        st.error("No data for {ticker} in time range {start_date} to {end_date}")
    else:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df.set_index('date', inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(price_df['close'], label='Close Price', color='blue')
        ax.set_title('Close Price Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)

