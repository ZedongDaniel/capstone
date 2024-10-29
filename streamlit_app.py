from nlp.news_extractor import SectorNewsExtractor
from nlp.word_cloud import SectorWordCloud

import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stock_data import StockData

st.title('Bloomberg Capstone Project: time series anomaly detection')

api_key = "cfd38812-1b95-4114-b0c0-39efacba95cf"
sector_keywords = {
    "Materials": ["mining", "chemical manufacturing", "raw materials"],
    "Industrials": ["manufacturing", "industrial equipment", "aerospace"],
    "Health Care": ["pharmaceuticals", "biotechnology", "health services"],
    "Real Estate": ["property development", "housing market", "commercial real estate"],
    "Consumer Discretionary": ["retail", "leisure products", "automobiles"],
    "Financials": ["banking", "financial services"],
    "Utilities": ["electricity", "natural gas", "water services"],
    "Information Technology": ["software", "hardware", "tech services"],
    "Energy": ["oil", "renewable energy", "gas"],
    "Consumer Staples": ["food products", "household goods", "beverages"],
    "Communication Services": ["telecom", "media", "advertising"]
}



sector_info = list(pd.read_csv('data/sp_400_midcap.csv')['GICS Sector'].unique())
all_return = pd.read_csv('data/mid_cap_all_sectors_ret.csv', index_col='date')
all_return.index = pd.to_datetime(all_return.index)

with st.sidebar:
    st.header("Input information")
    default_index = sector_info.index('Financials') if 'Financials' in sector_info else 0
    sector = st.selectbox('Sector Name', tuple(sector_info), index=default_index)

    start_date = st.date_input('Start Date', value=datetime(2022, 10, 1), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 8, 30))
    end_date = st.date_input("End Date", value= datetime(2022, 10, 7), min_value=datetime(2000, 1, 1), max_value=datetime(2024, 8, 30))

    if start_date > end_date:
        st.error("End date must be after start date")


st.write(f"Selected Sector: {sector}")
st.write(f"Selected Time Range: {start_date} to {end_date}")


with st.expander("View Sector Return"):
    ret = all_return.loc[start_date:end_date, sector]
    
    fig, ax = plt.subplots()
    ax.plot(ret.index, ret, label=sector)
    ax.set_title(f"{sector} Sector Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)

if st.button("Generate Word Cloud"):
    st.write(f"world cloud for : {sector}")

    news_extractor = SectorNewsExtractor(api_key, sector_keywords, general_keywords=["SP400 Mid Cap", "mid-cap stocks"])
    news_extractor.fetch_articles(sector_name=sector, date_start="2024-10-11", date_end="2024-10-11", max_articles=2)
    articles = news_extractor.get_articles()

    world_cloud = SectorWordCloud(articles)
    world_cloud.generate_word_cloud(sector, instreamlit=True)


