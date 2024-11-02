from nlp.news_extractor import SectorNewsExtractor, sector_keywords
from nlp.word_cloud import SectorWordCloud

import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os



st.title('Bloomberg Capstone Project: time series anomaly detection')




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

    news_extractor = SectorNewsExtractor("4af8dc7c-de0c-4ded-b276-492685db7350", sector_keywords, general_keywords=['mid cap', 's&p 400'])
    news_extractor.fetch_articles(sector_name=sector, date_start="2022-10-01", date_end="2022-10-30", max_articles=10)
    articles = news_extractor.get_articles()
    st.table(news_extractor.get_summary_table())
    world_cloud = SectorWordCloud(articles)
    world_cloud.generate_word_cloud(sector, instreamlit=True)


