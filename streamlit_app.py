import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nlp.news_extractor import SectorNewsExtractor, sector_keywords
from nlp.word_cloud import SectorWordCloud
from stramlit_app_untils import get_model_dict, get_ohlc_dict


# load necessary info
model_dict = get_model_dict()
model_list = list(model_dict.keys())
ohlc_dict = get_ohlc_dict()

ret = ohlc_dict['log_ret']
close = ohlc_dict['close']



st.title('Bloomberg Capstone Project: time series anomaly detection')

with st.sidebar:
    st.header("Input information")
    default_index = model_list.index('cnn') if 'cnn' in model_list else 0
    model = st.selectbox('Model Name', tuple(model_list), index=default_index)

    start_date = st.date_input('Start Date', value=datetime(2020, 10, 1), min_value=datetime(2019, 9, 26), max_value=datetime(2024, 8, 30))
    end_date = st.date_input("End Date", value= datetime(2022, 10, 30), min_value=datetime(2019, 9, 26), max_value=datetime(2024, 8, 30))

    if start_date >= end_date:
        st.error("End date must be after start date")


st.write(f"Selected Model: {model}")
st.write(f"Selected Time Range: {start_date} to {end_date}")

anomalies = model_dict[model]


curr_ret = ret.loc[start_date:end_date,:]
curr_anomalies = anomalies.loc[start_date:end_date,:]

fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=False)
fig.suptitle("Sector Return & anomalies", fontsize=16)
axes = axes.flatten()

for i, sector in enumerate(curr_ret.columns):
    ax = axes[i]
    sector_ret = curr_ret[sector]
    sector_anomalies = curr_anomalies[sector]
    anomaly_dates = sector_ret[sector_anomalies == 1].index
    anomaly_values = sector_ret[sector_anomalies == 1].values
    
    ax.plot(sector_ret.index, sector_ret.values, label=sector, color='blue')
    ax.scatter(anomaly_dates, anomaly_values, color='red', marker='o', label='Anomaly')
    ax.set_title(sector)
    ax.set_ylabel("Return")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45) 
    ax.legend()

for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.subplots_adjust(top=0.9)

st.pyplot(fig)


with st.expander("View Close Price"):
    curr_close = close.loc[start_date:end_date, :]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=False)
    fig.suptitle("Sector Closing Prices", fontsize=16)
    axes = axes.flatten()

    for i, sector in enumerate(curr_close.columns):
        ax = axes[i]
        sector_data = curr_close[sector]
        
        ax.plot(sector_data.index, sector_data.values, label=sector, color='blue')
        ax.set_title(sector)
        ax.set_ylabel("Closing Price")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)


    st.pyplot(fig)


with st.expander("Word Cloud"):
    if st.button("Generate Word Cloud"):
        sector = "Financials"
        st.write(f"world cloud for : {sector}")
        news_extractor = SectorNewsExtractor("4af8dc7c-de0c-4ded-b276-492685db7350", sector_keywords, general_keywords=['mid cap', 's&p 400'])
        news_extractor.fetch_articles(sector_name=sector, date_start=start_date, date_end=start_date, max_articles=10)
        articles = news_extractor.get_articles()
        st.table(news_extractor.get_summary_table())
        world_cloud = SectorWordCloud(articles)
        world_cloud.generate_word_cloud(sector, instreamlit=True)

