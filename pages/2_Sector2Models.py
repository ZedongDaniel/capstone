import streamlit as st
from pathlib import Path
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from stramlit_app_untils import load_sector_anomalies, load_sector_return

st.set_page_config(page_title="2_Sector2Models", page_icon="📈")

sectors = ['Materials', 'Industrials', 'Health Care', 'Real Estate', 'Consumer Discretionary', 'Financials', 
               'Utilities', 'Information Technology', 'Energy', 'Consumer Staples', 'Communication Services']


with st.sidebar:
    st.header("Input information")
    default_index = sectors.index('Materials') if 'Materials' in sectors else 0
    sector = st.selectbox('Sector Name', tuple(sectors), index=default_index)

    start_date = st.date_input('Start Date', value=datetime(2020, 10, 1), min_value=datetime(2019, 9, 26), max_value=datetime(2024, 8, 30))
    end_date = st.date_input("End Date", value= datetime(2022, 10, 30), min_value=datetime(2019, 9, 26), max_value=datetime(2024, 8, 30))

    if start_date >= end_date:
        st.error("End date must be after start date")

log_ret = load_sector_return(data_path="Data", sector=sector) * 100
anomalies = load_sector_anomalies(data_path="Anomalies Dataset", sector=sector)

curr_ret = log_ret.loc[start_date:end_date]
curr_anomalies = anomalies.loc[start_date:end_date, :]

st.write("placeholder")

fig, ax = plt.subplots(figsize=(20, 15))
ax.plot(curr_ret.index, curr_ret.values, color='blue',linewidth=0.5)
offsets = [-0.5, 0, 0.5, -1.0, 1.0, -1.5, 1.5]  # Adjust as needed for more models

for i, model in enumerate(curr_anomalies.columns):

    model_anomaly = curr_anomalies.loc[:, model]
    model_anomaly_dates = model_anomaly[model_anomaly == 1].index
    model_anomaly_values =  curr_ret.loc[model_anomaly == 1].values

    offset = offsets[i % len(offsets)]
    anomaly_display_values = model_anomaly_values + offset
    ax.scatter(model_anomaly_dates, anomaly_display_values, label=model, s=20)
    ax.legend()
st.pyplot(fig)


fig, axes = plt.subplots(7, 1, figsize=(12, 20), sharex=True, sharey=True)
axes = axes.flatten()
for i, model in enumerate(curr_anomalies.columns):
    ax = axes[i]    
    model_anomaly = curr_anomalies.loc[:, model]
    ax.plot(model_anomaly.index, model_anomaly.values, label=model)
    ax.set_title(model)
    ax.set_ylabel("anomaly indicator")
    ax.set_yticks([0, 1]) 
    ax.tick_params(axis='x') 

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
st.pyplot(fig)

