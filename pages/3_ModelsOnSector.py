import streamlit as st
from pathlib import Path
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from stramlit_app_untils import load_sector_anomalies, load_sector_return, load_sector_drawdown

st.set_page_config(page_title="Models Performacne on Specific Sector")

sectors = ['Materials', 'Industrials', 'Health Care', 'Real Estate', 'Consumer Discretionary', 'Financials', 
               'Utilities', 'Information Technology', 'Energy', 'Consumer Staples', 'Communication Services']

st.markdown("""
## Models on Sector: Anomaly Analysis by Sector and Date Range

This page allows you to explore anomalies flagged by the six models for a selected sector and within a user-defined date range. Through visualizations and insights, this section provides a deeper understanding of sector-specific return behavior and anomaly patterns.

### Visualizations:

#### 1. Sector Drawdown:
The first plot displays the **drawdown** for the selected sector's returns.

#### 2. Cumulative Return with Anomaly Markers:
The next plot shows the **cumulative return** for the sector's returns over time.
- **Six colored markers** are overlaid on the cumulative return time series, corresponding to the anomalies flagged by each model.
- **Insight**: Pay attention to regions where the colored markers cluster together, as this may indicate significant anomaly consensus among the models.

#### 3. Anomaly Indicators by Model:
A subplot presents the **anomaly indicators** for all six models:
  - **X-axis**: Time.
  - **Y-axis**: Binary indicator (0 or 1), where 1 represents an anomaly detected at a given date.
Each plot corresponds to one model, providing a clear timeline of anomalies flagged.

### User Interaction:
- **Sector Selection**: Choose a sector to focus the analysis on.
- **Date Range Customization**: Define the start and end dates to tailor the visualizations to your desired time window.
""")

with st.sidebar:
    st.header("Input information")
    default_index = sectors.index('Materials') if 'Materials' in sectors else 0
    sector = st.selectbox('Sector Name', tuple(sectors), index=default_index)

    start_date = st.date_input('Start Date', value=datetime(2020, 10, 1), min_value=datetime(2019, 9, 26), max_value=datetime(2024, 8, 30))
    end_date = st.date_input("End Date", value= datetime(2022, 10, 30), min_value=datetime(2019, 9, 26), max_value=datetime(2024, 8, 30))

    if start_date >= end_date:
        st.error("End date must be after start date")

log_ret = np.exp((load_sector_return(data_path="Data", sector=sector)).cumsum()) * 100
anomalies = load_sector_anomalies(data_path="Anomalies Dataset", sector=sector)
dd = load_sector_drawdown(data_path="Data", sector = sector) 


curr_ret = log_ret.loc[start_date:end_date]
curr_anomalies = anomalies.loc[start_date:end_date, :]
curr_drawdown = dd.loc[start_date:end_date]

fig, ax = plt.subplots(figsize=(20, 15))
ax.plot(curr_drawdown.index, curr_drawdown.values, color='red',linewidth=0.5)
ax.set_title(f"drawdown for {sector}")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(20, 15))
ax.plot(curr_ret.index, curr_ret.values, color='blue',linewidth=0.5)
ax.set_title(f"cumulative return for {sector}")
offsets = [-2, -1, 0, 1, 2, 3 ]  # Adjust as needed for more models
for i, model in enumerate(curr_anomalies.columns):
    model_anomaly = curr_anomalies.loc[:, model]
    model_anomaly_dates = model_anomaly[model_anomaly == 1].index
    model_anomaly_values =  curr_ret.loc[model_anomaly == 1].values

    offset = offsets[i % len(offsets)]
    anomaly_display_values = model_anomaly_values + offset
    ax.scatter(model_anomaly_dates, anomaly_display_values, label=model, s=20)
    ax.legend()
st.pyplot(fig)

fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True, sharey=True)
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

