import streamlit as st
import Anomaly_Analysis_helper as aa
from stramlit_app_untils import get_model_dict, format_anomaly_periods


st.set_page_config(page_title="Consensus Model")

model_dict = get_model_dict()
model_list = list(model_dict.keys())
sectors = ['Materials', 'Industrials', 'Health Care', 'Real Estate', 'Consumer Discretionary', 'Financials', 
               'Utilities', 'Information Technology', 'Energy', 'Consumer Staples', 'Communication Services']

with st.sidebar:
    st.header("Input information")
    default_index = sectors.index('Materials') if 'Materials' in sectors else 0
    sector = st.selectbox('Sector Name', tuple(sectors), index=default_index)


st.markdown("""
## Consensus Model Anomaly Analysis

This page provides a comprehensive analysis of anomalies identified using the **Consensus Model**, which aggregates the outputs of multiple models through a hard voting technique. The focus is on understanding anomalies at both the sector-specific level and broader macroeconomic periods.

""")

majority_anomaly_consensus = aa.majority_anomalies_consensus(model_dict, consensus_threshold = 3, plot = True)
st.markdown("""
### Definitions and Methodology:
#### Sector-Specific Anomalies:
- **Identification**: Anomalies are flagged when the majority of models (**three or more**) agree on the same date.
- **Period Formation**: Anomalies occurring within **five consecutive days** are clustered into the same anomaly period. This ensures that short-term anomalies are grouped for easier analysis.

#### Macroeconomic Periods:
- **Definition**: The longest overlapping anomaly periods across all sectors are considered **macro events**, reflecting broader market or economic impacts.
- **Exclusion**: By removing macroeconomic periods, we isolate sector-specific anomalies that are independent of overarching market conditions.
""")




sector_anomaly_period = aa.anomaly_period(majority_anomaly_consensus)
macro_anomaly_periods = aa.macro_anomalies_period(majority_anomaly_consensus)
sector_specific_periods = aa.sector_specific_period(sector_anomaly_period, macro_anomaly_periods)


st.markdown("""
In addition, we present two tables to summarize the identified anomaly periods:

#### 1. Macroeconomic Anomaly Periods
This table lists the periods identified as **macro events** that impact all sectors.
""")
st.write(format_anomaly_periods(macro_anomaly_periods))

st.markdown("""
#### 2. Sector-Specific Anomaly Periods
This table provides the **start** and **end dates** for sector-specific anomaly periods, excluding the macroeconomic periods. 
Each row corresponds to a distinct sector anomaly period.
""")
sector_specific_period = sector_specific_periods[sector]
format_anomaly_periods(sector_specific_period, instreamlit=True)