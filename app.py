import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Capstone Main",
    page_icon="👋",
)

st.title("Bloomberg Capstone Project")
st.markdown("""Jiayi Dong, Chenyu Liu, Yuxuan Liu, Murong Zhao, Zedong Chen, Qiwei Mo""")
st.write("## S&P 400 Mid-Cap Sector Return Anomaly Detection")
st.sidebar.success("Select a section above.")


st.markdown("""Welcome to the Anomaly Detection Dashboard for the S&P 400 Mid Cap sectors! 
            This tool provides insights into sector anomalies detected by various machine learning models. 
            Explore and analyze model performance across all 11 sectors or focus on a specific sector of interest.""")


st.markdown("""
### Data Analysis Overview

We are analyzing **SP400 Mid Cap Index companies** for return anomaly detection. 
The data was fetched using the **EOD Data API**, and we constructed equal-weight sector indices for a total of **11 sectors**. 
Using an **equal-weight index** can help some small-cap companies gain more prominence, enhancing the ability to identify anomalies.

#### Data Overview:
**Training Data**: Includes OHLCV and daily return data from **early 2000 to 2019-09-25** for training all models.  
**Out-of-Sample Data**: Covers the period from **2019-09-26 to 2024-08-30** for anomaly identification.

#### Sectors Analyzed:
1. **Materials**: Companies involved in mining, refining, and chemical production.
2. **Industrials**: Businesses focused on manufacturing and infrastructure.
3. **Health Care**: Providers of medical services, equipment, and pharmaceuticals.
4. **Real Estate**: Firms managing property investments and developments.
5. **Consumer Discretionary**: Retailers and luxury goods producers.
6. **Financials**: Banks, insurance, and investment firms.
7. **Utilities**: Providers of essential services like electricity and water.
8. **Information Technology**: Developers of software, hardware, and IT services.
9. **Energy**: Oil, gas, and renewable energy companies.
10. **Consumer Staples**: Producers of essential goods like food and household items.
11. **Communication Services**: Media, telecommunications, and entertainment providers.
""")


st.markdown("""
### Model Demonstration

We deployed **6 models**, here is a short demonstration of each model:

##### Dynamic Thresholding CNN Autoencoder
The CNN-Dynamic model uses two stacks of 1D CNNs in the encoder and decoder to reconstruct 15-day return sequences by minimizing MSE loss. 
During inference, it processes unseen sector returns, reconstructs the data, and flags anomalies when reconstruction errors exceed volatility-adjusted thresholds.
This dynamic thresholding adapts to different market conditions, enhancing anomaly detection.

##### Dynamic Thresholding LSTM Autoencoder     
Placeholder
            
##### K-nearest neighbors
Placeholder    
            
##### Isolation forest
Placeholder         
            
##### Mahalanobis distance
Placeholder
            
##### Statistics   
Placeholder

Please go to the **ModelComparison** page for a detailed quantitative comparison between models for anomaly flags across different sectors.
""")

st.markdown("""
### Consensus Model for Anomaly Detection

To enhance the robustness and accuracy of our analysis, we created a **Consensus Model** that combines the outputs of all models to identify meaningful anomaly points. 

#### Why Use a Consensus Model?
- **Improved Accuracy**: By aggregating results from multiple models, we reduce the likelihood of false positives and false negatives.
- **Reduced Bias**: Individual model biases are mitigated through a collective decision-making process.
- **Enhanced Robustness**: The consensus approach leverages diverse model strengths to improve reliability.

#### How It Works:
The consensus model operates on a **majority voting mechanism**:
If **three or more models** flag a specific date as anomalous, the consensus model marks that date as an anomaly.
            
Please go to the **ConsensusModelAnomaly** page for detailed identification of anomaly periods and dates.
""")

st.markdown("""
### Explore Anomalies by Sector and Date Range

If you'd like to delve deeper into the anomalies flagged by each of the six models for a specific sector, please visit the **ModelsOnSector** page. 

#### Features of the ModelsOnSector Page:
- **Sector Selection**: Choose a sector to visualize anomalies detected by the models.
- **Date Range Customization**: Define a specific start and end date to focus the analysis on your desired time window.
- **Visual Representation**: The page provides a clear comparison of anomaly points flagged by all six models for the selected sector and date range.

""")



