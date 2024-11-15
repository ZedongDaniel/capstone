import streamlit as st

st.set_page_config(
    page_title="Capstone Main",
    page_icon="👋",
)

st.title("Bloomberg Capstone Project")
st.write("## S&P 400 Mid-Cap Sector Return Anomaly Detection")
st.sidebar.success("Select a section above.")

st.markdown("""
# Anomaly Detection Dashboard

Welcome to the Anomaly Detection Dashboard for the S&P 400 Mid Cap sectors! This tool provides insights into sector anomalies detected by various machine learning models. Explore and analyze model performance across all 11 sectors or focus on a specific sector of interest.

## Navigation Instructions

The app is organized into **three sections**:

### 1. Main Page
- **Purpose**: Overview of the tool and navigation instructions.
- **What to Do**: Use this page to familiarize yourself with the features of the dashboard and decide which analysis to perform.

### 2. **Sector-Wide Model Performance** (First Subpage)
- **Purpose**: Evaluate how a selected model performs across all 11 sectors simultaneously.
- **What to Do**:
  - Use the sidebar to **select a model** from the dropdown.
  - View a comprehensive anomaly map that visualizes the model's performance across all sectors.
- **Ideal for**: Comparing the effectiveness of a single model across the entire market.

### 3. **Model-Specific Sector Analysis** (Second Subpage)
- **Purpose**: Analyze all models' performances on a user-selected sector.
- **What to Do**:
  - Use the sidebar to **select a sector** from the dropdown.
  - View and compare the anomaly detection results from all available models for the chosen sector.
- **Ideal for**: In-depth analysis of a specific sector's anomaly patterns.

---

## Tips for Effective Use
- **Switch Between Pages**: Use the sidebar navigation to switch between the Main Page, Sector-Wide Model Performance, and Model-Specific Sector Analysis.
- **Customize Parameters**: Make use of the dropdown menus on the subpages to customize your analysis.
- **Understand the Outputs**:
  - On the **Sector-Wide Model Performance** page, focus on patterns and outliers in anomalies across sectors.
  - On the **Model-Specific Sector Analysis** page, identify which model performs best for your chosen sector.

Explore the dashboard and gain deeper insights into sector anomalies and model performance!
""")


