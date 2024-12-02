import streamlit as st
import Anomaly_Analysis_helper as aa
import pandas as pd
from stramlit_app_untils import get_model_dict, format_anomaly_periods


st.set_page_config(page_title="Consensus Model")

model_dict = get_model_dict()
model_list = list(model_dict.keys())

st.markdown("""
## Model Comparison: Sector-Wise Anomaly Analysis

This page provides a detailed quantitative comparison of the six models across the 11 analyzed sectors. 
We start with a heatmap showing the percentage of anomaly flags by each model for each sector. 
Next, we present a table summarizing the number of anomalies detected specifically in the Materials sector. 
Following that, we display a timeline distribution of anomalies detected by each model, aggregated over time. 
Finally, we conclude with a heatmap showcasing the Jaccard similarity between the six models.

### Key Findings:
- The **CNN Dynamic Model** flagged **13.13% anomalies** in the **Consumer Discretionary** sector.
- Among the six models, the **CNN Dynamic Model** shows the highest similarity with the **LSTM Dynamic Model**, indicating strong alignment between their anomaly detection patterns.
""")


st.markdown("### Heatmap: Anomaly Flags by Model and Sector")
anomaly_counts, anomaly_percentage = aa.summarize_anomalies(model_dict, print_info=False, plot=True, instreamlit=True)

st.markdown("### Table: Anomalies in Materials Sector")
st.table(anomaly_counts)

st.markdown("### Subplots: Anomaly Distribution Over Time (All Sectors)")
aa.plot_anomaly_distribution(model_dict, instreamlit=True)


st.markdown("""
### Jaccard Similarity Analysis

At the final step of our analysis, we measure the **Jaccard Similarity Index** between the models to understand how similar their anomaly detections are. This is visualized using a heatmap that highlights the pairwise similarity scores.

#### How Jaccard Similarity is Measured:
The Jaccard Similarity Index measures the overlap of anomalies flagged by two models. It is calculated as:
""")

st.latex(r"""
\text{Jaccard Similarity} = \frac{\text{Number of shared anomalies (intersection)}}{\text{Total number of anomalies flagged by either model (union)}}
""")

st.markdown("""
#### Steps:
1. Flatten the anomaly arrays for each pair of models.
2. Compute the **intersection** as the count of anomalies flagged by both models.
3. Compute the **union** as the total anomalies flagged by either model.
4. Divide the intersection by the union to obtain the similarity score.
5. For identical models, the similarity is set to 1.0.

#### Heatmap Visualization:
The heatmap displays the Jaccard Similarity Index for all model pairs:
- A higher value (closer to 1) indicates that the two models have high agreement in identifying anomalies.
- A lower value suggests less overlap in their flagged anomalies, indicating potential model diversity.
""")
aa.get_jaccard(model_dict, instreamlit=True)
