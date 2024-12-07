# 📦 Capstone Charlie Streamlit App 

This repository contains a comprehensive framework for detecting and analyzing S&P 400 mid-cap 11 sectors return series anomalies. It includes various models, datasets, and anomaly detection and analysis utilities.

## App Link

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://capstone-groupcharlie.streamlit.app/)

## Repository Structure

### 1. **Anomalies Dataset**
Contains CSV files with anomaly detection results from various models:
- `KNN.csv`: Results from KNN-based anomaly detection.
- `cnn_dynamic.csv`: Results from CNN-based dynamic detection.
- `isolation_forest.csv`: Results from Isolation Forest.
- `lstm_dynamic.csv`: Results from LSTM dynamic thresholds.
- `mahalanobis_distance.csv`: Results using Mahalanobis distance.
- `statistics.csv`: Statistical anomaly detection results.

### 2. **Anomaly_Analysis**
Scripts and notebooks for anomaly analysis:
- `Anomalies_Analysis.ipynb`: Jupyter notebook for detailed anomaly analysis.
- `Anomaly_Analysis_helper.py`: Helper functions for anomaly analysis.
- `test_data.csv`: Sample dataset for testing.

### 3. **Data**
Raw and processed financial data:
- `adjusted_close.csv`, `close.csv`, `high.csv`, `low.csv`, `open.csv`, `volume.csv`: Financial time series data.
- `log_ret.csv`: Log returns of the data.

### 4. **Model Cnn**
Resources for CNN-based anomaly detection:
- `2024_11_06_cnn1d_channel.pth`: Saved CNN model.
- `CnnAnomalyDetector.py`: Script for detecting anomalies using CNN.
- `autoencoder.py`: Autoencoder model for CNN-based detection.
- `dynamic_thresholds.py`: Script for dynamic thresholding.
- `threshold_config.json`: Configuration for thresholds.

### 5. **Model Isolation_Forest**
Files related to Isolation Forest:
- `isolation_forests.ipynb`: Jupyter notebook for Isolation Forest analysis.
- `isolation_forest.png`: Visualization of Isolation Forest results.

### 6. **Model Lstm**
LSTM-based anomaly detection resources:
- `LSTM_AE_dynamic.py`: LSTM Autoencoder for dynamic detection.
- `LSTM_AE_horizontal.py`: LSTM Autoencoder for horizontal detection.
- `dynamic_treshold.py`: Script for dynamic thresholding.
- `lstm_autoencoder_model_dynamic.keras`: Trained LSTM Autoencoder model.

### 7. **Model Mahalanobis**
Scripts for Mahalanobis distance anomaly detection:
- `mahalanobis_distance_anomalies.ipynb`: Jupyter notebook for Mahalanobis distance analysis.
- `mahalanobis_distance_anomalies_new.ipynb`: Updated analysis notebook.

### 8. **Model Stats**
Statistical anomaly detection resources:
- `Statistical_Model_updated.py`: Updated statistical model.
- `statistical_model_20241116.py`: Earlier version of the statistical model.

### 9. **Model KNN**
KNN-based anomaly detection:
- `KNN_dynamic.ipynb`: Jupyter notebook for dynamic KNN-based detection.

### 11. **Scripts and Utilities**
- `app.py`: Main application script.
- `index_construction.py`: Script for constructing indices.
- `stock_data.py`: Data processing utilities.
- `stramlit_app_untils.py`: Streamlit utilities for the app.

### 12. **NLP**
Natural language processing scripts:
- `news_extractor.py`: Extracts news data.
- `stopwords.txt`: Stopwords file for text processing.
- `word_cloud.py`: Script for generating word clouds.

### 13. **Pages**
Streamlit app pages:
- `1_ConsensusModelAnomaly.py`: Page for consensus model analysis.
- `2_ModelComparsion.py`: Page for model comparison.
- `3_ModelsOnSector.py`: Page for sector-based model analysis.

### 14. **Miscellaneous**
- `README.md`: This file.
- `requirements.txt`: Python dependencies.
- `sp_400_midcap.csv`: SP 400 midcap companies.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ZedongDaniel/capstone.git
   ```
   
2.	Navigate to the repository directory:
   ```bash
   cd anomaly-detection
   ```
3.	Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

