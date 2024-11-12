import numpy as np
import pandas as pd

def compute_rolling_vol(data: pd.DataFrame, window_size=20):
    vol_df = data.rolling(window=window_size).std().dropna()
    return vol_df

def vol_regimes(vol_df: pd.DataFrame, low: float = 0.3, high: float = 0.7):
    thresholds_vol = {}
    regimes = pd.DataFrame(index=vol_df.index, columns=vol_df.columns)

    for sector in vol_df.columns:
        sector_vol = vol_df[sector]
        low_thresh = np.quantile(sector_vol, low)
        high_thresh = np.quantile(sector_vol, high)
        thresholds_vol[sector] = {'low': low_thresh, 'high': high_thresh}

        def regime_assignment(vol):
            if vol <= low_thresh:
                return 'low'
            elif vol <= high_thresh:
                return 'medium'
            else:
                return 'high'

        regimes[sector] = sector_vol.apply(regime_assignment)

    return regimes, thresholds_vol

def compute_dynamic_thresholds(scores_df: pd.DataFrame, regimes_df: pd.DataFrame, score_threshold: float =0.95):
    dynamic_thresholds = {}
    for sector in scores_df.columns:
        sector_scores = scores_df[sector]
        dynamic_thresholds[sector] = {}
        for regime in ['low', 'medium', 'high']:
            indices = regimes_df[sector] == regime
            scores_in_regime = sector_scores[indices]
            threshold = np.quantile(scores_in_regime, score_threshold)
            dynamic_thresholds[sector][regime] = threshold

    return dynamic_thresholds

def detect_anomalies_sample(scores_df: pd.DataFrame, regimes_df: pd.DataFrame, dynamic_thresholds: dict):
    anomalies = pd.DataFrame(False, index=scores_df.index, columns=scores_df.columns)

    for sector in scores_df.columns:
        setor_score = scores_df[sector]
        for regime in ['low', 'medium', 'high']:
            indices = regimes_df[sector] == regime
            threshold = dynamic_thresholds[sector][regime]
            anomalies.iloc[indices, anomalies.columns.get_loc(sector)] = (setor_score[indices] > threshold).astype(bool)

    return anomalies

def detect_anomalies_index(data: pd.DataFrame, anomalies_df: pd.DataFrame, seq_n: int, weight_on_seq: float):

    threshold = int(seq_n * weight_on_seq)

    anomalies_index = {}
    for i, sector in enumerate(anomalies_df.columns):
        sector_anomalies = anomalies_df.iloc[:, i]
        anomalies_index[sector] = []
        for data_idx in range(seq_n , len(data) - seq_n):
            anomaly_count = sector_anomalies.iloc[data_idx - seq_n : data_idx].sum()
            if anomaly_count >= threshold: 
                anomalies_index[sector].append(data_idx)

    return anomalies_index
