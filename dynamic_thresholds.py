import numpy as np
import pandas as pd

def compute_rolling_vol(data: pd.DataFrame, window_size: int=20):
    vol_df = data.rolling(window=window_size).std().dropna()
    return vol_df



def calc_vol_thershold(vol_df: pd.DataFrame, low: float = 0.3, high: float = 0.7):
    thresholds_vol = {}

    for sector in vol_df.columns:
        sector_vol = vol_df[sector]
        low_thresh = np.quantile(sector_vol, low)
        high_thresh = np.quantile(sector_vol, high)
        thresholds_vol[sector] = {'low': low_thresh, 'high': high_thresh}

    return thresholds_vol

def vol_regimes(vol_df: pd.DataFrame, vol_thershold: dict):
    regimes = pd.DataFrame(index=vol_df.index, columns=vol_df.columns)
    for sector in vol_df.columns:
        sector_vol = vol_df[sector]
        low_thresh = vol_thershold[sector]['low']
        high_thresh = vol_thershold[sector]['high']

        def regime_assignment(vol):
            if vol <= low_thresh:
                return 'low'
            elif vol <= high_thresh:
                return 'medium'
            else:
                return 'high'

        regimes[sector] = sector_vol.apply(regime_assignment)

    return regimes

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
        for data_idx in range(seq_n - 1, len(data) - seq_n + 1):
            anomaly_count = sector_anomalies.iloc[data_idx - seq_n + 1: data_idx + 1].sum()
            if anomaly_count >= threshold: 
                anomalies_index[sector].append(data_idx)

    return anomalies_index


# test data index 0, 1, 2, 3, ..., 20, ..., 80 .., 100
# am data index 0 : include test data index [x, y)
# am data index 0 : 0-20
# am data index 1 : 1-21
# am data index 2 : 2-22
# ...
# am data index 20 : 20-40
# ...
# am data index 60 : 60-80
# am data index 61 : 61-81

# am data index 80 : 80-100
