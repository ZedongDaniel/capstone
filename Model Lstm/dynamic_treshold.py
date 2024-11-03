import numpy as np
import pandas as pd

def compute_rolling_vol(data, window_size=20):
    volatility = data.rolling(window=window_size).std()
    return volatility


def assign_vol_regimes(volatility, low_percentile=30, high_percentile=70):
    """
    Assign volatility regimes ('low', 'medium', 'high') based on volatility percentiles.

    Returns:
    - regimes: pd.DataFrame
        DataFrame with the volatility regime ('low', 'medium', 'high') assigned to each time point.
    - thresholds: dict
        Dictionary containing low and high volatility thresholds per feature.
    """

    thresholds = {}
    regimes = pd.DataFrame(index=volatility.index, columns=volatility.columns)

    for feature in volatility.columns:
        vol_series = volatility[feature].dropna()
        low_thresh = np.percentile(vol_series, low_percentile)
        high_thresh = np.percentile(vol_series, high_percentile)
        thresholds[feature] = {'low': low_thresh, 'high': high_thresh}

        def regime_assignment(vol):
            if vol <= low_thresh:
                return 'low'
            elif vol <= high_thresh:
                return 'medium'
            else:
                return 'high'

        regimes[feature] = volatility[feature].apply(regime_assignment)

    return regimes, thresholds


def compute_dynamic_thresholds(train_scores, train_volatility_regimes, threshold_percentile=95):
    """
    Compute dynamic thresholds for anomaly detection based on volatility regimes.

    Returns:
    - dynamic_thresholds: dict
        Nested dictionary with thresholds per feature and regime.
    """
    dynamic_thresholds = {}

    if isinstance(train_scores, pd.Series):
        train_scores = train_scores.to_frame()

    for feature in train_scores.columns:
        dynamic_thresholds[feature] = {}
        for regime in ['low', 'medium', 'high']:
            indices = train_volatility_regimes[feature] == regime
            scores_in_regime = train_scores[feature][indices]
            if not scores_in_regime.empty:
                threshold = np.percentile(scores_in_regime, threshold_percentile)
                dynamic_thresholds[feature][regime] = threshold
            else:
                dynamic_thresholds[feature][regime] = None

    return dynamic_thresholds


def detect_anomalies(test_scores, test_volatility_regimes, dynamic_thresholds):
    """
    Detect anomalies in the test data using dynamic thresholds.

    Returns:
    - anomalies: pd.DataFrame
        DataFrame with binary anomaly labels (1 for anomaly, 0 for normal).
    """
    anomalies = pd.DataFrame(0, index=test_scores.index, columns=test_scores.columns)

    if isinstance(test_scores, pd.Series):
        test_scores = test_scores.to_frame()

    for feature in test_scores.columns:
        for regime in ['low', 'medium', 'high']:
            indices = test_volatility_regimes[feature] == regime
            threshold = dynamic_thresholds[feature][regime]
            if threshold is not None:
                anomalies[feature][indices] = (test_scores[feature][indices] > threshold).astype(int)
            else:
                anomalies[feature][indices] = 0

    return anomalies

