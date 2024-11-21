import json
from autoencoder import CnnChannelAutoEncoder
from dynamic_thresholds import (
    compute_rolling_vol,
    vol_regimes,
    detect_anomalies_sample,
    detect_anomalies_index,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CnnAnomalyDetector:
    def __init__(self, model:CnnChannelAutoEncoder, threshold_config: str):
        self.model = model
        self.seq_n = self.model.get_seq_n()
        self._load_threshold_config(threshold_config)

        self.data = None
        self.mae = None
        self.anomalies_index = None

    def load_data(self, data_path):
        self.data = pd.read_csv(data_path, index_col='date') * 100

    def compute_mae(self):
        y_pred, y_true = self.model.predict(self.data, output_ground_truth=True)
        mae = np.mean(np.abs(y_pred - y_true), axis=1)
        self.mae = pd.DataFrame(mae, index=self.data.shift(self.seq_n-1).dropna().index, columns=self.data.columns)

    def detect_anomalies(self, weight_on_seq: float):
        vol_data = compute_rolling_vol(self.data, window_size=self.seq_n)
        regimes = vol_regimes(vol_df=vol_data, vol_thershold=self.vol_threshold)
        anomalies_sample = detect_anomalies_sample(scores_df = self.mae, regimes_df = regimes, dynamic_thresholds= self.mae_dynamic_thresholds)
        self.anomalies_index_dict = detect_anomalies_index(data=self.data, anomalies_df = anomalies_sample, seq_n=self.seq_n, weight_on_seq=weight_on_seq)

    def generate_anomalies_index_dataframe(self, save_csv):
        output = pd.DataFrame(0, index=self.data.index, columns=self.data.columns)
        for sector in self.anomalies_index_dict.keys():
            index = self.anomalies_index_dict[sector]
            output.iloc[index, output.columns.get_loc(sector)] = 1

        if save_csv:
            output.to_csv('cnn_ouput.csv')
        return output
    
    def get_mae(self):
        return self.mae

    def _load_threshold_config(self, threshold_config: str):
        with open(threshold_config, 'r') as file:
            config = json.load(file)
        self.vol_threshold = {sector: data['vol_threshold'] for sector, data in config.items()}
        self.mae_dynamic_thresholds = {sector: data['mae_dynamic_thresholds'] for sector, data in config.items()}

    def plot(self):
        fig, axes = plt.subplots(3, 4, figsize=(10, 8), sharex=True) 
        axes = axes.flatten()
        for i, column_name in enumerate(self.data.columns):
            axes[i].plot(self.data[column_name], label="Return", color="blue", linewidth=0.5) 
            anomalous_indices = self.anomalies_index_dict.get(column_name, [])
            axes[i].scatter(anomalous_indices, self.data.iloc[anomalous_indices][column_name], color="red", label="Anomalous Return", s=10)
            
            axes[i].set_title(f"Sector: {column_name}")
            axes[i].set_ylabel("Return")
            axes[i].legend()
            
            axes[i].set_xticks(np.arange(0, len(self.data), 200))
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(len(self.data.columns), 12):
            fig.delaxes(axes[j])
        plt.suptitle('CNN Autoencoder anomalies')
        plt.tight_layout()
        plt.show()

    

    




if __name__ == "__main__":
    model = CnnChannelAutoEncoder('2024_11_06_cnn1d_channel.pth')
    detector = CnnAnomalyDetector(model=model, threshold_config = 'threshold_config.json')
    detector.load_data('../test_data.csv')
    detector.compute_mae()
    detector.detect_anomalies(0.5)
    output = detector.generate_anomalies_index_dataframe(save_csv=False)

    for setor in output.columns:
        print(f"{setor} count: {output.loc[:, setor].sum()}")
    detector.plot()
