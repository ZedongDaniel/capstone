from cnn1d import ConvAutoencoder
import torch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt 

def data_to_tensor(data, dtype=torch.float32):
    device = torch.device("cpu")
    return torch.tensor(np.array(data), dtype=dtype).to(device)

class CnnSeqAutoEncoderModelConfig:
    in_channels = 1
    hidden_channels1 = 32
    hidden_channels2 = 16
    kernel_size = 7
    stride = 2
    padding = 3
    dropout_prob = 0.1
    seq_n = 100

class CnnSeqAutoEncoderModel:
    def __init__(self, model_path, data_path, threshold_dict):
        self.model = self._load_model(model_path)
        self.data = pd.read_csv(data_path, index_col='date') * 100
        self.model_data = self._prepare_data()

        with open(threshold_dict, "r") as json_file:
            self.threshold_dict = json.load(json_file)

        self.anomalies_sample = {}
        self.anomalies_index = {}
        
    def _load_model(self, model_path):
        model = self._init_model()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        model.eval()
        return model
    
    def _init_model(self):
        model = ConvAutoencoder(in_channels = CnnSeqAutoEncoderModelConfig.in_channels,
                                hidden_channels1 = CnnSeqAutoEncoderModelConfig.hidden_channels1, 
                                hidden_channels2 = CnnSeqAutoEncoderModelConfig.hidden_channels2,
                                kernel_size = CnnSeqAutoEncoderModelConfig.kernel_size,
                                stride = CnnSeqAutoEncoderModelConfig.stride,
                                padding = CnnSeqAutoEncoderModelConfig.padding, 
                                dropout_prob= CnnSeqAutoEncoderModelConfig.dropout_prob).to("cpu")
        return model
        

    def _prepare_data(self):
        data = self.data.copy()
        data = data.stack()
        df = data.reset_index()
        df.columns = ['date', 'sector', 'ret']
        df = df.set_index(['date', 'sector'])
        return df

    def set_threold(self, mae_path, quantile):
        with open(mae_path, "r") as json_file:
            mae_dict = json.load(json_file)

        for sector, mae in mae_dict.items():
             threshold = np.quantile(mae, quantile)
             self.threshold_dict[sector] = threshold

    def _sector_inference(self, sector_data):
        seq_n = CnnSeqAutoEncoderModelConfig.seq_n
        df = sector_data.swaplevel().sort_index().copy()
        sample_index = df.shift(seq_n-1).dropna().index.tolist()
        data_list = []
        for sample in sample_index:
            data_tensor = data_to_tensor(df.loc[:sample].iloc[-seq_n:].T)
            data_list.append(data_tensor)

        y_pred = []
        for X_i in data_list:
            with torch.no_grad():
                y_i = self.model(X_i).detach().cpu().numpy().T
                y_pred.append(y_i)
        y_true = [x.cpu().numpy().T for x in data_list]

        return np.array(y_pred), np.array(y_true)
    
    def sector_anomalies(self):
        for sector in  self.threshold_dict.keys():
            sector_df = self.model_data.loc[(slice(None), sector), :]
            y_pred, y_true = self._sector_inference(sector_data = sector_df)
            mae = np.mean(np.abs(y_pred - y_true), axis=1).squeeze()
            anomalies = mae > self.threshold_dict[sector]
            self.anomalies_sample[sector] = anomalies

    def sector_anomalies_index(self):
        '''
        data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        '''
        seq_n = CnnSeqAutoEncoderModelConfig.seq_n
        n = len(self.data)
        for sector in self.anomalies_sample.keys():
            sector_anomaly = self.anomalies_sample[sector]
            idx_ls = []
            for data_idx in range(seq_n - 1, n - seq_n + 1):
                if np.all(sector_anomaly[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
                    idx_ls.append(data_idx)
            self.anomalies_index[sector] = idx_ls

    def summary(self):
        output = pd.DataFrame(0, index= self.data.index, columns= self.data.columns)
        for sector, indices in self.anomalies_index.items():
            output.iloc[indices, output.columns.get_loc(sector)] = 1
        return output


    def _plot_anomalies(self, sector_name, ax):
        """Plot anomalies for a specific sector on a given axis."""
        sector = self.data.loc[:, sector_name]
        ax.plot(sector.index, sector, color="blue", linewidth=0.5)
        
        indx = self.anomalies_index[sector_name]
        ax.scatter(sector.index[indx], sector.iloc[indx], color="red", label="Anomalous Return", s=10)
        
        ax.set_title(f"Sector: {sector_name}")
        ax.set_ylabel("Return")
        ax.set_xticks(np.arange(0, len(self.data), 300))
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

    def plot_all_anomalies(self):
        """Generate a 3x4 subplot for all sectors."""
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True)
        fig.suptitle("Anomalies Across Sectors")
        
        axes = axes.flatten()
        
        for i, sector_name in enumerate(self.data.columns):
            self._plot_anomalies(sector_name, axes[i])

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        
        plt.tight_layout()
        plt.show()





if __name__ == "__main__":
    model = CnnSeqAutoEncoderModel(model_path="Model Cnn/cnn1d_seq.pth",
                        data_path="Data/log_ret.csv",
                        threshold_dict="Model Cnn/threshold.json")
    model.set_threold()
    model.sector_anomalies()
    model.sector_anomalies_index()
    print(model.summary())
    model.summary().to_csv('cnn_anomalies.csv')
    model.plot_all_anomalies()









