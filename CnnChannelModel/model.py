from cnn1d import ConvAutoencoder
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn as nn


class CnnChannelAutoEncoderModelConfig:
    in_channels = 11
    hidden_channels1 = 64
    activation_func = nn.Tanh()
    kernel_size = 3
    stride = 2

class CnnChannelAutoEncoder:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.seq_n = 20

    def predict(self, data: pd.DataFrame, output_ground_truth: bool):
        data_list = self._process_data(data)
        y_pred = []
        for X_i in data_list:
            with torch.no_grad():
                y_i = self.model(X_i).detach().cpu().numpy().T
                y_pred.append(y_i)
        
        if output_ground_truth:
            y_true = [x.cpu().numpy().T for x in data_list]
            return np.array(y_pred), np.array(y_true)

        return np.array(y_pred)
    
    def get_seq_n(self):
        return self.seq_n
    
    def _load_model(self, model_path: str):
        model = self._init_model()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    def _init_model(self):
        model = ConvAutoencoder(in_channels = CnnChannelAutoEncoderModelConfig.in_channels, 
                                hidden_channels1 = CnnChannelAutoEncoderModelConfig.hidden_channels1,
                                activation_func=CnnChannelAutoEncoderModelConfig.activation_func,
                                kernel_size = CnnChannelAutoEncoderModelConfig.kernel_size,
                                stride = CnnChannelAutoEncoderModelConfig.stride).to("cpu")
        return model
    
    def _process_data(self, data: pd.DataFrame):
        sample_index = data.shift(self.seq_n - 1).dropna().index.tolist()
        data_list = []
        for sample in sample_index:
            data_list.append(self._data_to_tensor(data.loc[:sample].iloc[-self.seq_n:].T))

        return data_list
    
    def _data_to_tensor(self, data, dtype=torch.float32):
        device = torch.device("cpu")
        return torch.tensor(np.array(data), dtype=dtype).to(device)
    
    def __str__(self):
        return str(self.model)
    

if __name__ == "__main__":
    m = CnnChannelAutoEncoder('CnnChannelModel/2024_11_06_cnn1d_channel.pth')
    data = pd.read_csv("data/test_data.csv", index_col='date') * 100
    y_pred, y_true = m.predict(data, output_ground_truth=True)

    plt.plot(y_pred[60][:, 7], label = 'pred')
    plt.plot(y_true[60][:, 7], label = 'true')
    plt.legend()
    plt.show()

    print(m)

