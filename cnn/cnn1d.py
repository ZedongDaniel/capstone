import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def data_to_tensor(data, dtype=torch.float32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.tensor(np.array(data), dtype=dtype).to(device)

class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.Series, seq_n: int) -> None:
        sample_index = data.shift(seq_n-1).dropna().index.tolist()
        self.data_list = []
        for sample in sample_index:
            data_tensor = data_to_tensor(data.loc[:sample].iloc[-seq_n:].T)
            data_tuple = (data_tensor, data_tensor)
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
class CnnSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, selected_features, seq_n):
        df = data.swaplevel().sort_index().copy()
        sample_index = df.groupby('sector')[selected_features].shift(seq_n-1).dropna().index.tolist()
        inputs = df.loc[:, selected_features]

        self.data_list = []
        for sample in sample_index:
            data_tensor = data_to_tensor(inputs.loc[:sample].iloc[-seq_n:].T)
            data_tuple = (data_tensor, data_tensor)
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class CustomSectorLoss(nn.Module):
    def __init__(self, return_weight=2.0, volume_weight=1.0):
        super(CustomSectorLoss, self).__init__()
        self.return_weight = return_weight
        self.volume_weight = volume_weight
        self.mse_loss = nn.MSELoss(reduction='none') 

    def forward(self, true, pred):
        return_true = true[:, :11, :]  # First 11 channels for returns
        volume_true = true[:, 11:, :]  # Next 11 channels for volumes
        
        return_pred = pred[:, :11, :]  # First 11 channels for predicted returns
        volume_pred = pred[:, 11:, :]  # Next 11 channels for predicted volumes

        return_loss = self.mse_loss(return_true, return_pred)  # Return loss (batch, 11, seq_len)
        volume_loss = self.mse_loss(volume_true, volume_pred)  # Volume loss (batch, 11, seq_len)

        return_loss_mean = return_loss.mean()  # Mean return loss across sectors
        volume_loss_mean = volume_loss.mean()  # Mean volume loss

        total_loss = self.return_weight * return_loss_mean + self.volume_weight * volume_loss_mean

        return total_loss

class ConvAutoencoder(nn.Module):
    def __init__(self,in_channels, hidden_channels1, hidden_channels2, kernel_size, stride, padding, dropout_prob=0.2):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels= in_channels, out_channels= hidden_channels1, kernel_size= kernel_size, stride= stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(in_channels= hidden_channels1, out_channels= hidden_channels2, kernel_size= kernel_size, stride= stride, padding= padding),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels= hidden_channels2, out_channels= hidden_channels2, kernel_size= kernel_size,stride= stride, padding= padding, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose1d(in_channels= hidden_channels2, out_channels=hidden_channels1, kernel_size= kernel_size,stride= stride, padding= padding, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=hidden_channels1, out_channels=in_channels, kernel_size= kernel_size, padding=padding)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
