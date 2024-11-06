import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

def data_to_tensor(data, dtype=torch.float32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.tensor(np.array(data), dtype=dtype).to(device)

class CNNChannelDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, seq_n: int, exclude = True) -> None:
        sample_index = data.shift(seq_n-1).dropna().index.tolist()
        self.data_list = []
        for sample in sample_index:
            if exclude:
                 exclude_date_start = pd.to_datetime('2008-08-01')
                 exclude_date_end = pd.to_datetime('2009-04-01')
                 if exclude_date_start <= pd.to_datetime(sample) <= exclude_date_end:
                    continue
                 
            data_tensor = data_to_tensor(data.loc[:sample].iloc[-seq_n:].T)
            data_tuple = (data_tensor, data_tensor)
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
class CnnSeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_n, exclude_date_start = '2008-08-01', exclude_date_end ='2009-04-01'):
        df = data.swaplevel().sort_index().copy()
        sample_index = df.groupby('sector').shift(seq_n-1).dropna().index.tolist()
        
        self.data_list = []
        for sample in sample_index:
            if pd.to_datetime(exclude_date_start) <= pd.to_datetime(sample[1]) <= pd.to_datetime(exclude_date_end):
                continue
            else:
                data_tensor = data_to_tensor(df.loc[:sample].iloc[-seq_n:].T)
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
    def __init__(self,in_channels, hidden_channels1, kernel_size, activation_func, stride = 2, dropout_prob=0.1):
        super(ConvAutoencoder, self).__init__()
        padding = math.ceil((kernel_size - stride) / 2)
        hidden_channels2 = hidden_channels1 // 2

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels= in_channels, out_channels= hidden_channels1, kernel_size= kernel_size, stride= stride, padding=padding), 
            activation_func,
            nn.Dropout(dropout_prob),
            nn.Conv1d(in_channels= hidden_channels1, out_channels= hidden_channels2, kernel_size= kernel_size, stride= stride, padding= padding),
            activation_func
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels= hidden_channels2, out_channels= hidden_channels2, kernel_size= kernel_size,stride= stride, padding= padding, output_padding=1),
            activation_func,
            nn.Dropout(dropout_prob),
            nn.ConvTranspose1d(in_channels= hidden_channels2, out_channels=hidden_channels1, kernel_size= kernel_size,stride= stride, padding= padding, output_padding=1),
            activation_func,
            nn.ConvTranspose1d(in_channels=hidden_channels1, out_channels=in_channels, kernel_size= kernel_size, padding=padding, output_padding=0)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# # Define an example input tensor with shape (batch_size, in_channels, sequence_length)
# batch_size = 32  # You can set the batch size as needed
# sequence_length = 20  # Length of the input sequence
# x = torch.randn(batch_size, 11, sequence_length)  # Example input for testing

# # Instantiate the model
# model = ConvAutoencoder(11, 32, 3, nn.LeakyReLU(), stride=2)

# # Forward pass
# output = model(x)
# print("Output shape:", output.shape)
    