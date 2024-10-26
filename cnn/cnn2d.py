import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import torch
def data_to_tensor(data:pd.DataFrame, dtype=torch.float32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    returns = data.map(lambda x: x[0]).to_numpy() 
    volumes = data.map(lambda x: x[1]).to_numpy()

    full = np.stack([returns, volumes], axis=1)
    full = full.transpose(1, 2, 0)

    return torch.tensor(full, dtype=dtype).to(device)

class Cnn2dDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.Series, seq_n: int) -> None:
        sample_index = data.shift(seq_n-1).dropna().index.tolist()
        self.data_list = []
        for sample in sample_index:
            data_tensor = data_to_tensor(data.loc[:sample].iloc[-seq_n:])
            data_tuple = (data_tensor, data_tensor)
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    

class ConvAutoencoder2D(nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, kernel_size, stride, padding, dropout_prob=0.2):
        super(ConvAutoencoder2D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(in_channels=hidden_channels1, out_channels=hidden_channels2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_channels2, out_channels=hidden_channels1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose2d(in_channels=hidden_channels1, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
