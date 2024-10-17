import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def data_to_tensor(data, dtype=torch.float32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.tensor(np.array(data), dtype=dtype).squeeze(-1).to(device)

class AutoencoderDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.Series, seq_n: int) -> None:
        sample_index = data.shift(seq_n-1).dropna().index.tolist()
        self.data_list = []
        for sample in sample_index:
            data_tensor = data_to_tensor(data.loc[:sample].iloc[-seq_n:]).unsqueeze(0)
            data_tuple = (data_tensor, data_tensor)
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=7, padding=3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
