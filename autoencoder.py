from stock_data import StockData
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def data_to_tensor(data, dtype = torch.float32):
    return torch.tensor(np.array(data), dtype=dtype).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class autoencoder_dataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.Series, seq_n: int) -> None:
        sample_index = data.shift(seq_n-1).dropna().index.tolist()
        self.data_list = []
        for sample in sample_index:
            data_tuple = (data_to_tensor(data.loc[:sample][-seq_n:]),
                          data_to_tensor(data.loc[:sample].iloc[-seq_n:]))
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
class lstm_autoencoder(nn.Module):
    def __init__(self, input_size, seq_n):
        super(lstm_autoencoder, self).__init__()
        self.input_size = input_size
        self.seq_n = seq_n

        self.encoder_1 = nn.LSTM(input_size=input_size, hidden_size=8, batch_first=True)
        self.encoder_2 = nn.LSTM(input_size=8, hidden_size=4, batch_first=True)

        self.bridge = nn.Linear(4, 4)

        self.decoder_1 = nn.LSTM(input_size=4, hidden_size=4, batch_first=True)
        self.decoder_2 = nn.LSTM(input_size=4, hidden_size=8, batch_first=True)

        self.output_layer = nn.Linear(8, input_size)

    def forward(self, x: torch.tensor):
        # x: (b, t, 1)
        x, _ = self.encoder_1(x) # (b, t, 8)
        x, (h, c) = self.encoder_2(x) # (b, t, 4)

        x = self.bridge(h[-1,:,:])  # (b, 4)
        x = x.unsqueeze(1).repeat(1, self.seq_n, 1) # (b, 4) -> (b, 1, 4) -> (b, seq_n, 4)

        x, _ = self.decoder_1(x) # (b, seq_n, 4)
        x, _ = self.decoder_2(x)  # (b, seq_n, 8)

        x = self.output_layer(x) # (b, seq_n, 1)
        return x

























# class lstm_autoencoder(nn.Module):
#     def __init__(self, input_size, hidden_size, latent_size, num_layers = 1) -> None:
#         super(lstm_autoencoder, self).__init__()

#         self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.encoder_hidden = nn.Linear(hidden_size, latent_size)

#         self.decoder_hidden = nn.Linear(latent_size, hidden_size)
#         self.decoder_lstm = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

#     def forward(self, x: torch.tensor):
#         encoder_output, (encoder_hidden, encoder_cell) = self.encoder_lstm(x)
#         latent_layer = self.encoder_hidden(encoder_hidden[-1,:,:])

#         decoder_hidden = self.decoder_hidden(latent_layer)
#         decoder_hidden = decoder_hidden.unsqueeze(0)
        
#         decoder_input = torch.zeros_like(x)
#         decoder_output, _ = self.decoder_lstm(decoder_input, (decoder_hidden, encoder_cell))

#         return decoder_output