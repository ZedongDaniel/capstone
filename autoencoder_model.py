import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def data_to_tensor(data, dtype = torch.float32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.tensor(np.array(data), dtype=dtype).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class autoencoder_dataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.Series, seq_n: int) -> None:
        sample_index = data.shift(seq_n-1).dropna().index.tolist()
        self.data_list = []
        for sample in sample_index:
            data_tuple = (data_to_tensor(data.loc[:sample].iloc[-seq_n:]).unsqueeze(-1),
                          data_to_tensor(data.loc[:sample].iloc[-seq_n:]).unsqueeze(-1))
            self.data_list.append(data_tuple)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]


class RecurrentEncoder(nn.Module):
    def __init__(self):
        super(RecurrentEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=100, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=30, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(x)
        return h_n.squeeze(0)

class RecurrentDecoder(nn.Module):
    def __init__(self):
        super(RecurrentDecoder, self).__init__()
        self.repeat_vector = 100  
        self.lstm = nn.LSTM(input_size=30, hidden_size=100, batch_first=True)
        self.dense = nn.Linear(100, 1)

    def forward(self, x):
        # Repeat the input vector across the sequence length
        x = x.unsqueeze(1).repeat(1, self.repeat_vector, 1)
        x, _ = self.lstm(x)
        x = self.dense(x)
        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = RecurrentEncoder()
        self.decoder = RecurrentDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class lstm_autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers=1):
        super(lstm_autoencoder, self).__init__()
        self.relu = nn.ReLU()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_hidden = nn.Linear(hidden_size, latent_size)

        self.decoder_hidden = nn.Linear(latent_size, input_size)  # Project to input_size for hidden state compatibility
        self.decoder_cell = nn.Linear(hidden_size, input_size)  # Project cell state to input_size as well
        self.decoder_lstm = nn.LSTM(input_size, input_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor):
        # Encoding
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder_lstm(x)
        latent_layer = self.encoder_hidden(encoder_hidden[-1, :, :])  # (batch, latent_size)

        # Decoding
        decoder_hidden = self.decoder_hidden(latent_layer).unsqueeze(0)  # (1, batch, input_size)
        decoder_cell = self.decoder_cell(encoder_cell[-1, :, :]).unsqueeze(0)  # (1, batch, input_size)
        
        decoder_output, _ = self.decoder_lstm(x, (decoder_hidden, decoder_cell))

        return decoder_output
    
if __name__ == "__main__":
    num_params = count_parameters(lstm_autoencoder(input_size = 1, hidden_size = 64, latent_size= 32, num_layers = 1))
    print(f"total params: {num_params}")

    


























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