from cnn.cnn_sectors_ret_vol import CNNDataset, CustomSectorLoss, ConvAutoencoder
from train_utils import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f



if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ret = pd.read_csv('index_data/mid_cap_all_sectors_ret.csv', index_col='date') * 100
    ret.columns = [f'{col}_ret' for col in ret.columns]
    vol = pd.read_csv('index_data/mid_cap_all_sectors_volume.csv', index_col='date')
    vol.columns = [f'{col}_volume' for col in vol.columns]
    full = pd.concat([ret, vol], axis=1)

    n = int(len(full) * 0.8)
    train_n = int(n * 0.95)
    tmp = full.iloc[:n]
    train_df = tmp.iloc[:train_n]
    valid_df = tmp.iloc[train_n:]

    z_score_map = {}
    for col in train_df.columns:
        z_score_map[col] = (train_df[col].mean(), train_df[col].std())

    train_df = train_df.copy()
    for col in train_df.columns:
        mu, std = z_score_map[col] 
        train_df[col] = (train_df[col] - mu) / std
        
    valid_df = valid_df.copy()
    for col in valid_df.columns:
        mu, std = z_score_map[col] 
        valid_df[col] = (valid_df[col] - mu) / std



    seq_n = 100
    train_dataset = CNNDataset(train_df, seq_n)
    valid_dataset = CNNDataset(valid_df, seq_n)
    model = ConvAutoencoder(in_channels = 22, 
                            hidden_channels1 = 32, 
                            hidden_channels2 = 16,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3, 
                            dropout_prob=0.2).to(device)

    train(
        model = model,
        batch_size = 32,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        num_epochs= 200,
        lr = 1e-3,
        loss_function = CustomSectorLoss(),
        early_stop = False,
        patience = 10,
        verbose = True
    )

    model_path = 'models/2024_10_24_conv1d_sectors.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")