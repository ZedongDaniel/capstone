from Lstm_model import autoencoder_dataset, lstm_autoencoder,RecurrentAutoencoder
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

    mid_cap_index = pd.read_csv('index_data/mid_cap_index.csv', index_col='date')
    ret = mid_cap_index.loc[:,'log_ret'] * 100

    n = int(len(ret) * 0.8)
    train_n = int(n * 0.95)
    tmp = ret.iloc[:n]
    train_df = tmp.iloc[:train_n]
    valid_df = tmp.iloc[train_n:]

    seq_n = 100
    train_dataset = autoencoder_dataset(train_df, seq_n)
    valid_dataset = autoencoder_dataset(valid_df, seq_n)

    model = RecurrentAutoencoder().to(device)

    train(
        model = model,
        batch_size = 32,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        num_epochs= 200,
        lr = 1e-3,
        loss_function = nn.MSELoss(),
        early_stop = False,
        patience = 5,
        verbose = True
    )

    model_path = 'model/autoencoder_2024_10_15.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")







            


    