from not_for_now.cnn2d import Cnn2dDataset, ConvAutoencoder2D
from train_utils import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)

    ret = pd.read_csv('data/mid_cap_all_sectors_ret.csv', index_col='date') * 100
    vol = pd.read_csv('data/mid_cap_all_sectors_volume.csv', index_col='date')

    ret_tmp, _ = train_test_split(ret, test_size=0.2, shuffle=False)
    ret_train, ret_valid = train_test_split(ret_tmp, test_size=0.05, shuffle=False)

    z_score_ret = {}
    for col in ret_train.columns:
        z_score_ret[col] = (ret_train[col].mean(), ret_train[col].std())

    ret_train = ret_train.copy()
    for col in ret_train.columns:
        mu, std = z_score_ret[col] 
        ret_train[col] = (ret_train[col] - mu) / std
    
    ret_valid = ret_valid.copy()
    for col in ret_valid.columns:
        mu, std = z_score_ret[col] 
        ret_valid[col] = (ret_valid[col] - mu) / std


    vol_tmp, _ = train_test_split(vol, test_size=0.2, shuffle=False)
    vol_train, vol_valid = train_test_split(vol_tmp, test_size=0.05, shuffle=False)


    z_score_vol = {}
    for col in vol_train.columns:
        z_score_vol[col] = (vol_train[col].mean(), vol_train[col].std())

    vol_train = vol_train.copy()
    for col in vol_train.columns:
        mu, std = z_score_vol[col] 
        vol_train[col] = (vol_train[col] - mu) / std
    
    vol_valid = vol_valid.copy()
    for col in vol_valid.columns:
        mu, std = z_score_vol[col] 
        vol_valid[col] = (vol_valid[col] - mu) / std

    train_combind = np.stack((ret_train.values, vol_train.values), axis= -1)
    train_combind = [[list(train_combind[i, j]) for j in range(train_combind.shape[1])] for i in range(train_combind.shape[0])]
    train_combind = pd.DataFrame(train_combind, index=ret_train.index, columns=ret_train.columns)

    valid_combind = np.stack((ret_valid.values, vol_valid.values), axis= -1)
    valid_combind = [[list(valid_combind[i, j]) for j in range(valid_combind.shape[1])] for i in range(valid_combind.shape[0])]
    valid_combind = pd.DataFrame(valid_combind, index=ret_valid.index, columns=ret_valid.columns)

    input_dim = 2
    seq_n = 100

    train_dataset = Cnn2dDataset(train_combind, seq_n)
    valid_dataset = Cnn2dDataset(valid_combind, seq_n)

    model = ConvAutoencoder2D(in_channels = input_dim, 
                            hidden_channels1 = 32, 
                            hidden_channels2 = 16,
                            kernel_size = (5,7),
                            stride = (1, 1),
                            padding = (1, 1), 
                            dropout_prob=0.1).to(device)
    
    train(
        model = model,
        batch_size = 32,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        num_epochs= 100,
        lr = 1e-3,
        loss_function = nn.MSELoss(),
        early_stop = False,
        patience = 10,
        verbose = True
    )

    model_path = 'models/2024_10_25_cnn2d.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")




