from cnn.cnn1d import CNNChannelDataset, ConvAutoencoder, data_to_tensor
from train_utils import train
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import itertools
from tqdm import tqdm

def cnn_predict(model,data,seq_n):
    sample_index = data.shift(seq_n - 1).dropna().index.tolist()
    data_list = []
    for sample in sample_index:
        data_list.append(data_to_tensor(data.loc[:sample].iloc[-seq_n:].T))
        
    y_pred = []
    for X_i in data_list:
        with torch.no_grad():
            y_i = model(X_i).detach().cpu().numpy().T
            y_pred.append(y_i)
    y_true = [x.cpu().numpy().T for x in data_list]
    return np.array(y_pred), np.array(y_true)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)

    mid_cap_index = pd.read_csv('data/mid_cap_all_sectors_ret.csv', index_col='date')
    ret = mid_cap_index * 100
    n = int(len(ret) * 0.8)
    train_n = int(n * 0.8)
    tmp = ret.iloc[:n]
    train_df = tmp.iloc[:train_n]
    valid_df = tmp.iloc[train_n:]

    input_dim = train_df.shape[1]
    seq_n = 20
    train_dataset = CNNChannelDataset(train_df, seq_n)
    valid_dataset = CNNChannelDataset(valid_df, seq_n)

    param_grid = {
            'hidden_channels1': [16,32, 64],
            'activation_func': [nn.ReLU(), nn.LeakyReLU(), nn.Tanh()],
            'dropout_rate': [0.1, 0.2],
            'kernel_size': [3, 5, 7, 11],
            'lr': [1e-3]
        }
    
    param_list = [(hidden, func, dropout_rate, kernel, lr)
        for hidden, func, dropout_rate, kernel,lr  in itertools.product(
            param_grid['hidden_channels1'],
            param_grid['activation_func'],
            param_grid['dropout_rate'],
            param_grid['kernel_size'],
            param_grid['lr']
        )
    ]

    reslt = []
    for param in tqdm(param_list, desc="Hyperparameter Tuning"):
        hidden, func, dropout_rate, kernel, lr = param

        model = ConvAutoencoder(in_channels = input_dim, 
                                hidden_channels1 = hidden,
                                activation_func=func,
                                kernel_size = kernel,
                                stride = 2).to(device)

        train(
            model = model,
            batch_size = 32,
            train_dataset = train_dataset,
            valid_dataset = valid_dataset,
            num_epochs= 100,
            lr = lr,
            loss_function = nn.MSELoss(),
            early_stop = True,
            patience = 10,
            verbose = False
        )

        model.eval()
        y_pred, y_true = cnn_predict(model = model ,data = valid_df,seq_n = seq_n)
        mae = np.mean(np.abs(y_pred - y_true))

        reslt.append({
            'hidden_channels1': hidden,
            'activation_func': str(func),
            'dropout_rate': dropout_rate,
            'kernel_size': kernel,
            'lr':lr,
            'mae': mae
        })

    reslt = pd.DataFrame(reslt)
    reslt.to_csv('finetune.csv')

    best_model_params = reslt.loc[reslt['mae'].idxmin()]
    print(best_model_params)