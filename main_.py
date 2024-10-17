import numpy as np
import pandas as pd
import torch 
from Lstm_model import data_to_tensor, lstm_autoencoder,RecurrentAutoencoder
import matplotlib.pyplot as plt 


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

    model_path = 'model/autoencoder_2024_10_09.pth'
    model = RecurrentAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    sample_index = train_df.shift(seq_n-1).dropna().index.tolist()
    data_list = []
    for sample in sample_index:
        data_list.append(data_to_tensor(train_df.loc[:sample].iloc[-seq_n:]).unsqueeze(-1))

    y_train_pred = []
    for X_i in data_list:
        with torch.no_grad():
            y_i = model(X_i.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
        y_train_pred.append(y_i)

    plt.plot(data_list[100].cpu().numpy(), label = 'y_true')
    plt.plot(y_train_pred[100], label = 'y_pred')
    plt.legend()
    plt.show()

    print(y_train_pred[100])






