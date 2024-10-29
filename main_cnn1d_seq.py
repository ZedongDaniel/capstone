import numpy as np
import pandas as pd
import torch 
from cnn.cnn1d import ConvAutoencoder,data_to_tensor
import matplotlib.pyplot as plt 

def cnn_sector_predict(model, data, seq_n):
    df = data.swaplevel().sort_index().copy()
    sample_index = df.shift(seq_n-1).dropna().index.tolist()

    data_list = []
    for sample in sample_index:
        data_tensor = data_to_tensor(df.loc[:sample].iloc[-seq_n:].T)
        data_list.append(data_tensor)

    y_pred = []
    for X_i in data_list:
        with torch.no_grad():
            y_i = model(X_i).detach().cpu().numpy().T
            y_pred.append(y_i)
    y_true = [x.cpu().numpy().T for x in data_list]

    return np.array(y_pred), np.array(y_true)

def sector_sample_anomalies(data, sector_threshold):
    anomalies_sample_dict = {}
    for sector in sector_threshold.keys():
        sector_df = data.loc[(slice(None), sector), :]
        y_pred, y_true = cnn_sector_predict(model = model,data = sector_df ,seq_n = seq_n)
        mae = np.mean(np.abs(y_pred - y_true), axis=1).squeeze()
        anomalies = mae > sector_threshold[sector]
        anomalies_sample_dict[sector] = anomalies
    return anomalies_sample_dict

def sector_anomalies_index(date, anomalies_sample_dict):
    '''
    data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    '''
    anomalous_data_index_dict = {}
    for sector in sectors:
        sector_anomaly = anomalies_sample_dict[sector]
        idx_ls = []
        for data_idx in range(seq_n - 1, len(date) - seq_n + 1):
            if np.all(sector_anomaly[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
                idx_ls.append(data_idx)
        anomalous_data_index_dict[sector] = idx_ls
    return anomalous_data_index_dict

def plot_anomalies(data, dates, anomalies_index):
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True) 
    axes = axes.flatten()
    for i, sector in enumerate(anomalies_index.keys()):
        sector_df = data.loc[(slice(None), sector), :].reset_index()
        axes[i].plot(sector_df['date'],sector_df['ret'] , label="Return", color="blue", linewidth=0.5) 
        anomalous_indices = anomalies_index[sector]
        axes[i].scatter(anomalous_indices, sector_df['ret'].iloc[anomalous_indices], color="red", label="Anomalous Return", s=10)
        
        axes[i].set_title(f"Sector: {sector}")
        axes[i].set_ylabel("Return")
        axes[i].legend()
        
        axes[i].set_xticks(np.arange(0, len(dates), 300))
        axes[i].tick_params(axis='x', rotation=45)

    fig.delaxes(axes[-1])
    plt.suptitle('CNN Autoencoder')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sectors = ["Materials", "Industrials","Health Care","Real Estate","Consumer Discretionary","Financials",
                       "Utilities","Information Technology","Energy","Consumer Staples","Communication Services"]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ret = pd.read_csv('data/mid_cap_all_sectors_ret.csv', index_col='date') * 100
    ret = ret.stack()
    df = ret.reset_index()
    df.columns = ['date', 'sector', 'ret']
    df = df.set_index(['date', 'sector'])

    unique_dates = df.index.get_level_values('date').unique()
    n = int(len(unique_dates) * 0.8)
    train_n = int(n * 0.95)
    tmp_dates = unique_dates[:n]
    train_dates = tmp_dates[:train_n]
    valid_dates = tmp_dates[train_n:]
    test_dates = unique_dates[n:]

    train_df = df.loc[train_dates]
    valid_df = df.loc[valid_dates]
    full_df = df.loc[tmp_dates]
    test_df = df.loc[test_dates]

    
    input_dim = 1
    seq_n = 100
    model_path = 'models/2024_10_29_cnn1d_seq.pth'
    model = ConvAutoencoder(in_channels = input_dim, 
                            hidden_channels1 = 32, 
                            hidden_channels2 = 16,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3, 
                            dropout_prob=0.2).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    train_mae_dict = {}
    sector_threshold_dict = {}
    for sector in sectors:
        sector_df = train_df.loc[(slice(None), sector), :]
        y_train_pred, y_train = cnn_sector_predict(model = model,data = sector_df ,seq_n = seq_n)
        train_mae = np.mean(np.abs(y_train_pred - y_train), axis=1).squeeze()
        train_mae_dict[sector] = train_mae
        threshold = np.quantile(train_mae, 0.80, axis=0)
        sector_threshold_dict[sector] = threshold.item()

    # print(y_train_pred.shape)
    # print(y_train.shape)
    # print(y_train_pred[89])
    # print(y_train[89])
    # plt.plot(y_train_pred[89], label = 'pred')
    # plt.plot(y_train[89], label = 'true')
    # plt.legend()
    # plt.show()


    fig, axes = plt.subplots(3, 4, figsize=(15, 10)) 
    axes = axes.flatten()
    for i, sector in enumerate(sectors):
        axes[i].hist(train_mae_dict[sector], bins=50) 
        axes[i].set_xlabel("Train MAE")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{sector}")
        axes[i].axvline(sector_threshold_dict[sector], color='red', linestyle='--', label=f'Threshold: {sector_threshold_dict[sector]:.2f}')
        axes[i].legend() 
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()


    # # # in sample
    anomalies_in_sample_dict = sector_sample_anomalies(full_df, sector_threshold_dict)
    anomalous_data_index_insample = sector_anomalies_index(tmp_dates, anomalies_in_sample_dict)
    plot_anomalies(full_df, tmp_dates, anomalous_data_index_insample)


    # # # out sample
    anomalies_out_sample_dict = sector_sample_anomalies(test_df, sector_threshold_dict)
    anomalous_data_index_outsample = sector_anomalies_index(test_dates, anomalies_out_sample_dict)
    plot_anomalies(test_df, test_dates, anomalous_data_index_outsample)


