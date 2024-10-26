import numpy as np
import pandas as pd
import torch 
from cnn.cnn1d import ConvAutoencoder,data_to_tensor
import matplotlib.pyplot as plt 

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
    y_true = np.array([x.cpu().numpy().T for x in data_list])
    return np.array(y_pred)[:,:, :11], y_true[:,:, :11]



if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ret = pd.read_csv('data/mid_cap_all_sectors_ret.csv', index_col='date') * 100
    ret.columns = [f'{col}_ret' for col in ret.columns]
    vol = pd.read_csv('data/mid_cap_all_sectors_volume.csv', index_col='date')
    vol.columns = [f'{col}_volume' for col in vol.columns]
    full = pd.concat([ret, vol], axis=1)

    n = int(len(full) * 0.8)
    train_n = int(n * 0.95)
    tmp = full.iloc[:n]
    train_df = tmp.iloc[:train_n]
    valid_df = tmp.iloc[train_n:]
    test_df = full.iloc[n:]

    z_score_map = {}
    for col in train_df.columns:
        z_score_map[col] = (train_df[col].mean(), train_df[col].std())

    train_df_scale = train_df.copy()
    for col in train_df_scale.columns:
        mu, std = z_score_map[col] 
        train_df_scale[col] = (train_df_scale[col] - mu) / std
        
    valid_df_scale = valid_df.copy()
    for col in valid_df_scale.columns:
        mu, std = z_score_map[col] 
        valid_df_scale[col] = (valid_df_scale[col] - mu) / std

    test_df_scale = test_df.copy()
    for col in test_df_scale.columns:
        mu, std = z_score_map[col] 
        test_df_scale[col] = (test_df_scale[col] - mu) / std

    tmp_scale = tmp.copy()
    for col in tmp_scale.columns:
        mu, std = z_score_map[col] 
        tmp_scale[col] = (tmp_scale[col] - mu) / std
    
    input_dim = train_df.shape[1]
    seq_n = 100
    model_path = 'models/2024_10_25_cnn1d_return_volume.pth'
    model = ConvAutoencoder(in_channels = input_dim, 
                            hidden_channels1 = 32, 
                            hidden_channels2 = 16,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3, 
                            dropout_prob=0.2).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    y_train_pred, y_train = cnn_predict(model = model,data = train_df_scale ,seq_n = seq_n)

    print(y_train_pred.shape)
    print(y_train.shape)
    # plt.plot(y_train_pred[10][:,1], label = 'pred')
    # plt.plot(y_train[10][:,1], label = 'true')
    # plt.legend()
    # plt.show()

    train_mae = np.mean(np.abs(y_train_pred - y_train), axis=1)

    threshold = np.quantile(train_mae, 0.80, axis=0)
    print(threshold)

    fig, axes = plt.subplots(3, 4, figsize=(15, 10)) 
    axes = axes.flatten()
    for i in range(11):
        axes[i].hist(train_mae[:, i], bins=50) 
        axes[i].set_xlabel("Train MAE")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{train_df.columns[i]}")
        axes[i].axvline(threshold[i], color='red', linestyle='--', label=f'Threshold: {threshold[i]:.2f}')
        axes[i].legend() 
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()


    # # in sample
    y_in_sample_pred, y_in_sample = cnn_predict(model = model,data = tmp_scale ,seq_n = seq_n)
    in_sample_mae = np.mean(np.abs(y_in_sample_pred - y_in_sample), axis=1)

    # # # Detect in sample which are anomalies.
    anomalies_in_sample = in_sample_mae > threshold
    print("Number of anomaly samples in sample: ", np.sum(anomalies_in_sample))

    # # # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_in_sample = {}
    for i in range(11):
        sector_anomalies = anomalies_in_sample[:, i]
        column_name = tmp_scale.iloc[:,:11].columns[i]
        if column_name not in anomalous_data_in_sample:
            anomalous_data_in_sample[column_name] = []
        for data_idx in range(seq_n - 1, len(tmp) - seq_n + 1):
            if np.all(sector_anomalies[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
                anomalous_data_in_sample[column_name].append(data_idx)

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True) 
    axes = axes.flatten()

    for i, column_name in enumerate(tmp.columns[:11]):
        axes[i].plot(tmp[column_name], label="Return", color="blue", linewidth=0.5) 
        anomalous_indices = anomalous_data_in_sample.get(column_name, [])
        axes[i].scatter(anomalous_indices, tmp.iloc[anomalous_indices][column_name], color="red", label="Anomalous Return", s=10)
        
        axes[i].set_title(f"Sector: {column_name}")
        axes[i].set_ylabel("Return")
        axes[i].legend()
        
        axes[i].set_xticks(np.arange(0, len(tmp), 300))
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(tmp.columns[:11]), 12):
        fig.delaxes(axes[j])
    plt.suptitle('CNN Autoencoder In sample')
    plt.tight_layout()
    plt.show()


    # # out sample
    y_out_sample_pred, y_out_sample = cnn_predict(model = model,data = test_df_scale, seq_n = seq_n)
    out_sample_mae = np.mean(np.abs(y_out_sample_pred - y_out_sample), axis=1)

    # # Detect out sample which are anomalies.
    anomalies_out_sample = out_sample_mae > threshold
    print("Number of anomaly samples in sample: ", np.sum(anomalies_out_sample))

    # # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_out_sample = {}
    for i in range(11):
        sector_anomalies = anomalies_out_sample[:, i]
        column_name = test_df.columns[i]
        if column_name not in anomalous_data_out_sample:
            anomalous_data_out_sample[column_name] = []
        for data_idx in range(seq_n - 1, len(test_df) - seq_n + 1):
            if np.all(sector_anomalies[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
                anomalous_data_out_sample[column_name].append(data_idx)

   
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True) 
    axes = axes.flatten()

    for i, column_name in enumerate(test_df.columns[:11]):
        axes[i].plot(test_df[column_name], label="Return", color="blue", linewidth=0.5) 
        anomalous_indices = anomalous_data_out_sample.get(column_name, [])
        axes[i].scatter(anomalous_indices, test_df.iloc[anomalous_indices][column_name], color="red", label="Anomalous Return",  s=10)
        
        axes[i].set_title(f"Sector: {column_name}")
        axes[i].set_ylabel("Return")
        axes[i].legend()
        
        axes[i].set_xticks(np.arange(0, len(test_df), 200))
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(tmp.columns[:11]), 12):
        fig.delaxes(axes[j])
    plt.suptitle('CNN Autoencoder out sample')
    plt.tight_layout()
    plt.show()







