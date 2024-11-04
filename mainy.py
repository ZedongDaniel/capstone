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
    return np.array(y_pred), y_true

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    mid_cap_index = pd.read_csv('data/mid_cap_all_sectors_ret.csv', index_col='date')
    ret = mid_cap_index * 100
    n = int(len(ret) * 0.8)
    train_n = int(n * 0.8)
    tmp = ret.iloc[:n]
    train_df = tmp.iloc[:train_n]
    valid_df = tmp.iloc[train_n:]
    test_df = ret.iloc[n:]

    print((train_df.shape, valid_df.shape, test_df.shape))
    input_dim = train_df.shape[1]
    seq_n = 100
    model_path = 'models_repo/2024_11_03_cnn1d_channel.pth'
    model = ConvAutoencoder(in_channels = input_dim, 
                            hidden_channels1 = 32, 
                            hidden_channels2 = 16,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    y_train_pred, y_train = cnn_predict(model = model,data = valid_df,seq_n = seq_n)
    # print(y_train_pred.shape)
    # print(y_train.shape)
    # plt.plot(y_train_pred[100][:,0], label = 'pred')
    # plt.plot(y_train[100][:,0], label = 'true')
    # plt.legend()
    # plt.show()

    train_mae = np.mean(np.abs(y_train_pred - y_train), axis=1)

    threshold = np.quantile(train_mae, 0.85, axis=0)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8)) 
    axes = axes.flatten()
    for i in range(input_dim):
        axes[i].hist(train_mae[:, i], bins=50) 
        axes[i].set_xlabel("Train MAE")
        axes[i].set_ylabel("Frequency")
        axes[i].set_title(f"{train_df.columns[i]}")
        axes[i].axvline(threshold[i], color='red', linestyle='--', label=f'Threshold: {threshold[i]:.2f}')
        axes[i].legend() 
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()


    # out sample
    y_out_sample_pred, y_out_sample = cnn_predict(model = model,data = test_df, seq_n = seq_n)
    out_sample_mae = np.mean(np.abs(y_out_sample_pred - y_out_sample), axis=1)
    # print(y_out_sample_pred.shape)
    # print(y_out_sample.shape)
    # plt.plot(y_out_sample_pred[1000][:,8], label = 'pred')
    # plt.plot(y_out_sample[1000][:,8], label = 'true')
    # plt.legend()
    # plt.show()

    # # Detect out sample which are anomalies.
    anomalies_out_sample = out_sample_mae > threshold
    print("Number of anomaly samples in sample: ", np.sum(anomalies_out_sample))

    # # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_out_sample = {}
    for i in range(input_dim):
        sector_anomalies = anomalies_out_sample[:, i]
        column_name = test_df.columns[i]
        if column_name not in anomalous_data_out_sample:
            anomalous_data_out_sample[column_name] = []
        for data_idx in range(seq_n - 1, len(test_df) - seq_n + 1):
            if np.all(sector_anomalies[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
                anomalous_data_out_sample[column_name].append(data_idx)

   
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), sharex=True) 
    axes = axes.flatten()

    for i, column_name in enumerate(test_df.columns):
        axes[i].plot(test_df[column_name], label="Return", color="blue", linewidth=0.5) 
        anomalous_indices = anomalous_data_out_sample.get(column_name, [])
        axes[i].scatter(anomalous_indices, test_df.iloc[anomalous_indices][column_name], color="red", label="Anomalous Return", s=10)
        
        axes[i].set_title(f"Sector: {column_name}")
        axes[i].set_ylabel("Return")
        axes[i].legend()
        
        axes[i].set_xticks(np.arange(0, len(test_df), 200))
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(test_df.columns), 12):
        fig.delaxes(axes[j])
    plt.suptitle('CNN Autoencoder out sample')
    plt.tight_layout()
    plt.show()







