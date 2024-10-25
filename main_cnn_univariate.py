import numpy as np
import pandas as pd
import torch 
from cnn.cnn_sectors_ret import ConvAutoencoder,data_to_tensor
import matplotlib.pyplot as plt 

def cnn_predict(model,data,seq_n):
    sample_index = data.shift(seq_n - 1).dropna().index.tolist()
    data_list = []
    for sample in sample_index:
        data_list.append(data_to_tensor(data.loc[:sample].iloc[-seq_n:].T))
        
    y_pred = []
    for X_i in data_list:
        with torch.no_grad():
            y_i = model(X_i).detach().cpu().numpy().reshape(-1)
            y_pred.append(y_i)
    y_true = np.array([x.cpu().numpy().reshape(-1) for x in data_list])
    return y_pred, y_true

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    mid_cap_index = pd.read_csv('data/mid_cap_index.csv', index_col='date')
    features = ['log_ret']
    ret = mid_cap_index[features] * 100
    n = int(len(ret) * 0.8)
    train_n = int(n * 0.95)
    tmp = ret.iloc[:n]
    train_df = tmp.iloc[:train_n]
    valid_df = tmp.iloc[train_n:]
    test_df = ret.iloc[n:]

    seq_n = 100
    model_path = 'models/2024_10_17_cnn1d_univariate.pth'
    model = ConvAutoencoder(in_channels = 1, 
                            hidden_channels1 = 32, 
                            hidden_channels2 = 16,
                            kernel_size = 7,
                            stride = 2,
                            padding = 3, 
                            dropout_prob=0.2).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    y_train_pred, y_train = cnn_predict(model = model,data = train_df,seq_n = seq_n)

    train_mae = np.mean(np.abs(y_train_pred - y_train), axis=1)

    plt.hist(train_mae, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("frequency")
    plt.show()

    threshold = np.quantile(train_mae, 0.80)
    print("Reconstruction error threshold: ", threshold)

    # in sample
    y_in_sample_pred, y_in_sample = cnn_predict(model = model,data = tmp,seq_n = seq_n)
    in_sample_mae = np.mean(np.abs(y_in_sample_pred - y_in_sample), axis=1)

    # Detect in sample which are anomalies.
    anomalies_in_sample = in_sample_mae > threshold
    print("Number of anomaly samples in sample: ", np.sum(anomalies_in_sample))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices_in_sample = []
    for data_idx in range(seq_n - 1, len(tmp) - seq_n + 1):
        if np.all(anomalies_in_sample[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
             anomalous_data_indices_in_sample.append(data_idx)

    plt.figure(figsize=(10, 6))
    plt.plot(tmp, label="return", color="blue",linewidth=0.5)
    plt.scatter(anomalous_data_indices_in_sample, tmp.iloc[anomalous_data_indices_in_sample], color="red", label="anomalous return")
    plt.xticks(np.arange(0, len(tmp), 200), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # out sample
    y_out_sample_pred, y_out_sample = cnn_predict(model = model,data = test_df, seq_n = seq_n)
    out_sample_mae = np.mean(np.abs(y_out_sample_pred - y_out_sample), axis=1)

    # Detect out sample which are anomalies.
    anomalies_out_sample = out_sample_mae > threshold
    print("Number of anomaly samples out sample: ", np.sum(anomalies_out_sample))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices_out_sample = []
    for data_idx in range(seq_n - 1, len(test_df) - seq_n + 1):
        if np.all(anomalies_out_sample[data_idx - seq_n + 1 : data_idx]): # if data_idx span seq length, it is anomalous_data_point
             anomalous_data_indices_out_sample.append(data_idx)

    plt.figure(figsize=(10, 6))
    plt.plot(test_df, label="return", color="blue",linewidth=0.5)
    plt.scatter(anomalous_data_indices_out_sample, test_df.iloc[anomalous_data_indices_out_sample], color="red", label="anomalous return")
    plt.xticks(np.arange(0, len(test_df), 200), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


