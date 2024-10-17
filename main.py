import numpy as np
import pandas as pd
import torch 
from Conv_model import ConvAutoencoder,data_to_tensor
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
    test_df = ret.iloc[n:]


    seq_n = 100

    model_path = 'model/autoencoder_2024_10_15_conv.pth'
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    sample_index_train = train_df.shift(seq_n - 1).dropna().index.tolist()
    data_list_train = []
    for sample in sample_index_train:
        data_list_train.append(data_to_tensor(train_df.loc[:sample].iloc[-seq_n:]).unsqueeze(0).unsqueeze(0))
    y_train_pred = []

    for X_i in data_list_train:
        with torch.no_grad():
            y_i = model(X_i).detach().cpu().numpy().reshape(-1)
            y_train_pred.append(y_i)
    y_train = np.array([x.cpu().numpy().reshape(-1) for x in data_list_train])
    train_mae_loss = np.mean(np.abs(y_train_pred - y_train), axis=1)
    # plt.hist(train_mae_loss, bins=50)
    # plt.xlabel("Train MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    threshold = np.quantile(train_mae_loss, 0.9)
    print("Reconstruction error threshold: ", threshold)

    sample_index_test = test_df.shift(seq_n - 1).dropna().index.tolist()
    data_list_test = []
    for sample in sample_index_test:
        data_list_test.append(data_to_tensor(test_df.loc[:sample].iloc[-seq_n:]).unsqueeze(0).unsqueeze(0))
    y_test_pred = []

    for X_i in data_list_test:
        with torch.no_grad():
            y_i = model(X_i).detach().cpu().numpy().reshape(-1)
            y_test_pred.append(y_i)
    
    y_test = np.array([x.cpu().numpy().reshape(-1) for x in data_list_test])
    test_mae_loss = np.mean(np.abs(y_test_pred - y_test), axis=1)

    # plt.hist(test_mae_loss, bins=50)
    # plt.xlabel("test MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))


    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    for data_idx in range(seq_n - 1, len(test_df) - seq_n + 1):
        if np.all(anomalies[data_idx - seq_n + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    df_subset = test_df.iloc[anomalous_data_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(test_df, label="return", color="blue")
    plt.plot(anomalous_data_indices, test_df.iloc[anomalous_data_indices], color="red", label="anomalous return")
    plt.xticks(np.arange(0, len(test_df), 50), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


