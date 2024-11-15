import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from cnn.cnn1d import ConvAutoencoder,data_to_tensor
from dynamic_thresholds import compute_rolling_vol, calc_vol_thershold, vol_regimes, compute_dynamic_thresholds, detect_anomalies_sample, detect_anomalies_index
import matplotlib.pyplot as plt 

def cnn_predict(model: nn.Module ,data: pd.DataFrame ,seq_n: int):
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
    print(f"train_df shape: {train_df.shape}; valid_df shape: {valid_df.shape};test_df shape: {test_df.shape};")


    input_dim = train_df.shape[1]
    seq_n = 20
    model_path = 'models_repo/2024_11_06_cnn1d_channel.pth'
    model = ConvAutoencoder(in_channels = input_dim, 
                            hidden_channels1 = 64,
                            activation_func=nn.Tanh(),
                            kernel_size = 3,
                            stride = 2).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    y_val_pred, y_val = cnn_predict(model = model,data = valid_df, seq_n = seq_n)
    mae = np.mean(np.abs(y_val_pred - y_val), axis=1)
    mae_df = pd.DataFrame(mae, index=valid_df.shift(seq_n-1).dropna().index, columns=valid_df.columns)

    vol_df = compute_rolling_vol(data = valid_df, window_size=seq_n)
    insample_vol_threshold = calc_vol_thershold(vol_df=vol_df, low=0.3, high=0.7)
    insampe_regimes = vol_regimes(vol_df=vol_df, vol_thershold=insample_vol_threshold)
    dynamic_thresholds = compute_dynamic_thresholds(scores_df = mae_df, regimes_df = insampe_regimes, score_threshold=0.95)
    dynamic_thresholds_df = pd.DataFrame.from_dict(dynamic_thresholds, orient='index').T
    print(dynamic_thresholds_df)


    y_out_sample_pred, y_out_sample = cnn_predict(model = model,data = test_df, seq_n = seq_n)
    out_sample_mae = np.mean(np.abs(y_out_sample_pred - y_out_sample), axis=1)
    out_sample_mae_df = pd.DataFrame(out_sample_mae, index=test_df.shift(seq_n-1).dropna().index, columns=test_df.columns)

    outsample_vol_df = compute_rolling_vol(test_df, window_size=seq_n)
    outsampe_regimes = vol_regimes(vol_df=outsample_vol_df, vol_thershold=insample_vol_threshold)

    anomalies_sample = detect_anomalies_sample(scores_df = out_sample_mae_df, regimes_df = outsampe_regimes, dynamic_thresholds=dynamic_thresholds)
    anomalies_index = detect_anomalies_index(data=test_df, anomalies_df = anomalies_sample, seq_n=20, weight_on_seq=0.75)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8), sharex=True) 
    axes = axes.flatten()
    for i, column_name in enumerate(test_df.columns):
        axes[i].plot(test_df[column_name], label="Return", color="blue", linewidth=0.5) 
        anomalous_indices = anomalies_index.get(column_name, [])
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
    



    





