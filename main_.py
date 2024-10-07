import numpy as np
import pandas as pd
import torch 
from autoencoder_model import autoencoder_dataset, lstm_autoencoder
import matplotlib.pyplot as plt 
def data_to_tensor(data, dtype = torch.float32):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.tensor(np.array(data), dtype=dtype).to(device)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


mid_cap_index = pd.read_csv('index_data/mid_cap_index.csv', index_col='date')
n = int(len(mid_cap_index) * 0.8)
train_n = int(n * 0.8)
train_df = mid_cap_index.iloc[:train_n]
valid_df = mid_cap_index.iloc[train_n:]
seq_n = 100

model_path = 'model/autoencoder_2024_10_07.pth'
model = lstm_autoencoder(input_size=1, seq_n=seq_n).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))

sample_index = train_df.shift(seq_n-1).dropna().index.tolist()
data_list = []
for sample in sample_index:
    data_list.append(data_to_tensor(train_df.loc[:sample].iloc[-seq_n:]))

y_train_pred = []
for X_i in data_list:
    with torch.no_grad():
        y_i = model(X_i.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
    y_train_pred.append(y_i)

plt.plot(data_list[1000].cpu().numpy(), label = 'y_true')
plt.plot(y_train_pred[1000], label = 'y_pred')
plt.legend()
plt.show()

print(y_train_pred[1000])






