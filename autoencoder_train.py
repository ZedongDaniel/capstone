from stock_data import StockData
from autoencoder_model import autoencoder_dataset, lstm_autoencoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


def equal_index_construction(data: pd.DataFrame):
    def log_ret(s:pd.Series):
        return np.log(s / s.shift(1))
    
    data['log_ret'] = data.groupby('ticker', group_keys=False)['adjusted_close'].apply(log_ret)
    df = data.dropna().copy()
    df['w'] = df.groupby('date', group_keys=False)['adjusted_close'].transform(lambda x : 1 / len(x))
    index = df.groupby('date').apply(lambda x: x['log_ret']@x['w'])
    return index

def train(model, batch_size, train_dataset, valid_dataset, num_epochs, lr, loss_function, early_stop, patience, verbose):
    def init_weights(model):
        for name, param in model.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)

    init_weights(model)
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    train_loss = []
    valid_loss = []
    best_valid_loss = float("inf")
    curr_patience = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss_batch, valid_loss_batch = [], []

        for inputs, outputs in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, outputs)
            train_loss_batch.append(loss.item())
            loss.backward()
            optimizer.step()

        mean_train_loss = sum(train_loss_batch) / len(train_loss_batch)
        train_loss.append(mean_train_loss)

        if verbose:
            print(f"epoch [{epoch + 1}/{num_epochs}]: \ttraining loss: {mean_train_loss:.4f}", end="")

        model.eval()
        with torch.no_grad():
            for inputs, outputs in valid_dataloader:
                y_pred = model(inputs)
                loss = criterion(y_pred, outputs)
                valid_loss_batch.append(loss.item())

            mean_valid_loss = sum(valid_loss_batch) / len(valid_loss_batch)
            valid_loss.append(mean_valid_loss)

        if verbose:
            print(f"\tvalidation Loss: {mean_valid_loss:.4f}")

        if early_stop:
            if mean_valid_loss < best_valid_loss:
                best_valid_loss = mean_valid_loss
                curr_patience = 0
            else:
                curr_patience += 1

            if curr_patience >= patience:
                print(f"\nEarly stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
                break

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(valid_loss, label='Validation Loss')
        plt.title("Training and Validation Losses Over Epochs")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    return None


# stock = StockData('sp_400_midcap.csv', '662166cb8e3d13.57537943')
# df = stock.fetch_all_stocks(period = 'd', start = '2000-01-01', end = '2024-8-30')
# mid_cap_index = equal_index_construction(df)
# mid_cap_index.to_csv('mid_cap_index.csv')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

mid_cap_index = pd.read_csv('index_data/mid_cap_index.csv', index_col='date')
n = int(len(mid_cap_index) * 0.8)
train_n = int(n * 0.8)
train_df = mid_cap_index.iloc[:train_n]
valid_df = mid_cap_index.iloc[train_n:]

seq_n = 100
train_dataset = autoencoder_dataset(train_df, seq_n)
valid_dataset = autoencoder_dataset(valid_df, seq_n)

model = lstm_autoencoder(input_size = 1, seq_n = seq_n)
model.to(device)

train(
    model = model,
    batch_size = 32,
    train_dataset = train_dataset,
    valid_dataset = valid_dataset,
    num_epochs= 100,
    lr = 1e-5,
    loss_function = nn.L1Loss(),
    early_stop = True,
    patience = 5,
    verbose = True
)

model_path = 'model/autoencoder_2024_10_07.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")







            


    