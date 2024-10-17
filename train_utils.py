import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


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