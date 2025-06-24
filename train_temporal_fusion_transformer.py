import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from temporal_fusion_transformer import MyModel  
# --- Define quantile loss ---
def quantile_loss(y_pred, y_true, q=0.5):
    error = y_true - y_pred
    return torch.max((q - 1) * error, q * error).mean()

# --- Load data ---
df = pd.read_csv("AXISBANK.csv")
cols = ['Open', 'High', 'Low', 'Prev Close', 'Last', 'VWAP']
target_col = 'Close'

full_X = torch.tensor(df[cols].values, dtype=torch.float32)
full_y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(-1)

# --- Hyperparameters ---
k = 2
T = 2
q = 0.5
learning_rate = 0.1
epoch_losses = []

# --- Visualization helpers ---
def run_prediction(model, full_X, full_y, k, T):
    true_vals = []
    pred_vals = []
    model.eval()
    with torch.no_grad():
        for i in range(20 - (k + T) + 1):
            window = full_X[i:i+k+T]
            window_y = full_y[i:i+k+T]
            x_k = window[:k].unsqueeze(0)
            x_T = window[k:k+T].unsqueeze(0)
            y_T = window_y[k:k+T].unsqueeze(0)
            y_pred = model(x_k, x_T)
            true_vals.extend(y_T.squeeze().tolist())
            pred_vals.extend(y_pred.squeeze().tolist())
    return true_vals, pred_vals

def plot_predictions(epoch, true_vals, pred_vals):
    plt.figure(figsize=(10, 4))
    plt.plot(true_vals, label='True', marker='o')
    plt.plot(pred_vals, label='Predicted', marker='x')
    plt.title(f"Epoch {epoch} - True vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Initialize model and optimizer ---
model = MyModel(num_vars_k=6, num_vars_T=6)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training loop ---
for epoch in range(101):
    total_loss = 0.0
    model.train()
    for i in range(20 - (k + T) + 1):
        window = full_X[i:i+k+T]
        window_y = full_y[i:i+k+T]
        x_k = window[:k].unsqueeze(0)
        x_T = window[k:k+T].unsqueeze(0)
        y_T = window_y[k:k+T].unsqueeze(0)

        y_pred = model(x_k, x_T)
        loss = quantile_loss(y_pred, y_T, q=q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)

    # Print and plot at selected epochs
    if epoch in [0, 15, 30, 100]:
        print(f"Epoch {epoch} - Loss: {total_loss:.4f}")
        true_vals, pred_vals = run_prediction(model, full_X, full_y, k, T)
        plot_predictions(epoch, true_vals, pred_vals)
        
