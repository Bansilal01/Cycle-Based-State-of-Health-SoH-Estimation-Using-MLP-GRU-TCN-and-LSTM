
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams.update({'font.size': 28, 'font.family': 'Times New Roman'})

def extract_capacity(battery_data):
    cycle_array = battery_data['cycle'][0, 0][0]
    capacities = []
    for cycle in cycle_array:
        if 'type' in cycle.dtype.names:
            cycle_type = cycle['type'][0]
            if isinstance(cycle_type, np.ndarray):
                cycle_type = cycle_type[0]
            if cycle_type != 'discharge':
                capacities.append(np.nan)
                continue
        data = cycle['data'][0, 0]
        I = data['Current_measured'][0]
        t = data['Time'][0]
        if len(I) > 0 and len(t) > 0:
            cap = np.trapz(np.abs(I), t) / 3600
            capacities.append(cap)
        else:
            capacities.append(np.nan)
    return np.array(capacities)

mat = scipy.io.loadmat('B0005.mat')
capacities = extract_capacity(mat['B0005'])
valid_idx = ~np.isnan(capacities)
X = np.where(valid_idx)[0].reshape(-1, 1).astype(np.float32)
X = X / X.max()
Y = capacities[valid_idx].astype(np.float32)
Y_norm = Y / Y[0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_norm, test_size=0.2, shuffle=False)

# Deep learning models
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.relu(self.conv(x))
        res = self.downsample(x) if self.downsample else x
        return out[:, :, :x.size(2)] + res

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            TCNBlock(1, 64, 3, 1),
            TCNBlock(64, 64, 3, 2),
            nn.Conv1d(64, 1, 1)
        )

    def forward(self, x):
        return self.net(x).transpose(1, 2)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x.transpose(1, 2))
        return self.fc(out)

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x.transpose(1, 2))
        return self.fc(out)

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))

def train_model(model, X_train, Y_train, X_all, Y_all):
    model.train()
    X_tensor = torch.tensor(X_train.reshape(1, 1, -1), dtype=torch.float32)
    Y_tensor = torch.tensor(Y_train.reshape(1, -1, 1), dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_history = []

    for epoch in range(3000):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output.view_as(Y_tensor), Y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.tensor(X_all.reshape(1, 1, -1), dtype=torch.float32)).view(-1).numpy()

    rmse = np.sqrt(mean_squared_error(Y_all, Y_pred))
    mae = mean_absolute_error(Y_all, Y_pred)
    r2 = r2_score(Y_all, Y_pred)
    residuals = Y_all - Y_pred
    return Y_pred, rmse, mae, r2, loss_history, residuals

models = {
    'TCN': TCN(),
    'LSTM': LSTMModel(),
    'GRU': GRUModel(),
    'MLP': MLPModel()
}

results = {}
for name, model in models.items():
    pred, rmse, mae, r2, loss, residuals = train_model(model, X_train, Y_train, X, Y_norm)
    results[name] = {'pred': pred, 'rmse': rmse, 'mae': mae, 'r2': r2, 'loss': loss, 'residuals': residuals}
    print(f"{name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Combined comparison plot
plt.figure(figsize=(12, 6))
plt.plot(Y_norm, label='Actual SoH', color='black', linewidth=2)
for name, res in results.items():
    plt.plot(res['pred'], label=f'{name} Prediction', linestyle='--')
plt.title("Comparison of Predicted SoH - All Models")
plt.xlabel("Cycle Number")
plt.ylabel("Normalized Capacity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/prediction_comparison_all_models.png", dpi=600)
plt.close()

# Individual plots per model
for name, res in results.items():
    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(res['loss'], color='teal')
    plt.title(f"{name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_loss_curve.png", dpi=600)
    plt.close()

    # 2. Predicted vs Actual
    plt.figure(figsize=(8, 5))
    plt.plot(Y_norm, label='Actual SoH', color='black', linewidth=2)
    plt.plot(res['pred'], label='Predicted SoH', linestyle='--', color='green')
    plt.title(f"{name} - Predicted vs Actual SoH")
    plt.xlabel("Cycle Number")
    plt.ylabel("Normalized Capacity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_predicted_vs_actual.png", dpi=600)
    plt.close()

    # 3. Residual Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(res['residuals'], bins=25, color='coral', edgecolor='black', alpha=0.75)
    plt.title(f"{name} - Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_residuals_histogram.png", dpi=600)
    plt.close()

    # 4. Actual vs Predicted Scatter
    plt.figure(figsize=(8, 5))
    plt.scatter(Y_norm, res['pred'], alpha=0.6, color='blue', label='Predicted')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal Fit')
    plt.title(f"{name} - Actual vs Predicted Scatter")
    plt.xlabel("Actual SoH")
    plt.ylabel("Predicted SoH")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_actual_vs_pred_scatter.png", dpi=600)
    plt.close()

# Summary Table
summary_df = pd.DataFrame({
    name: {'RMSE': res['rmse'], 'MAE': res['mae'], 'R2': res['r2']}
    for name, res in results.items()
}).T

print("\\nModel Evaluation Summary:")
print(summary_df.round(4))


