# Cell 1: Imports and Utility Functions
import time
import psutil
import GPUtil
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def get_hardware_stats(start_time):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 ** 2)
    vms = process.memory_info().vms / (1024 ** 2)
    duration = time.time() - start_time

    try:
        gpu = GPUtil.getGPUs()[0]
        gpu_util = gpu.load * 100
        gpu_mem = gpu.memoryUsed
    except:
        gpu_util, gpu_mem = 0, 0

    return cpu, rss, vms, duration, gpu_util, gpu_mem


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    loss = np.mean((y_true - y_pred) ** 2)
    return rmse, mae, loss

# Cell 2: Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=72, hidden_size=64, output_size=24, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, seq_len=1, input_size)
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # Take the last timestep
        return out  # (batch_size, output_size)

# Cell 3: Load and Prepare Data
file_path = "D:/MinhKhoi/SIU/DoAnTotNghiep/Project/data/aqi_station_district_11.csv"
df = pd.read_csv(file_path)

X = df[[f"AQI_t-{i}" for i in range(71, 0, -1)] + ['AQI_t0']].values
Y = df[[f"AQI_t+{i}" for i in range(1, 25)]].values

valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
X, Y = X[valid_idx], Y[valid_idx]

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, Y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Cell 4: Train One Model for All 24 Outputs
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

    print("Epoch",epoch)

cpu, rss, vms, duration, gpu_util, gpu_mem = get_hardware_stats(start_time)

# Save model
torch.save(model.state_dict(), f"models/lstm_model_district_11.pt")

# Cell 5: Evaluate and Save Metrics for Each of 24 Outputs
model.eval()
with torch.no_grad():
    pred_train = model(X_train_tensor).numpy()
    pred_val = model(X_val_tensor).numpy()
    pred_test = model(X_test_tensor).numpy()

metrics_records = []
for i in range(24):
    target_name = f"AQI_t+{i+1}"
    y_train_i = y_train[:, i]
    y_val_i = y_val[:, i]
    y_test_i = y_test[:, i]
    pred_train_i = pred_train[:, i]
    pred_val_i = pred_val[:, i]
    pred_test_i = pred_test[:, i]

    rmse_tr, mae_tr, loss_tr = compute_metrics(y_train_i, pred_train_i)
    rmse_va, mae_va, loss_va = compute_metrics(y_val_i, pred_val_i)
    rmse_te, mae_te, loss_te = compute_metrics(y_test_i, pred_test_i)

    metrics_records.append([target_name, rmse_tr, mae_tr, loss_tr,
                            rmse_va, mae_va, loss_va,
                            rmse_te, mae_te, loss_te])

metrics_df = pd.DataFrame(metrics_records, columns=[
    "target", "rmse_train", "mae_train", "loss_train",
    "rmse_val", "mae_val", "loss_val",
    "rmse_test", "mae_test", "loss_test"
])

hardware_df = pd.DataFrame([["AQI_t+1 to t+24", cpu, rss, vms, duration, gpu_util, gpu_mem]], columns=[
    "target", "cpu_percent", "ram_rss_mb", "ram_vms_mb",
    "epoch_duration_seconds", "gpu_util_percent", "gpu_mem_used_mb"
])

metrics_df.to_csv("metrics/lstm_metrics_district_11_3.csv", index=False)
hardware_df.to_csv("metrics/lstm_hardware_district_11_3.csv", index=False)
print(file_path)