import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_csv("simulated_joint_data.csv")
data = df[['q', 'qd', 'qdd', 'tau']].values

# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 构造滑窗
window_size = 20
X, y = [], []
for i in range(len(data_scaled) - window_size):
    X.append(data_scaled[i:i+window_size, :3])
    y.append(data_scaled[i+window_size, 3])
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# 划分训练集测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

class JointDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(JointDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(JointDataset(X_test, y_test), batch_size=32)

# 定义 BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = BiLSTM(3, 64, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(30):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        pred = model(Xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/30, Loss: {total_loss / len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "bilstm_model.pth")

# 测试预测可视化
model.eval()
y_preds, y_trues = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        pred = model(Xb)
        y_preds.append(pred.numpy())
        y_trues.append(yb.numpy())

y_preds = np.vstack(y_preds)
y_trues = np.vstack(y_trues)

plt.figure(figsize=(10,4))
plt.plot(y_trues, label='Actual', linewidth=1)
plt.plot(y_preds, label='Predicted', linewidth=1)
plt.legend()
plt.title("Torque Prediction vs Actual")
plt.xlabel("Sample Index")
plt.ylabel("Torque (Normalized)")
plt.tight_layout()
plt.savefig("prediction_vs_actual.png")
plt.show()
