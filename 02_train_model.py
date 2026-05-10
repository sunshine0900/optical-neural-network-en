# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:19:12 2026

@author: Dashka
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:36:59 2026

@author: Dashka
"""

"""
STEP 2: NEURAL NETWORK TRAINING
Creates the architecture, trains the model, saves the best model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import os

print("="*80)
print("STEP 2: NEURAL NETWORK TRAINING")
print("="*80)

# === DEVICE CHECK ===
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print(" GPU not detected, using CPU")
    print(" This will take 1-2 hours, but the result will be the same!")

print(f"PyTorch version: {torch.__version__}")
print("="*80)

# === LOAD DATA (from previous step) ===
print("\n Loading prepared data...")

# Check if required files exist
for f in ['scaler_X.pkl', 'scaler_y.pkl']:
    if not os.path.exists(f):
        print(f"File '{f}' not found")
        print(" Run 01_prepare_data.py first")
        raise FileNotFoundError(f)

# Load scalers (for inverse normalization during testing)
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load source data for splitting
import pandas as pd
data = pd.read_csv('multilayer_data2.csv')

X = data[['lambda_nm', 'mat1_idx', 'd1_nm', 'mat2_idx', 'd2_nm']].values
y = data[['R_TE', 'R_TM', 'T_TE', 'T_TM']].values

# One-Hot Encoding (same function)
def encode_materials(X):
    n_samples = X.shape[0]
    X_encoded = np.zeros((n_samples, 9))
    for i in range(n_samples):
        X_encoded[i, 0] = X[i, 0]
        mat1_idx = int(X[i, 1])
        X_encoded[i, 1 + mat1_idx] = 1.0
        X_encoded[i, 4] = X[i, 2]
        mat2_idx = int(X[i, 3])
        X_encoded[i, 5 + mat2_idx] = 1.0
        X_encoded[i, 8] = X[i, 4]
    return X_encoded

X_encoded = encode_materials(X)

# Normalization
X_normalized = scaler_X.transform(X_encoded)
y_normalized = scaler_y.transform(y)

# Split (same random_state=42!)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_normalized,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Create DataLoader
class OpticalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

BATCH_SIZE = 512

train_dataset = OpticalDataset(X_train, y_train)
test_dataset = OpticalDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f" Train: {len(X_train):,} samples")
print(f" Test: {len(X_test):,} samples")
print("="*80)

# === CREATE MODEL ===
print("\n Creating neural network architecture...")

class OpticalMLP(nn.Module):
    def __init__(self, input_size=9, output_size=4):
        super(OpticalMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

model = OpticalMLP(input_size=9, output_size=4)
model = model.to(device)

print("Architecture:")
print(model)
print(f"\n Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("="*80)

# === TRAINING SETUP ===
print("\n Configuring optimizer...")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

print(" Loss function: MSELoss")
print(" Optimizer: Adam (lr=0.001)")
print(" Scheduler: ReduceLROnPlateau")
print("="*80)

# === TRAINING FUNCTIONS ===
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / len(loader)

print(" Training functions are ready")
print("="*80)

# === TRAINING LOOP ===
print("\n TRAINING STARTED")
print("="*80)

EPOCHS = 10
best_test_loss = float('inf')
history = {'train_loss': [], 'test_loss': []}

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Training
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

    # Testing
    test_loss = evaluate(model, test_loader, criterion, device)

    # Update learning rate
    scheduler.step(test_loss)

    # Save best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
        }, 'best_model.pth')
        print(f" Best model saved!")

    # History
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)

    # Progress
    epoch_time = time.time() - epoch_start
    remaining = (EPOCHS - epoch - 1) * epoch_time
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train: {train_loss:.6f} | "
          f"Test: {test_loss:.6f} | "
          f"Time: {epoch_time:.1f}s | "
          f"Remaining: {remaining/60:.1f} min")

total_time = time.time() - start_time

print("\n" + "="*80)
print(" TRAINING COMPLETED")
print(f" Total time: {total_time/60:.1f} minutes")
print(f" Best Test Loss: {best_test_loss:.6f}")
print("="*80)

# === VISUALIZATION ===
print("\n Visualizing training process...")

plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['test_loss'], label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_history.png', dpi=300)
print(" Plot saved: training_history.png")
plt.show()

# === TEST PREDICTION ===
print("\n Test prediction...")

# Load best model
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Take random samples
with torch.no_grad():
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    predictions = model(X_batch)

# Denormalization
predictions_denorm = scaler_y.inverse_transform(predictions.cpu().numpy())
y_true_denorm = scaler_y.inverse_transform(y_batch.cpu().numpy())

print("\nExamples (first 5):")
print("-" * 70)
print(f"{'R_TE':>10} {'R_TM':>10} {'T_TE':>10} {'T_TM':>10}")
print("-" * 70)

for i in range(5):
    print(f"Predicted: {predictions_denorm[i, 0]:>10.6f} "
          f"{predictions_denorm[i, 1]:>10.6f} "
          f"{predictions_denorm[i, 2]:>10.6f} "
          f"{predictions_denorm[i, 3]:>10.6f}")
    print(f"Actual:    {y_true_denorm[i, 0]:>10.6f} "
          f"{y_true_denorm[i, 1]:>10.6f} "
          f"{y_true_denorm[i, 2]:>10.6f} "
          f"{y_true_denorm[i, 3]:>10.6f}")
    print("-" * 70)

mae = np.mean(np.abs(predictions_denorm - y_true_denorm))
print(f"\n Mean Absolute Error (MAE): {mae:.6f}")
print("="*80)

# === SUMMARY ===
print("\n" + "="*80)
print(" SAVED FILES:")
print("="*80)
print(" best_model.pth (best model)")
print(" scaler_X.pkl (input scaler)")
print(" scaler_y.pkl (output scaler)")
print(" training_history.png (plot)")
print(" data_distribution.png (distribution)")
print("="*80)

print("\n DONE! Now run predictor.py to use the model")
print("="*80)