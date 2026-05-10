# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:19:46 2026

@author: Dashka
"""

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
ENVIRONMENT CHECK AND LIBRARY SETUP
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive
import os
import time

print("="*80)
print("ENVIRONMENT CHECK")
print("="*80)

# GPU check
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f" GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device('cpu')
    print("⚠️ GPU not detected, using CPU")

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print("="*80)

# -*- coding: utf-8 -*-
"""
DATA LOADING
"""

print("="*80)
print("DATA LOADING")
print("="*80)

# Option 1: if the file was uploaded directly to Colab
if os.path.exists('multilayer_data2.csv'):
    print("📂 Loading from the local Colab folder...")
    data = pd.read_csv('multilayer_data2.csv')

# Option 2: if you are using Google Drive
elif os.path.exists('/content/drive/MyDrive/multilayer_data.csv'):
    print("📂 Loading from Google Drive...")
    data = pd.read_csv('/content/drive/MyDrive/multilayer_data.csv')

else:
    print("❌ File not found!")
    print("Upload multilayer_data2.csv:")
    print(" 1. Click the 📁 icon on the left")
    print(" 2. Click 📤 (upload)")
    print(" 3. Select the file")
    raise FileNotFoundError("multilayer_data2.csv not found")

print(f"✅ Loaded {len(data):,} records")
print(f"Columns: {list(data.columns)}")
print("="*80)

# -*- coding: utf-8 -*-
"""
DATA PREPARATION
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("="*80)
print("DATA PREPARATION")
print("="*80)

# Split into X and y
X = data[['lambda_nm', 'mat1_idx', 'd1_nm', 'mat2_idx', 'd2_nm']].values
y = data[['R_TE', 'R_TM', 'T_TE', 'T_TM']].values

print(f"Original X shape: {X.shape}")
print(f"Original y shape: {y.shape}")

# One-Hot Encoding for materials
def encode_materials(X):
    n_samples = X.shape[0]
    X_encoded = np.zeros((n_samples, 9))

    for i in range(n_samples):
        X_encoded[i, 0] = X[i, 0]  # lambda
        mat1_idx = int(X[i, 1])
        X_encoded[i, 1 + mat1_idx] = 1.0  # one-hot mat1
        X_encoded[i, 4] = X[i, 2]  # d1
        mat2_idx = int(X[i, 3])
        X_encoded[i, 5 + mat2_idx] = 1.0  # one-hot mat2
        X_encoded[i, 8] = X[i, 4]  # d2

    return X_encoded

print("\n🔢 One-Hot Encoding of materials...")
X_encoded = encode_materials(X)
print(f"✅ After encoding: {X_encoded.shape}")

# Normalization
print("\n📏 Normalizing data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X_encoded)
y_normalized = scaler_y.fit_transform(y)

print("✅ Data normalized")

# Train/test split
print("\n✂️ Splitting into train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_normalized,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"✅ Train: {len(X_train):,} samples")
print(f"✅ Test: {len(X_test):,} samples")

# Save scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("\n💾 Scalers saved: scaler_X.pkl, scaler_y.pkl")
print("="*80)

# -*- coding: utf-8 -*-
"""
DATALOADER CREATION
"""

import torch
from torch.utils.data import Dataset, DataLoader

print("="*80)
print("DATALOADER CREATION")
print("="*80)

class OpticalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Parameters
BATCH_SIZE = 256  # Increased from 64 to 256 (faster, fewer batches)
NUM_WORKERS = 2   # Parallel data loading

# Create datasets
train_dataset = OpticalDataset(X_train, y_train)
test_dataset = OpticalDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == 'cuda' else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True if device.type == 'cuda' else False
)

print(f"✅ Train DataLoader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f"✅ Test DataLoader: {len(test_loader)} batches")
print(f" Batch size: {BATCH_SIZE}")
print(f" Workers: {NUM_WORKERS}")
print("="*80)

# -*- coding: utf-8 -*-
"""
NEURAL NETWORK CREATION
"""

import torch.nn as nn
import torch.optim as optim

print("="*80)
print("NEURAL NETWORK ARCHITECTURE CREATION")
print("="*80)

class OpticalMLP(nn.Module):
    def __init__(self, input_size=9, output_size=4):
        super(OpticalMLP, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: 9 → 128
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Normalization between layers
            nn.Dropout(0.2),      # Dropout against overfitting

            # Layer 2: 128 → 256
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),

            # Layer 4: 128 → 64
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer: 64 → 4
            nn.Linear(64, output_size)
            # No activation on the output (regression)
        )

    def forward(self, x):
        return self.network(x)

# Create model
model = OpticalMLP(input_size=9, output_size=4)
model = model.to(device)  # Move to GPU/CPU

print("✅ Model architecture:")
print(model)
print(f"\n📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f" Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("="*80)

# -*- coding: utf-8 -*-
"""
OPTIMIZER AND LOSS FUNCTION SETUP
"""

print("="*80)
print("TRAINING SETUP")
print("="*80)

# Loss function (MSE for regression)
criterion = nn.MSELoss()

# Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler (reduces lr if loss stops decreasing)
# verbose removed because it is not supported in PyTorch 2.x
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

print(f"✅ Loss function: MSELoss")
print(f"✅ Optimizer: Adam (lr=0.001)")
print(f"✅ Scheduler: ReduceLROnPlateau (without verbose)")
print("="*80)

# -*- coding: utf-8 -*-
"""
TRAINING AND EVALUATION FUNCTIONS
"""

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
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

print("✅ Training functions are ready")

# -*- coding: utf-8 -*-
"""
TRAINING LOOP
"""

print("="*80)
print("START OF TRAINING")
print("="*80)

EPOCHS = 10  # Number of epochs
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

    # Save the best model
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_loss': test_loss,
        }, 'best_model.pth')
        print(f" 💾 Best model saved!")

    # History
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)

    # Progress
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.6f} | "
          f"Test Loss: {test_loss:.6f} | "
          f"Time: {epoch_time:.1f}s")

total_time = time.time() - start_time
print("\n" + "="*80)
print(f"✅ TRAINING COMPLETED!")
print(f" Total time: {total_time/60:.1f} minutes")
print(f" Best Test Loss: {best_test_loss:.6f}")
print("="*80)

# -*- coding: utf-8 -*-
"""
RESULT VISUALIZATION
"""

print("="*80)
print("VISUALIZING THE TRAINING PROCESS")
print("="*80)

# Loss plot
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['test_loss'], label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Process')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('training_history.png', dpi=300)
print("✅ Plot saved: training_history.png")
plt.show()

# Final evaluation on the test set
final_test_loss = evaluate(model, test_loader, criterion, device)
print(f"\n📊 Final Test Loss: {final_test_loss:.6f}")
print("="*80)

# -*- coding: utf-8 -*-
"""
SAVING THE MODEL AND SCALERS
"""

print("="*80)
print("SAVING RESULTS")
print("="*80)

# Mount Google Drive (if not mounted yet)
try:
    drive.mount('/content/drive', force_remount=False)
    drive_path = '/content/drive/MyDrive/Optical_Neural_Network/'

    # Create the folder
    import os
    os.makedirs(drive_path, exist_ok=True)

    # Save files
    torch.save(model.state_dict(), drive_path + 'optical_model.pth')
    joblib.dump(scaler_X, drive_path + 'scaler_X.pkl')
    joblib.dump(scaler_y, drive_path + 'scaler_y.pkl')

    # Copy training history artifacts
    import shutil
    shutil.copy('best_model.pth', drive_path + 'best_model.pth')
    shutil.copy('training_history.png', drive_path + 'training_history.png')

    print(f"✅ All files saved to Google Drive:")
    print(f" 📁 {drive_path}")
    print(f" - optical_model.pth (model weights)")
    print(f" - best_model.pth (best model)")
    print(f" - scaler_X.pkl (input scaler)")
    print(f" - scaler_y.pkl (output scaler)")
    print(f" - training_history.png (plot)")

except Exception as e:
    print(f"⚠️ Error while saving to Drive: {e}")
    print("Files were saved locally in Colab")

print("="*80)

# -*- coding: utf-8 -*-
"""
MODEL TESTING
"""

print("="*80)
print("TEST PREDICTION")
print("="*80)

# Load best model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Take random examples from the test set
model.eval()
with torch.no_grad():
    # Take the first batch
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    # Prediction
    predictions = model(X_batch)

# Denormalization
predictions_denorm = scaler_y.inverse_transform(predictions.cpu().numpy())
y_true_denorm = scaler_y.inverse_transform(y_batch.cpu().numpy())

# Print first 5 examples
print("\nPrediction examples (first 5 from the batch):")
print("-" * 80)
print(f"{'R_TE':>10} {'R_TM':>10} {'T_TE':>10} {'T_TM':>10}")
print("-" * 80)

for i in range(5):
    print(f"Predicted: {predictions_denorm[i, 0]:>10.6f} "
          f"{predictions_denorm[i, 1]:>10.6f} "
          f"{predictions_denorm[i, 2]:>10.6f} "
          f"{predictions_denorm[i, 3]:>10.6f}")
    print(f"Actual:    {y_true_denorm[i, 0]:>10.6f} "
          f"{y_true_denorm[i, 1]:>10.6f} "
          f"{y_true_denorm[i, 2]:>10.6f} "
          f"{y_true_denorm[i, 3]:>10.6f}")
    print("-" * 80)

# Mean error
mae = np.mean(np.abs(predictions_denorm - y_true_denorm))
print(f"\n📊 Mean Absolute Error (MAE): {mae:.6f}")
print("="*80)