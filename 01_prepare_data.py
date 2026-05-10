# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:17:58 2026

@author: Dashka
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:35:28 2026

@author: Dashka
"""

"""
STEP 1: DATA PREPARATION
Loads CSV, encodes materials, normalizes data, creates DataLoader
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import os

print("="*80)
print("STEP 1: DATA PREPARATION")
print("="*80)

# === LOAD DATA ===
print("\n Loading data...")

# Check if file exists
if not os.path.exists('multilayer_data2.csv'):
    print("File 'multilayer_data2.csv' not found!")
    print(" Place the file in the same folder as this script.")
    raise FileNotFoundError("multilayer_data2.csv not found")

data = pd.read_csv('multilayer_data2.csv')
print(f" Loaded {len(data):,} records")

# === SPLIT INTO X AND y ===
X = data[['lambda_nm', 'mat1_idx', 'd1_nm', 'mat2_idx', 'd2_nm']].values
y = data[['R_TE', 'R_TM', 'T_TE', 'T_TM']].values

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# === ONE-HOT ENCODING ===
print("\n One-Hot Encoding of materials...")

def encode_materials(X):
    n_samples = X.shape[0]
    X_encoded = np.zeros((n_samples, 9))

    for i in range(n_samples):
        X_encoded[i, 0] = X[i, 0]  # lambda
        mat1_idx = int(X[i, 1])
        X_encoded[i, 1 + mat1_idx] = 1.0
        X_encoded[i, 4] = X[i, 2]  # d1
        mat2_idx = int(X[i, 3])
        X_encoded[i, 5 + mat2_idx] = 1.0
        X_encoded[i, 8] = X[i, 4]  # d2

    return X_encoded

X_encoded = encode_materials(X)
print(f" After encoding: {X_encoded.shape}")

# === NORMALIZATION ===
print("\n Normalizing data...")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X_encoded)
y_normalized = scaler_y.fit_transform(y)

print(" Data normalized")

# === TRAIN/TEST SPLIT ===
print("\n Splitting into train/test (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_normalized,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f" Train: {len(X_train):,} samples")
print(f" Test: {len(X_test):,} samples")

# === CREATE DATASET AND DATALOADER ===
print("\n Creating DataLoader...")

class OpticalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# For CPU, increase batch_size to 64
BATCH_SIZE = 64
NUM_WORKERS = 0  # Better to use 0 on Windows

train_dataset = OpticalDataset(X_train, y_train)
test_dataset = OpticalDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(f" Train DataLoader: {len(train_loader)} batches of {BATCH_SIZE}")
print(f" Test DataLoader: {len(test_loader)} batches")

# === SAVE SCALERS ===
print("\n Saving scalers...")

joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

print("Scalers saved: scaler_X.pkl, scaler_y.pkl")

# === VISUALIZATION ===
print("\n Visualizing data distribution...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

target_names = ['R_TE', 'R_TM', 'T_TE', 'T_TM']

for i, name in enumerate(target_names):
    axes[i].hist(y[:, i], bins=50, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'Distribution of {name}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300)
print(" Plot saved: data_distribution.png")
plt.show()

# === STATISTICS ===
print("\n" + "="*80)
print("DATA PREPARATION SUMMARY:")
print("="*80)
print(f"Total samples: {len(data):,}")
print(f"Train: {len(X_train):,} ({len(X_train)/len(data)*100:.1f}%)")
print(f"Test: {len(X_test):,} ({len(X_test)/len(data)*100:.1f}%)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Batches per epoch: {len(train_loader)}")
print("="*80)

print("\n DATA READY! Run the next script: 02_train_model.py")
print("="*80)