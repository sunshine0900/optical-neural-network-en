
# 🔬 Optical Neural Network — Prediction of Reflection and Transmission Coefficients

> A neural network for predicting the optical properties of two-layer thin-film coatings

---

## 📌 Problem Description

The task is to predict the reflection and transmission coefficients for TE and TM polarizations from the parameters of a two-layer optical coating (materials + thicknesses + wavelength):

| Output | Description |
|-------|----------|
| `RTE` | Reflection coefficient, TE polarization |
| `RTM` | Reflection coefficient, TM polarization |
| `TTE` | Transmission coefficient, TE polarization |
| `TTM` | Transmission coefficient, TM polarization |

The “exact” values are calculated using the **Transfer Matrix Method (TMM)**, and the neural network is trained to approximate this mapping.

---

## 🧱 Model Architecture

Multilayer Perceptron (`OpticalMLP`):

```text
Input (9) → Linear(128) → ReLU → BN → Dropout(0.2)
         → Linear(256) → ReLU → BN → Dropout(0.2)
         → Linear(128) → ReLU → BN → Dropout(0.1)
         → Linear(64)  → ReLU
         → Linear(4)   → Output
```

- **Optimizer:** Adam (lr=0.001, weight_decay=1e-5)
- **Loss function:** MSELoss
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Training epochs:** 10

---

## 📊 Training Results

![Training History](training_history.jpg)

Train Loss decreased from ~0.071 to ~0.019, while Test Loss stabilized around ~0.007 — the model generalizes well and shows no signs of overfitting.

---

## 🗂️ Project Structure

```text
optical-neural-network/
│
├── 01_prepare_data.py       # CSV loading, One-Hot Encoding, normalization, DataLoader
├── 02_train_model.py        # MLP training, saving the best model
├── tsikl-po-vsem.py         # Dataset generation using the Transfer Matrix Method (TMM)
├── neironka.py              # Full Colab version (all-in-one file)
├── neironka.ipynb           # Jupyter Notebook version
│
├── training_history.jpg     # Training plot (Train / Test Loss)
│
├── .gitignore
└── README.md
```

> **Note:** Data files (`multilayerdata2.csv`, ~1 GB) and model weights (`best_model.pth`) are not included in the repository — they are generated when running the scripts (see below).

---

## Run

### 1. Install dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib joblib scipy
```

### 2. Generate the dataset

```bash
python tsikl-po-vsem.py
```

Creates the file `multilayerdata2.csv`  
The calculation takes several hours because it iterates over all combinations of materials × thicknesses × wavelengths.

### 3. Prepare the data

```bash
python 01_prepare_data.py
```

Saves `scalerX.pkl`, `scalery.pkl` and `data_distribution.png`.

### 4. Train the model

```bash
python 02_train_model.py
```

Saves `best_model.pth` and `training_history.png`.

---

## Input Features

| Feature | Description |
|---------|----------|
| `lambda_nm` | Wavelength (400–700 nm) |
| `mat1_idx` + one-hot | Material 1 (SiO₂, TiO₂, MgF₂) |
| `d1_nm` | Layer 1 thickness (50–200 nm) |
| `mat2_idx` + one-hot | Material 2 (SiO₂, TiO₂, MgF₂) |
| `d2_nm` | Layer 2 thickness (50–200 nm) |

Total: **9 normalized features** after One-Hot Encoding.

---

## References

1. A. Jentzen, B. Kuckuck, P. von Wurstemberger — *Mathematical Introduction to Deep Learning* (arXiv:2310.20360)
2. G. Carleo et al. — *Machine learning and the physical sciences*, Rev. Mod. Phys. 91, 045002 (2019)
3. Born M., Wolf E. — *Principles of Optics*, Moscow: Nauka, 2003
4. Stratton J. — *Electromagnetic Theory*, Moscow: GIIL, 1973

---

## 👤 Author

**Darya Sinkevich**