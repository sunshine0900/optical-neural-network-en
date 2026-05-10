# -*- coding: utf-8 -*-
"""
Created on Sun May 10 13:20:56 2026

@author: Dashka
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:23:16 2026

@author: Dashka
"""

# -*- coding: utf-8 -*-
"""
Data generation for neural network training:
Spectral coefficients for multilayer structures
Structure: air → layer 1 → layer 2 → substrate (air)
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time

# === Physical parameters ===
n_air = 1.0   # input medium — air
n_sub = 1.0   # substrate — air
eps_air = n_air**2
eps_sub = n_sub**2
b = 0.0       # normal incidence

# === Layer materials ===
materials = {
    'SiO2': 1.46,  # silicon dioxide
    'TiO2': 2.40,  # titanium dioxide
    'MgF2': 1.38   # magnesium fluoride
}

# === Parameter ranges ===
wavelengths_nm = np.arange(400, 701, 1)   # 400–700 nm, step 1 nm (301 points)
thicknesses_nm = np.arange(50, 201, 5)    # 50–200 nm, step 5 nm (31 points)

# === Functions ===

def build_M(eps, b=0.0):
    M = np.zeros((4, 4), dtype=complex)
    M[0, 2] = eps - b**2
    M[1, 3] = eps
    M[2, 0] = 1.0
    M[3, 1] = (eps - b**2) / eps if eps != 0 else 0.0
    return M

def layer_matrix(eps, d, k0, b=0.0):
    M = build_M(eps, b)
    Omega = expm(1j * k0 * M * d)
    return Omega

def surface_impedance(eps, b=0.0):
    val = np.sqrt(eps - b**2 + 0j)
    gamma = np.array([
        [1.0 / val, 0.0],
        [0.0, val / eps]
    ], dtype=complex)
    return gamma

def block_I_gamma(gamma):
    I_gamma = np.zeros((2, 4), dtype=complex)
    I_gamma[0, 0] = 1.0
    I_gamma[1, 1] = 1.0
    I_gamma[0, 2] = gamma[0, 0]
    I_gamma[1, 3] = gamma[1, 1]
    return I_gamma

def reflection_coefficient(Omega, gamma0, gammaN):
    I2 = np.eye(2, dtype=complex)
    I_plus_g0 = block_I_gamma(gamma0)
    I_minus_g0 = block_I_gamma(-gamma0)
    gamma_minus_I = np.hstack([gammaN, -I2])
    A = gamma_minus_I @ Omega @ I_minus_g0.T
    B = gamma_minus_I @ Omega @ I_plus_g0.T
    r = -np.linalg.solve(A, B)
    return r

def transmission_coefficient(Omega, gamma0, gammaN):
    I2 = np.eye(2, dtype=complex)
    gamma0_I = np.hstack([gamma0, I2])
    I_gammaN = np.hstack([I2, gammaN])
    Omega_inv = np.linalg.inv(Omega)
    A = gamma0_I @ Omega_inv @ I_gammaN.T
    t = 2 * np.linalg.solve(A, gamma0)
    return t

# === Data generation ===
print("="*80)
print("DATA GENERATION FOR NEURAL NETWORK TRAINING")
print("="*80)
print(f"Structure: air → layer 1 → layer 2 → air")
print(f"Materials: {list(materials.keys())}")
print(f"Wavelength range: 400–700 nm ({len(wavelengths_nm)} points)")
print(f"Thickness range: 50–200 nm ({len(thicknesses_nm)} values)")
print(f"Number of layers: 2")
print(f"Total combinations: {len(wavelengths_nm) * len(thicknesses_nm)**2 * len(materials)**2:,}")
print("="*80)

# List to store all data
all_data = []

# Progress counters
total_iterations = len(materials) * len(materials) * len(thicknesses_nm) * len(thicknesses_nm)
current_iteration = 0

start_time = time.time()

# === Loop over material of layer 1 ===
for mat1_name, n1 in materials.items():
    eps1 = n1**2

    # === Loop over material of layer 2 ===
    for mat2_name, n2 in materials.items():
        eps2 = n2**2

        # === Loop over thickness of layer 1 ===
        for d1_nm in thicknesses_nm:
            d1 = d1_nm * 1e-9

            # === Loop over thickness of layer 2 ===
            for d2_nm in thicknesses_nm:
                d2 = d2_nm * 1e-9

                # === Loop over wavelengths ===
                for lam_nm in wavelengths_nm:
                    lam = lam_nm * 1e-9
                    k0 = 2 * np.pi / lam

                    try:
                        # Matrices for each layer
                        Omega1 = layer_matrix(eps1, d1, k0, b)
                        Omega2 = layer_matrix(eps2, d2, k0, b)

                        # Total structure matrix: Ω = Ω2 · Ω1
                        Omega_total = Omega2 @ Omega1

                        # Impedances of external media
                        gamma0 = surface_impedance(eps_air, b)
                        gammaN = surface_impedance(eps_sub, b)

                        # Coefficients
                        r = reflection_coefficient(Omega_total, gamma0, gammaN)
                        t = transmission_coefficient(Omega_total, gamma0, gammaN)

                        # Energy coefficients
                        R_TE = np.abs(r[0, 0])**2
                        R_TM = np.abs(r[1, 1])**2
                        T_TE = (n_air / n_sub) * np.abs(t[0, 0])**2
                        T_TM = (n_air / n_sub) * np.abs(t[1, 1])**2

                        # Save all parameters + results
                        all_data.append([
                            lam_nm,     # wavelength, nm
                            mat1_name,  # material of layer 1
                            d1_nm,      # thickness of layer 1, nm
                            mat2_name,  # material of layer 2
                            d2_nm,      # thickness of layer 2, nm
                            R_TE,       # TE reflection
                            R_TM,       # TM reflection
                            T_TE,       # TE transmission
                            T_TM        # TM transmission
                        ])

                    except Exception as e:
                        print(f"Error during calculation: λ={lam_nm} nm, "
                              f"{mat1_name}({d1_nm}nm)→{mat2_name}({d2_nm}nm)")
                        print(f"Error: {e}")
                        continue

                # Progress
                current_iteration += 1
                if current_iteration % 1000 == 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed / current_iteration * (total_iterations - current_iteration)
                    print(f"Progress: {current_iteration}/{total_iterations} "
                          f"({current_iteration/total_iterations*100:.1f}%) | "
                          f"Remaining: {remaining/60:.1f} min")

end_time = time.time()
print(f"\n Generation completed! Time: {(end_time - start_time)/60:.1f} minutes")

# === Convert to numpy array ===
data_array = np.array(all_data, dtype=object)

print(f"\n Data statistics:")
print(f" Total records: {len(data_array):,}")
print(f" Columns: {data_array.shape[1]}")
print(f"\nData structure:")
print(" [0] λ_nm, [1] material_1, [2] thickness_1_nm, [3] material_2, [4] thickness_2_nm, "
      "[5] R_TE, [6] R_TM, [7] T_TE, [8] T_TM")

# === Example output of first 10 records ===
print("\n" + "="*80)
print("FIRST 10 RECORDS:")
print("="*80)
print(f"{'λ(nm)':>6} {'Mat1':>6} {'d1(nm)':>6} {'Mat2':>6} {'d2(nm)':>6} "
      f"{'R_TE':>10} {'R_TM':>10} {'T_TE':>10} {'T_TM':>10}")
print("-"*80)
for i in range(min(10, len(data_array))):
    row = data_array[i]
    print(f"{row[0]:>6.0f} {row[1]:>6} {row[2]:>6.0f} {row[3]:>6} {row[4]:>6.0f} "
          f"{row[5]:>10.6f} {row[6]:>10.6f} {row[7]:>10.6f} {row[8]:>10.6f}")

# === Save data to CSV ===
print("\n" + "="*80)
print("SAVING DATA...")
print("="*80)

# Convert to numeric format for saving
numeric_data = []
for row in data_array:
    numeric_data.append([
        float(row[0]),                           # λ_nm
        float(list(materials.keys()).index(row[1])),  # material 1 index
        float(row[2]),                           # d1_nm
        float(list(materials.keys()).index(row[3])),  # material 2 index
        float(row[4]),                           # d2_nm
        float(row[5]),                           # R_TE
        float(row[6]),                           # R_TM
        float(row[7]),                           # T_TE
        float(row[8])                            # T_TM
    ])

numeric_array = np.array(numeric_data)

np.savetxt('multilayer_data2.csv',
           numeric_array,
           delimiter=',',
           header='lambda_nm,mat1_idx,d1_nm,mat2_idx,d2_nm,R_TE,R_TM,T_TE,T_TM',
           comments='',
           fmt='%.1f,%.0f,%.1f,%.0f,%.1f,%.10f,%.10f,%.10f,%.10f')

print(" Data saved to file 'multilayer_data2.csv'")
print(f" Approximate file size: ~{len(numeric_array) * 9 * 8 / 1024 / 1024:.1f} MB")

# === Brief data analysis ===
print("\n" + "="*80)
print("BRIEF DATA ANALYSIS:")
print("="*80)

R_TE_vals = numeric_array[:, 5]
R_TM_vals = numeric_array[:, 6]
T_TE_vals = numeric_array[:, 7]
T_TM_vals = numeric_array[:, 8]

print(f"Reflection R_TE: min={R_TE_vals.min():.6f}, max={R_TE_vals.max():.6f}, "
      f"mean={R_TE_vals.mean():.6f}")
print(f"Reflection R_TM: min={R_TM_vals.min():.6f}, max={R_TM_vals.max():.6f}, "
      f"mean={R_TM_vals.mean():.6f}")
print(f"Transmission T_TE: min={T_TE_vals.min():.6f}, max={T_TE_vals.max():.6f}, "
      f"mean={T_TE_vals.mean():.6f}")
print(f"Transmission T_TM: min={T_TM_vals.min():.6f}, max={T_TM_vals.max():.6f}, "
      f"mean={T_TM_vals.mean():.6f}")

# Energy conservation check
energy_sum_TE = R_TE_vals + T_TE_vals
energy_sum_TM = R_TM_vals + T_TM_vals

print(f"\nEnergy conservation check (mean |R+T-1|):")
print(f" TE: {np.abs(energy_sum_TE - 1.0).mean():.10f}")
print(f" TM: {np.abs(energy_sum_TM - 1.0).mean():.10f}")

print("\n" + "="*80)
print("DONE! The data is ready for neural network training.")
print("="*80)