"""
Script para leer y explorar los datos del ensemble QG y ML emulator
Datos: 1000 miembros ensemble de 2 capas QG model vs ML emulator
Variable: Vorticidad Potencial (Potential Vorticity - PV)
Grilla: 64x64 puntos
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Cargar datos
nc_ens_path = "./Prediction_Target_ens1000.nc"
nc_ensemble = Dataset(nc_ens_path)

# Variables principales
pv_ml = nc_ensemble.variables['predicted_pv'][:]      # ML emulator: [ens, nz, ny, nx]
pv_qg = nc_ensemble.variables['target_pv'][:]         # QG model (truth): [ens, nz, ny, nx]

print(f"Shape de datos:")
print(f"  PV ML emulator: {pv_ml.shape}")
print(f"  PV QG model: {pv_qg.shape}")
print(f"  Ensemble members: {pv_ml.shape[0]}")
print(f"  Capas: {pv_ml.shape[1]}")
print(f"  Grilla: {pv_ml.shape[2]}x{pv_ml.shape[3]}")

# Estadísticas básicas
print("\n=== Estadísticas QG Model ===")
print(f"Media: {np.mean(pv_qg):.6f}")
print(f"Std: {np.std(pv_qg):.6f}")
print(f"Min: {np.min(pv_qg):.6f}")
print(f"Max: {np.max(pv_qg):.6f}")

print("\n=== Estadísticas ML Emulator ===")
print(f"Media: {np.mean(pv_ml):.6f}")
print(f"Std: {np.std(pv_ml):.6f}")
print(f"Min: {np.min(pv_ml):.6f}")
print(f"Max: {np.max(pv_ml):.6f}")

nc_ensemble.close()


