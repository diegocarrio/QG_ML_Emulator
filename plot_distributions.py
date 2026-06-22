"""
Script para plotear distribuciones de PV en varios puntos de la grilla
Analiza histogramas para detectar no-Gaussianidad
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Cargar datos
nc_ensemble = Dataset("./Prediction_Target_ens1000.nc")
pv_qg = nc_ensemble.variables['target_pv'][:]      # QG model
pv_ml = nc_ensemble.variables['predicted_pv'][:]   # ML emulator

def plot_distribution_at_point(data, layer, y, x, title, ax):
    """Plotear histograma y comparar con distribución normal"""
    ensemble_values = data[:, layer, y, x]
    
    # Estadísticas
    mean = np.mean(ensemble_values)
    std = np.std(ensemble_values)
    
    # Test de normalidad (Shapiro-Wilk)
    stat, p_value = stats.shapiro(ensemble_values)
    
    # Plot
    ax.hist(ensemble_values, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Distribución normal teórica
    x_range = np.linspace(ensemble_values.min(), ensemble_values.max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range, mean, std), 'r-', linewidth=2, label='Normal')
    
    ax.set_xlabel('PV')
    ax.set_ylabel('Density')
    ax.set_title(f"{title}\np-value={p_value:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return p_value

# Seleccionar diferentes puntos para plotear
# Usamos puntos en diferentes posiciones
points_to_plot = [
    (32, 32),  # Centro
    (16, 16),  # Esquina
    (48, 48),  # Otra esquina
    (32, 16),  # Borde
    (16, 32),  # Borde
]

fig, axes = plt.subplots(2, 5, figsize=(16, 10))
axes = axes.flatten()

print("=" * 60)
print("ANÁLISIS DE DISTRIBUCIONES - LAYER 0 (QG Model)")
print("=" * 60)

for idx, (y, x) in enumerate(points_to_plot):
    p_val_qg = plot_distribution_at_point(pv_qg, layer=0, y=y, x=x, 
                                          title=f"QG Model: ({y},{x})", 
                                          axes[idx])
    print(f"Punto ({y},{x}): p-value = {p_val_qg:.6f} {'[NON-Gaussian]' if p_val_qg < 0.05 else '[Gaussian]'}")

print("\n" + "=" * 60)
print("ANÁLISIS DE DISTRIBUCIONES - LAYER 0 (ML Emulator)")
print("=" * 60)

for idx, (y, x) in enumerate(points_to_plot):
    p_val_ml = plot_distribution_at_point(pv_ml, layer=0, y=y, x=x, 
                                          title=f"ML Emulator: ({y},{x})", 
                                          axes[idx+5])
    print(f"Punto ({y},{x}): p-value = {p_val_ml:.6f} {'[NON-Gaussian]' if p_val_ml < 0.05 else '[Gaussian]'}")

plt.tight_layout()
plt.savefig('pv_distributions_comparison.png', dpi=150)
print("\nGráfico guardado: pv_distributions_comparison.png")
plt.close()

nc_ensemble.close()
