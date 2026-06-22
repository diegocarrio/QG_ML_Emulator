# QG ML Emulator

## Overview

This project analyzes and compares the statistical distributions of Potential Vorticity (PV) fields from a 2-layer Quasi-Geostrophic (QG) model with predictions from a Machine Learning (ML) emulator. The analysis focuses on identifying non-Gaussian characteristics in ensemble distributions and evaluating how well the ML model reproduces the QG model's statistical properties.

## Dataset

- **Source**: NetCDF file containing 1000 ensemble members
- **Models**: 
  - QG Model (ground truth): 2-layer Quasi-Geostrophic model
  - ML Emulator: Neural network trained to emulate QG dynamics
- **Initialization**: Both ensemble runs initialized identically through MLGETKF (ML-based Ensemble Transform Kalman Filter) cycle
- **Integration period**: 3 hours (representing background ensemble)
- **Spatial grid**: 64×64 points
- **Vertical layers**: 2 layers
- **Variable**: Potential Vorticity (PV)

## Project Structure

```
QG_ML_Emulator/
├── QG_ML_Analysis.ipynb          # Main analysis notebook (all-in-one)
├── Prediction_Target_ens1000.nc  # Input data file (1000 ensemble members)
├── read_ens_sample.py            # Data loading utilities
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Key Analysis Sections

### 1. PV Field Visualization
- Ensemble mean PV fields for both layers
- Visual comparison between QG model and ML emulator
- Identification of spatial structure and gradients

### 2. Basic Statistics
- Comparison of mean, standard deviation, min/max values
- Error metrics (MAE, RMSE) between models
- Verification of ML emulator accuracy

### 3. Normality Analysis
- Shapiro-Wilk test for normality at various grid points
- Skewness and Kurtosis analysis
- Identification of Gaussian vs non-Gaussian distributions

### 4. Detailed Distribution Visualization
- Histograms of ensemble distributions at center and edge points
- Q-Q plots to visually assess deviation from normality
- Superimposed theoretical normal distributions for comparison

### 5. Window-Averaged Analysis
- Ensemble distributions computed from 3×3 windows
- Assessment of whether spatial averaging reduces non-Gaussianity
- Comparison of both models

### 6. Spatial Normality Map
- Shapiro-Wilk p-value computed at all 4096 grid points
- Contour plots showing Gaussian/non-Gaussian regions
- Quantification of non-Gaussian points across the domain

## Key Findings

- **~48% of grid points** exhibit non-Gaussian ensemble distributions in the QG model
- **~52% of grid points** are non-Gaussian in the ML emulator
- Both models show remarkably similar patterns of non-Gaussianity
- Non-Gaussianity persists even when averaging 3×3 windows, indicating fundamental characteristics
- The ML emulator effectively reproduces the statistical properties of the QG model (RMSE ≈ 0.81)
- Non-Gaussian regions appear to correlate with areas of strong PV gradients

## Usage

### Running the Analysis

1. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy netCDF4
   ```

2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook QG_ML_Analysis.ipynb
   ```

3. **Execute cells sequentially** or run all cells to generate:
   - Field visualizations
   - Statistical comparisons
   - Distribution plots
   - Normality maps

### Output

The notebook generates analysis outputs in the console and can save figures locally:
- PV field ensemble means
- Distribution histograms and Q-Q plots
- Window-averaged distributions
- Spatial normality maps

*Note: Figures are generated locally during notebook execution but not committed to the repository.*

## Methodology

### Normality Tests

- **Shapiro-Wilk Test**: Primary test for normality (H₀: data is normally distributed)
  - p-value > 0.05: Accept Gaussian hypothesis
  - p-value < 0.05: Reject Gaussian hypothesis
  
- **Skewness**: Measures asymmetry of the distribution
- **Kurtosis**: Measures tail behavior relative to normal distribution

### Model Comparison

- **QG Model**: Reference/ground truth model
- **ML Emulator**: Trained neural network approximating QG dynamics
- **Error Metrics**:
  - Mean Absolute Error (MAE)
  - Root Mean Square Error (RMSE)

## Variables and Dimensions

- **ens**: 1000 ensemble members
- **z**: 2 vertical layers
- **y**: 64 grid points (north-south)
- **x**: 64 grid points (east-west)
- **pv_qg**: Shape (1000, 2, 64, 64) - QG model PV
- **pv_ml**: Shape (1000, 2, 64, 64) - ML emulator PV

## Future Directions

1. Investigate physical mechanisms behind non-Gaussianity
2. Correlate non-Gaussian regions with PV gradient magnitudes
3. Analyze temporal evolution of non-Gaussianity
4. Examine second layer characteristics
5. Investigate other model variables (relative vorticity, streamfunction, etc.)
6. Extend analysis to longer forecast horizons
7. Develop adaptive ensemble methods accounting for non-Gaussianity

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- netCDF4
- jupyter (for interactive notebook)

## Author

Analysis for QG Model vs ML Emulator comparison project  
Collaboration with Xuguang

## References

- Quasi-Geostrophic model theory
- Ensemble Kalman Filter methods
- Machine Learning emulation techniques
- Statistical distribution analysis
