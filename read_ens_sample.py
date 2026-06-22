from netCDF4 import Dataset

nc_ens_path = "./Prediction_Target_ens1000.nc"

nc_ensemble = Dataset(nc_ens_path)
pv_cnn = nc_ensemble.variables['predicted_pv']
pv_sqg = nc_ensemble.variables['target_pv'] # [ens, nz, ny, nx]


