import xarray as xr
import numpy as np

bathy = xr.open_dataset("data/raw/gebco.nc")

# recorte SP
sub = bathy.sel(lat=slice(-25,-22), lon=slice(-48,-44))

depth = sub['elevation'].values  # negativo = oceano

# SWAN usa profundidade positiva
depth = -depth

# Subsample to match grid
depth = depth[::9, ::9][:80, :100]

np.savetxt("data/processed/depth.bot", depth, fmt="%.2f")