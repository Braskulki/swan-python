import xarray as xr
import numpy as np


ds = xr.open_dataset("data/raw/waves.nc")

hs = ds['swh']
tp = ds['mwp']
dir = ds['mwd']

time_dim = [d for d in ds.dims if "time" in d.lower()][0]
times = ds[time_dim].values

with open("data/processed/boundary.txt", "w") as f:
    for t in range(len(times)):
        date = str(np.datetime_as_string(times[t], unit='s')) \
            .replace("-", "").replace(":", "").replace("T", ".")
        
        f.write(f"{date} {float(hs.isel({time_dim: t}).mean()):.2f} "
                f"{float(tp.isel({time_dim: t}).mean()):.2f} "
                f"{float(dir.isel({time_dim: t}).mean()):.2f}\n")