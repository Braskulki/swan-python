import xarray as xr
import numpy as np

ds = xr.open_dataset("data/raw/wind.nc")

u = ds['u10']
v = ds['v10']

speed = np.sqrt(u**2 + v**2)
direction = (270 - np.degrees(np.arctan2(v, u))) % 360

time_dim = [d for d in ds.dims if "time" in d.lower()][0]
times = ds[time_dim].values


with open("data/processed/wind.txt", "w") as f:
    for t in range(len(times)):
        date = str(np.datetime_as_string(times[t], unit='s')) \
            .replace("-", "").replace(":", "").replace("T", ".")
        
        uu = float(u.isel({time_dim: t}).mean())
        vv = float(v.isel({time_dim: t}).mean())
        
        f.write(f"{date} {uu:.2f} {vv:.2f}\n")