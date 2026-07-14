import xarray as xr
import numpy as np

STEP = 9

bathy = xr.open_dataset("data/raw/gebco.nc")

# Recorte
sub = bathy.sel(
    lat=slice(-25, -22),
    lon=slice(-48, -44)
)

lon = sub.lon.values[::STEP]
lat = sub.lat.values[::STEP]

depth = -sub.elevation.values
depth = depth[::STEP, ::STEP]

# Opcional: substituir terra por profundidade mínima
depth[depth <= 0] = 0.1

np.savetxt(
    "data/processed/depth.bot",
    depth,
    fmt="%.2f"
)

nx = len(lon)
ny = len(lat)

mx = nx - 1
my = ny - 1

x0 = float(lon[0])
y0 = float(lat[0])

dx = float(lon[1] - lon[0])
dy = float(lat[1] - lat[0])

xlenc = dx * mx
ylenc = dy * my

print(f"""
CGRID REGULAR {x0} {y0} 0 {xlenc} {ylenc} {mx} {my} CIRCLE 36 0.04 1.0

INPGRID BOTTOM REGULAR {x0} {y0} 0 {mx} {my} {dx} {dy}

READINP BOTTOM 1 'depth.bot' 4 0 FREE
""")