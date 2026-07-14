import json
import numpy as np
import xarray as xr
from config import DEPTH_OUTPUT, GEBCO_FILE, GEBCO_STEP, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROCESSED_DIR, ensure_directories

def _coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords:
            return name
    raise KeyError(f"Nenhuma coordenada encontrada entre: {candidates}")

def generate_depth() -> dict:
    ensure_directories()
    if not GEBCO_FILE.exists():
        raise FileNotFoundError(f"Arquivo GEBCO não encontrado: {GEBCO_FILE}")
    with xr.open_dataset(GEBCO_FILE) as dataset:
        lon_name = _coord_name(dataset, ("lon", "longitude"))
        lat_name = _coord_name(dataset, ("lat", "latitude"))
        if "elevation" not in dataset:
            raise KeyError("A variável 'elevation' não existe no arquivo GEBCO.")
        dataset = dataset.sortby(lon_name).sortby(lat_name)
        subset = dataset.sel({lon_name: slice(LON_MIN, LON_MAX), lat_name: slice(LAT_MIN, LAT_MAX)})
        lon = np.asarray(subset[lon_name].values)[::GEBCO_STEP]
        lat = np.asarray(subset[lat_name].values)[::GEBCO_STEP]
        elevation = np.asarray(subset["elevation"].values)[::GEBCO_STEP, ::GEBCO_STEP]
    if lon.size < 2 or lat.size < 2:
        raise ValueError("O recorte/subsample gerou uma grade com menos de dois pontos.")
    if elevation.shape != (lat.size, lon.size):
        raise ValueError(f"Dimensão inconsistente: elevation={elevation.shape}, lat={lat.size}, lon={lon.size}")
    depth = -elevation.astype(float)
    if not np.isfinite(depth).all():
        raise ValueError("A batimetria contém NaN ou infinito.")
    np.savetxt(DEPTH_OUTPUT, depth, fmt="%.2f")
    dx_values = np.diff(lon); dy_values = np.diff(lat)
    dx = float(np.mean(dx_values)); dy = float(np.mean(dy_values))
    if not np.allclose(dx_values, dx, rtol=1e-6, atol=1e-10):
        raise ValueError("A longitude subamostrada não forma uma grade regular.")
    if not np.allclose(dy_values, dy, rtol=1e-6, atol=1e-10):
        raise ValueError("A latitude subamostrada não forma uma grade regular.")
    grid = {"x0": float(lon[0]), "y0": float(lat[0]), "dx": dx, "dy": dy, "nx": int(lon.size), "ny": int(lat.size), "mx": int(lon.size-1), "my": int(lat.size-1), "x_length": float(lon[-1]-lon[0]), "y_length": float(lat[-1]-lat[0]), "lon": lon.tolist(), "lat": lat.tolist()}
    grid_file = PROCESSED_DIR / "grid.json"
    grid_file.write_text(json.dumps(grid, indent=2), encoding="utf-8")
    print(f"Batimetria gerada: {DEPTH_OUTPUT}")
    print(f"Grade: {grid['nx']} x {grid['ny']} pontos")
    return grid

if __name__ == "__main__":
    generate_depth()
