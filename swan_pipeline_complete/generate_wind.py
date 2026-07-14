import json
import numpy as np
import xarray as xr
from config import PROCESSED_DIR, WIND_FILE, WIND_OUTPUT, ensure_directories
from generate_depth import generate_depth

def _find_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.variables or name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(f"Nenhum nome encontrado entre: {candidates}")

def _load_grid() -> dict:
    grid_file = PROCESSED_DIR / "grid.json"
    return generate_depth() if not grid_file.exists() else json.loads(grid_file.read_text(encoding="utf-8"))

def generate_wind() -> None:
    ensure_directories()
    if not WIND_FILE.exists():
        raise FileNotFoundError(f"Arquivo ERA5 de vento não encontrado: {WIND_FILE}")
    grid = _load_grid(); target_lon=np.asarray(grid['lon']); target_lat=np.asarray(grid['lat'])
    with xr.open_dataset(WIND_FILE) as dataset:
        lon_name=_find_name(dataset,("longitude","lon")); lat_name=_find_name(dataset,("latitude","lat")); time_name=_find_name(dataset,("valid_time","time","forecast_time","datetime")); u_name=_find_name(dataset,("u10","10u")); v_name=_find_name(dataset,("v10","10v"))
        dataset=dataset.sortby(lon_name).sortby(lat_name).sortby(time_name)
        u=dataset[u_name].squeeze(drop=True); v=dataset[v_name].squeeze(drop=True)
        u_i=u.interp({lon_name:xr.DataArray(target_lon,dims=(lon_name,)),lat_name:xr.DataArray(target_lat,dims=(lat_name,))}).transpose(time_name,lat_name,lon_name)
        v_i=v.interp({lon_name:xr.DataArray(target_lon,dims=(lon_name,)),lat_name:xr.DataArray(target_lat,dims=(lat_name,))}).transpose(time_name,lat_name,lon_name)
        u_values=np.asarray(u_i.values,float); v_values=np.asarray(v_i.values,float); times=np.asarray(dataset[time_name].values)
    expected=(times.size,grid['ny'],grid['nx'])
    if u_values.shape!=expected or v_values.shape!=expected:
        raise ValueError(f"Forma inesperada. Esperado {expected}; U={u_values.shape}, V={v_values.shape}")
    if not np.isfinite(u_values).all() or not np.isfinite(v_values).all():
        raise ValueError("A interpolação do vento produziu NaN.")
    with WIND_OUTPUT.open('w',encoding='ascii',newline='') as stream:
        for i in range(times.size):
            np.savetxt(stream,u_values[i],fmt='%.6f')
            np.savetxt(stream,v_values[i],fmt='%.6f')
    (PROCESSED_DIR/'wind_metadata.json').write_text(json.dumps({'time_count':int(times.size),'times':[np.datetime_as_string(t,unit='s') for t in times],'layout':'Para cada horário: matriz U10 seguida da matriz V10.'},indent=2),encoding='utf-8')
    print(f"Vento gerado: {WIND_OUTPUT}")

if __name__=='__main__':
    generate_wind()
