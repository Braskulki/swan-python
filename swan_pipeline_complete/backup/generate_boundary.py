import json

import numpy as np
import xarray as xr

from config import (
    BOUNDARY_OUTPUT,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    PROCESSED_DIR,
    WAVES_FILE,
    ensure_directories,
)
from generate_depth import generate_depth


def _find_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.variables or name in dataset.coords or name in dataset.dims:
            return name
    raise KeyError(f"Nenhum nome encontrado entre: {candidates}")


def _load_grid() -> dict:
    grid_file = PROCESSED_DIR / "grid.json"
    if not grid_file.exists():
        return generate_depth()
    return json.loads(grid_file.read_text(encoding="utf-8"))


def _circular_mean_degrees(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Não há direções de onda válidas para calcular a média circular.")

    radians = np.deg2rad(values)
    sine = np.mean(np.sin(radians))
    cosine = np.mean(np.cos(radians))
    return float(np.rad2deg(np.arctan2(sine, cosine)) % 360.0)


def _format_swan_time(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="s")
    date, clock = text.split("T")
    return f"{date.replace('-', '')}.{clock.replace(':', '')}"


def _perimeter(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Esperada matriz 2D; recebida forma {array.shape}.")

    if array.shape[0] < 2 or array.shape[1] < 2:
        return array.ravel()

    return np.concatenate(
        (
            array[0, :],
            array[-1, :],
            array[1:-1, 0],
            array[1:-1, -1],
        )
    )


def _valid_triplet(
    hs: np.ndarray,
    period: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(hs)
        & np.isfinite(period)
        & np.isfinite(direction)
        & (hs >= 0.0)
        & (period > 0.0)
    )
    return hs[mask], period[mask], direction[mask]


def generate_boundary() -> None:
    ensure_directories()

    if not WAVES_FILE.exists():
        raise FileNotFoundError(f"Arquivo ERA5 de ondas não encontrado: {WAVES_FILE}")

    grid = _load_grid()
    target_lon = np.asarray(grid["lon"], dtype=float)
    target_lat = np.asarray(grid["lat"], dtype=float)

    with xr.open_dataset(WAVES_FILE) as dataset:
        lon_name = _find_name(dataset, ("longitude", "lon"))
        lat_name = _find_name(dataset, ("latitude", "lat"))
        time_name = _find_name(
            dataset,
            ("valid_time", "time", "forecast_time", "datetime"),
        )
        hs_name = _find_name(dataset, ("swh",))
        direction_name = _find_name(dataset, ("mwd",))
        period_name = _find_name(dataset, ("mwp", "mwdp"))

        dataset = dataset.sortby(lon_name).sortby(lat_name).sortby(time_name)

        # Mantém também os valores nativos do ERA5 como fallback. Os campos de
        # onda possuem NaN sobre terra, o que é esperado em domínios costeiros.
        native_subset = dataset.sel(
            {
                lon_name: slice(LON_MIN, LON_MAX),
                lat_name: slice(LAT_MIN, LAT_MAX),
            }
        )

        def interpolate(variable_name: str) -> xr.DataArray:
            variable = dataset[variable_name].squeeze(drop=True)
            result = variable.interp(
                {
                    lon_name: xr.DataArray(target_lon, dims=(lon_name,)),
                    lat_name: xr.DataArray(target_lat, dims=(lat_name,)),
                },
                method="linear",
            )
            return result.transpose(time_name, lat_name, lon_name)

        hs_interp = np.asarray(interpolate(hs_name).values, dtype=float)
        period_interp = np.asarray(interpolate(period_name).values, dtype=float)
        direction_interp = np.asarray(interpolate(direction_name).values, dtype=float)

        hs_native = np.asarray(
            native_subset[hs_name].squeeze(drop=True).transpose(time_name, lat_name, lon_name).values,
            dtype=float,
        )
        period_native = np.asarray(
            native_subset[period_name].squeeze(drop=True).transpose(time_name, lat_name, lon_name).values,
            dtype=float,
        )
        direction_native = np.asarray(
            native_subset[direction_name].squeeze(drop=True).transpose(time_name, lat_name, lon_name).values,
            dtype=float,
        )
        times = np.asarray(dataset[time_name].values)

    lines = ["TPAR"]
    diagnostics: list[dict] = []

    for index, time_value in enumerate(times):
        # Primeira escolha: pontos oceânicos válidos no perímetro interpolado.
        hs_valid, period_valid, direction_valid = _valid_triplet(
            _perimeter(hs_interp[index]),
            _perimeter(period_interp[index]),
            _perimeter(direction_interp[index]),
        )
        source = "interpolated_perimeter"

        # Se o perímetro cruza muita terra, usa todos os pontos oceânicos válidos
        # da grade interpolada.
        if hs_valid.size == 0:
            hs_valid, period_valid, direction_valid = _valid_triplet(
                hs_interp[index].ravel(),
                period_interp[index].ravel(),
                direction_interp[index].ravel(),
            )
            source = "interpolated_domain"

        # Último fallback: valores oceânicos nativos do próprio ERA5 no recorte.
        if hs_valid.size == 0:
            hs_valid, period_valid, direction_valid = _valid_triplet(
                hs_native[index].ravel(),
                period_native[index].ravel(),
                direction_native[index].ravel(),
            )
            source = "native_era5_domain"

        if hs_valid.size == 0:
            raise ValueError(
                f"Nenhum dado oceânico válido de ondas em "
                f"{np.datetime_as_string(time_value, unit='s')}. "
                "Confirme a área e as variáveis existentes em waves.nc."
            )

        hs_mean = max(float(np.mean(hs_valid)), 0.01)
        period_mean = max(float(np.mean(period_valid)), 0.1)
        direction_mean = _circular_mean_degrees(direction_valid)
        directional_spread = 30.0

        lines.append(
            f"{_format_swan_time(time_value)} "
            f"{hs_mean:.4f} {period_mean:.4f} "
            f"{direction_mean:.2f} {directional_spread:.2f}"
        )
        diagnostics.append(
            {
                "time": np.datetime_as_string(time_value, unit="s"),
                "valid_points": int(hs_valid.size),
                "source": source,
                "hs": hs_mean,
                "period": period_mean,
                "direction": direction_mean,
            }
        )

    BOUNDARY_OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")

    metadata = {
        "time_count": int(times.size),
        "source_variables": {
            "significant_wave_height": hs_name,
            "wave_period": period_name,
            "wave_direction": direction_name,
        },
        "method": (
            "Usa pontos oceânicos válidos do perímetro interpolado; quando não "
            "existem, usa o domínio interpolado e depois o domínio nativo ERA5."
        ),
        "directional_spread_degrees": 30.0,
        "diagnostics": diagnostics,
    }
    metadata_file = PROCESSED_DIR / "boundary_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Condição de contorno TPAR gerada: {BOUNDARY_OUTPUT}")
    print(f"Horários: {times.size}")
    print(f"Metadados: {metadata_file}")


if __name__ == "__main__":
    generate_boundary()