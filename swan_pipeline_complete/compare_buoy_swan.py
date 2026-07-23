from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import loadmat
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent
MAT_FILE = BASE_DIR / "data" / "unstructured_research" / "output_unstructured.mat"
MESH_NODE_FILE = BASE_DIR / "data" / "unstructured_research" / "mesh.node"
BUOY_FILE = BASE_DIR / "data" / "buoy" / "Bsantos_observations.csv"
BUOY_NETCDF_FILE = BASE_DIR / "data" / "buoy" / "Bsantos.nc"
OUTPUT_DIR = BASE_DIR / "data" / "unstructured_research" / "publication" / "buoy_comparison"

BUOY_NAME = "Boia de Santos"
BUOY_DATETIME_COLUMN = "datetime"
BUOY_HS_COLUMN = "wave_hs"
BUOY_TP_COLUMN = "wave_period"
BUOY_DIR_COLUMN = "wave_dir"
BUOY_TIMEZONE = "UTC"
TIME_TOLERANCE = "45min"
MAX_NODE_DISTANCE_KM = 10.0
MODEL_DIRECTION_OFFSET = 0.0
BUOY_DIRECTION_OFFSET = 0.0
CONVERT_MODEL_TO_FROM_DIRECTION = False
HS_PREFIX = "Hsig"
TP_PREFIX = "TPsmoo"
DIR_PREFIX = "Dir"
INVALID_ABS_LIMIT = 1.0e20

TIMESTAMP_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<date>\d{8})_(?P<time>\d{6})$")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def available_mat_variables(mat: dict[str, Any]) -> list[str]:
    return sorted(key for key in mat if not key.startswith("__"))


def inspect_mat(mat: dict[str, Any]) -> None:
    print("\nVariáveis encontradas no MAT:")
    for key in available_mat_variables(mat):
        value = np.asarray(mat[key])
        print(f"  {key:<32} shape={str(value.shape):<18} dtype={value.dtype}")


def clean_swan_values(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.asarray(values).astype(np.float64, copy=True, casting="unsafe")
    result[~np.isfinite(result) | (np.abs(result) >= INVALID_ABS_LIMIT)] = np.nan
    return result


def extract_timestamped_mat_series(
    mat: dict[str, Any], prefix: str, expected_nodes: int
) -> tuple[pd.DatetimeIndex, np.ndarray, list[str]]:
    matches: list[tuple[pd.Timestamp, str, np.ndarray]] = []
    expected_prefix = prefix.casefold()

    for key, raw_value in mat.items():
        if key.startswith("__"):
            continue
        match = TIMESTAMP_PATTERN.match(key)
        if match is None or match.group("prefix").casefold() != expected_prefix:
            continue

        timestamp = pd.to_datetime(
            match.group("date") + match.group("time"),
            format="%Y%m%d%H%M%S",
            utc=True,
        )
        values = clean_swan_values(np.asarray(raw_value).squeeze())
        if values.ndim != 1:
            raise ValueError(f"A variável {key!r} possui shape {values.shape}; esperado vetor 1D.")
        if values.size != expected_nodes:
            raise ValueError(
                f"A variável {key!r} possui {values.size} valores, mas a malha possui {expected_nodes} nós."
            )
        matches.append((timestamp, key, values))

    if not matches:
        raise KeyError(f"Nenhuma variável com prefixo {prefix!r} foi encontrada no MAT.")

    matches.sort(key=lambda item: item[0])
    timestamps = pd.DatetimeIndex([item[0] for item in matches])
    names = [item[1] for item in matches]
    matrix = np.vstack([item[2] for item in matches])
    return timestamps, matrix, names


def load_triangle_node_file(filename: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not filename.exists():
        raise FileNotFoundError(f"Arquivo mesh.node não encontrado: {filename}")

    valid_lines: list[str] = []
    with filename.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#", 1)[0].strip()
            if line:
                valid_lines.append(line)

    if not valid_lines:
        raise ValueError(f"O arquivo mesh.node está vazio: {filename}")

    header = valid_lines[0].split()
    number_of_nodes = int(float(header[0]))
    dimension = int(float(header[1]))
    if dimension < 2:
        raise ValueError(f"Dimensão inválida no mesh.node: {dimension}")

    records: list[tuple[int, float, float]] = []
    for line in valid_lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            records.append((int(float(parts[0])), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
        if len(records) == number_of_nodes:
            break

    if len(records) != number_of_nodes:
        raise ValueError(
            f"O cabeçalho informa {number_of_nodes} nós, mas foram lidos {len(records)}."
        )

    records.sort(key=lambda item: item[0])
    node_ids = np.asarray([item[0] for item in records], dtype=int)
    node_lon = np.asarray([item[1] for item in records], dtype=float)
    node_lat = np.asarray([item[2] for item in records], dtype=float)

    print("\nMalha carregada:")
    print(f"  arquivo: {filename}")
    print(f"  nós: {len(node_ids):,}")
    print(f"  longitude/X: {node_lon.min():.6f} até {node_lon.max():.6f}")
    print(f"  latitude/Y: {node_lat.min():.6f} até {node_lat.max():.6f}")
    return node_ids, node_lon, node_lat


def extract_first_finite(values: Any) -> float | None:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)]
    return float(array[0]) if array.size else None


def read_buoy_coordinates_from_netcdf(filename: Path) -> tuple[float, float]:
    if not filename.exists():
        raise FileNotFoundError(f"NetCDF da boia não encontrado: {filename}")

    lon_aliases = ["longitude", "lon", "long", "x"]
    lat_aliases = ["latitude", "lat", "y"]
    longitude = None
    latitude = None

    with xr.open_dataset(filename) as dataset:
        print("\nEstrutura resumida do NetCDF da boia:")
        print(dataset)
        normalized_variables = {normalize_name(name): name for name in dataset.variables}

        for alias in lon_aliases:
            key = normalized_variables.get(normalize_name(alias))
            if key is not None:
                longitude = extract_first_finite(dataset[key].values)
                if longitude is not None:
                    break

        for alias in lat_aliases:
            key = normalized_variables.get(normalize_name(alias))
            if key is not None:
                latitude = extract_first_finite(dataset[key].values)
                if latitude is not None:
                    break

        normalized_attrs = {normalize_name(name): name for name in dataset.attrs}
        if longitude is None:
            for alias in lon_aliases:
                key = normalized_attrs.get(normalize_name(alias))
                if key is not None:
                    longitude = extract_first_finite(dataset.attrs[key])
                    if longitude is not None:
                        break

        if latitude is None:
            for alias in lat_aliases:
                key = normalized_attrs.get(normalize_name(alias))
                if key is not None:
                    latitude = extract_first_finite(dataset.attrs[key])
                    if latitude is not None:
                        break

        if longitude is None or latitude is None:
            for variable_name in dataset.variables:
                variable = dataset[variable_name]
                standard_name = str(variable.attrs.get("standard_name", "")).casefold()
                axis = str(variable.attrs.get("axis", "")).casefold()
                units = str(variable.attrs.get("units", "")).casefold()
                if longitude is None and (
                    standard_name == "longitude" or axis == "x" or "degrees_east" in units
                ):
                    longitude = extract_first_finite(variable.values)
                if latitude is None and (
                    standard_name == "latitude" or axis == "y" or "degrees_north" in units
                ):
                    latitude = extract_first_finite(variable.values)

    if longitude is None or latitude is None:
        raise ValueError("Não foi possível localizar latitude e longitude no Bsantos.nc.")

    if longitude > 180:
        longitude -= 360.0
    if not (-180 <= longitude <= 180) or not (-90 <= latitude <= 90):
        raise ValueError(f"Coordenadas inválidas no NetCDF: lon={longitude}, lat={latitude}")

    print("\nCoordenadas obtidas do NetCDF:")
    print(f"  longitude: {longitude:.6f}")
    print(f"  latitude: {latitude:.6f}")
    return longitude, latitude


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0088
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.radians, [lon1, lat1, lon2, lat2])
    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return float(2.0 * radius_km * np.arcsin(np.sqrt(a)))


def find_nearest_node(
    node_lon: np.ndarray, node_lat: np.ndarray, buoy_lon: float, buoy_lat: float
) -> tuple[int, float]:
    mean_latitude = np.radians(buoy_lat)
    projected_nodes = np.column_stack([node_lon * np.cos(mean_latitude), node_lat])
    projected_buoy = np.array([buoy_lon * np.cos(mean_latitude), buoy_lat])
    tree = cKDTree(projected_nodes)
    _, node_index = tree.query(projected_buoy, k=1)
    node_index = int(node_index)
    distance_km = haversine_distance_km(
        buoy_lon, buoy_lat, float(node_lon[node_index]), float(node_lat[node_index])
    )
    return node_index, distance_km


def normalize_direction(
    values: pd.Series | np.ndarray, offset: float = 0.0, reverse_by_180: bool = False
) -> np.ndarray:
    direction = np.asarray(values, dtype=float)
    if reverse_by_180:
        direction = direction + 180.0
    return (direction + offset) % 360.0


def circular_difference_degrees(modeled: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return (modeled - observed + 180.0) % 360.0 - 180.0


def detect_column(
    columns: list[str], configured_name: str, aliases: list[str], label: str
) -> str:
    normalized_columns = {normalize_name(column): column for column in columns}
    configured_key = normalize_name(configured_name)
    if configured_key in normalized_columns:
        return normalized_columns[configured_key]
    for alias in aliases:
        alias_key = normalize_name(alias)
        if alias_key in normalized_columns:
            detected = normalized_columns[alias_key]
            print(f"  coluna {label} detectada automaticamente: {detected}")
            return detected
    raise ValueError(
        f"Não foi possível localizar a coluna de {label}.\n"
        f"Nome configurado: {configured_name!r}\n"
        f"Colunas disponíveis: {columns}"
    )


def load_buoy_data() -> pd.DataFrame:
    if not BUOY_FILE.exists():
        raise FileNotFoundError(f"Arquivo da boia não encontrado: {BUOY_FILE}")

    buoy = pd.read_csv(BUOY_FILE)
    print("\nColunas encontradas no CSV da boia:")
    print(f"  {buoy.columns.tolist()}")
    columns = buoy.columns.tolist()

    datetime_column = detect_column(
        columns,
        BUOY_DATETIME_COLUMN,
        ["datetime", "date_time", "timestamp", "time", "date", "datahora", "data_hora", "valid_time"],
        "data/hora",
    )
    hs_column = detect_column(
        columns,
        BUOY_HS_COLUMN,
        [
            "hs",
            "hsig",
            "wave_hs",
            "significant_wave_height",
            "wave_height",
            "swh",
            "vhm0",
        ],
        "Hs",
    )
    tp_column = detect_column(
        columns,
        BUOY_TP_COLUMN,
        [
            "tp",
            "tpeak",
            "wave_period",
            "peak_period",
            "wave_peak_period",
            "vtpk",
            "rtp",
        ],
        "Tp",
    )
    dir_column = detect_column(
        columns,
        BUOY_DIR_COLUMN,
        [
            "dir",
            "wave_dir",
            "direction",
            "wave_direction",
            "mean_wave_direction",
            "peak_wave_direction",
            "mwd",
            "vmdr",
        ],
        "direção",
    )

    datetime_values = pd.to_datetime(buoy[datetime_column], errors="coerce")
    if datetime_values.dt.tz is None:
        datetime_values = datetime_values.dt.tz_localize(
            BUOY_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
        )
    datetime_values = datetime_values.dt.tz_convert("UTC")

    result = pd.DataFrame(
        {
            "datetime": datetime_values,
            "buoy_hs": pd.to_numeric(buoy[hs_column], errors="coerce"),
            "buoy_tp": pd.to_numeric(buoy[tp_column], errors="coerce"),
            "buoy_dir": pd.to_numeric(buoy[dir_column], errors="coerce"),
        }
    )
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["datetime"])
    result.loc[result["buoy_hs"] < 0, "buoy_hs"] = np.nan
    result.loc[result["buoy_tp"] <= 0, "buoy_tp"] = np.nan
    result["buoy_dir"] = normalize_direction(result["buoy_dir"], offset=BUOY_DIRECTION_OFFSET)
    result = (
        result.sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    print("\nDados da boia:")
    print(f"  arquivo: {BUOY_FILE}")
    print(f"  registros válidos: {len(result):,}")
    if not result.empty:
        print(f"  período: {result['datetime'].min()} até {result['datetime'].max()}")
    return result


def align_buoy_to_model_times(buoy: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    model_times = model[["datetime"]].sort_values("datetime")
    buoy_sorted = buoy.sort_values("datetime")
    matched_buoy = pd.merge_asof(
        model_times,
        buoy_sorted,
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(TIME_TOLERANCE),
    )
    comparison = model.merge(matched_buoy, on="datetime", how="left")
    comparison = comparison.dropna(subset=["buoy_hs", "buoy_tp", "buoy_dir"], how="all")
    if comparison.empty:
        raise ValueError(
            "Nenhum horário da boia foi associado aos horários do SWAN. "
            "Verifique datas, timezone e TIME_TOLERANCE."
        )
    comparison["dir_error"] = circular_difference_degrees(
        comparison["model_dir"].to_numpy(dtype=float),
        comparison["buoy_dir"].to_numpy(dtype=float),
    )
    return comparison.reset_index(drop=True)


def linear_metrics(observed: np.ndarray, modeled: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(modeled)
    observed = observed[valid]
    modeled = modeled[valid]

    if observed.size == 0:
        return {
            "n": 0,
            "observed_mean": np.nan,
            "modeled_mean": np.nan,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,
            "scatter_index": np.nan,
        }

    errors = modeled - observed
    correlation = np.nan
    if observed.size >= 2 and np.nanstd(observed) > 0 and np.nanstd(modeled) > 0:
        correlation = float(np.corrcoef(observed, modeled)[0, 1])

    observed_mean = float(np.nanmean(observed))
    centered_rmse = float(
        np.sqrt(np.nanmean((errors - np.nanmean(errors)) ** 2))
    )
    scatter_index = centered_rmse / abs(observed_mean) if observed_mean != 0 else np.nan

    return {
        "n": int(observed.size),
        "observed_mean": observed_mean,
        "modeled_mean": float(np.nanmean(modeled)),
        "bias": float(np.nanmean(errors)),
        "mae": float(np.nanmean(np.abs(errors))),
        "rmse": float(np.sqrt(np.nanmean(errors**2))),
        "correlation": correlation,
        "scatter_index": float(scatter_index),
    }


def directional_metrics(observed: np.ndarray, modeled: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(modeled)
    observed = observed[valid]
    modeled = modeled[valid]

    if observed.size == 0:
        return {
            "n": 0,
            "circular_bias": np.nan,
            "circular_mae": np.nan,
            "circular_rmse": np.nan,
        }

    differences = circular_difference_degrees(modeled, observed)
    return {
        "n": int(observed.size),
        "circular_bias": float(np.nanmean(differences)),
        "circular_mae": float(np.nanmean(np.abs(differences))),
        "circular_rmse": float(np.sqrt(np.nanmean(differences**2))),
    }


def build_metrics_table(comparison: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"variable": "Hs", **linear_metrics(comparison["buoy_hs"], comparison["model_hs"])},
            {"variable": "Tp", **linear_metrics(comparison["buoy_tp"], comparison["model_tp"])},
            {
                "variable": "Direction",
                **directional_metrics(comparison["buoy_dir"], comparison["model_dir"]),
            },
        ]
    )


def plot_time_series(
    comparison: pd.DataFrame,
    observed_column: str,
    modeled_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    valid = comparison[["datetime", observed_column, modeled_column]].dropna(
        subset=[observed_column, modeled_column], how="all"
    )
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(valid["datetime"], valid[observed_column], marker="o", label="Boia", linewidth=1.4)
    ax.plot(valid["datetime"], valid[modeled_column], marker="s", label="SWAN", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Data e hora — UTC")
    ax.set_ylabel(ylabel)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    comparison: pd.DataFrame,
    observed_column: str,
    modeled_column: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    observed = comparison[observed_column].to_numpy(dtype=float)
    modeled = comparison[modeled_column].to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(modeled)
    observed = observed[valid]
    modeled = modeled[valid]
    if observed.size == 0:
        return

    minimum = float(min(np.nanmin(observed), np.nanmin(modeled)))
    maximum = float(max(np.nanmax(observed), np.nanmax(modeled)))
    if np.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.05, 0.1)
        minimum -= padding
        maximum += padding

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(observed, modeled, alpha=0.7)
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=1.0, label="Linha 1:1")
    ax.set_xlim(minimum, maximum)
    ax.set_ylim(minimum, maximum)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_error(comparison: pd.DataFrame) -> None:
    valid = comparison[["datetime", "dir_error"]].dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(valid["datetime"], valid["dir_error"], marker="o", linewidth=1.4)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_title(f"{BUOY_NAME}: erro direcional SWAN − boia")
    ax.set_xlabel("Data e hora — UTC")
    ax.set_ylabel("Erro direcional (°)")
    ax.set_ylim(-180, 180)
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "direction_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MAT_FILE.exists():
        raise FileNotFoundError(f"MAT do SWAN não encontrado: {MAT_FILE}")

    print(f"Lendo MAT: {MAT_FILE}")
    mat = loadmat(MAT_FILE, squeeze_me=True, struct_as_record=False)
    inspect_mat(mat)

    node_ids, node_lon, node_lat = load_triangle_node_file(MESH_NODE_FILE)
    number_of_nodes = len(node_lon)
    buoy_longitude, buoy_latitude = read_buoy_coordinates_from_netcdf(BUOY_NETCDF_FILE)

    model_time, hs_values, hs_variables = extract_timestamped_mat_series(
        mat, HS_PREFIX, number_of_nodes
    )
    tp_time, tp_values, tp_variables = extract_timestamped_mat_series(
        mat, TP_PREFIX, number_of_nodes
    )
    dir_time, dir_values, dir_variables = extract_timestamped_mat_series(
        mat, DIR_PREFIX, number_of_nodes
    )

    if not model_time.equals(tp_time):
        raise ValueError("Os horários de Hsig e TPsmoo são diferentes.")
    if not model_time.equals(dir_time):
        raise ValueError("Os horários de Hsig e Dir são diferentes.")

    print("\nResultados organizados:")
    print(f"  Hsig:   {hs_values.shape}")
    print(f"  TPsmoo: {tp_values.shape}")
    print(f"  Dir:    {dir_values.shape}")
    print(f"  período: {model_time.min()} até {model_time.max()}")
    print("\nPrimeira e última variável:")
    print(f"  Hsig: {hs_variables[0]} ... {hs_variables[-1]}")
    print(f"  TPsmoo: {tp_variables[0]} ... {tp_variables[-1]}")
    print(f"  Dir: {dir_variables[0]} ... {dir_variables[-1]}")

    node_index, distance_km = find_nearest_node(
        node_lon, node_lat, buoy_longitude, buoy_latitude
    )

    print("\nNó usado na comparação:")
    print(f"  posição no vetor: {node_index}")
    print(f"  ID no mesh.node: {node_ids[node_index]}")
    print(f"  longitude: {node_lon[node_index]:.6f}")
    print(f"  latitude: {node_lat[node_index]:.6f}")
    print(f"  distância da boia: {distance_km:.3f} km")

    if distance_km > MAX_NODE_DISTANCE_KM:
        raise ValueError(
            "A boia está muito distante do nó mais próximo da malha.\n"
            f"Distância encontrada: {distance_km:.3f} km\n"
            f"Limite configurado: {MAX_NODE_DISTANCE_KM:.3f} km\n"
            "A comparação foi interrompida para evitar validação espacial incorreta."
        )

    model_hs = clean_swan_values(hs_values[:, node_index])
    model_tp = clean_swan_values(tp_values[:, node_index])
    model_dir = clean_swan_values(dir_values[:, node_index])
    model_hs[model_hs < 0] = np.nan
    model_tp[model_tp <= 0] = np.nan
    model_dir = normalize_direction(
        model_dir,
        offset=MODEL_DIRECTION_OFFSET,
        reverse_by_180=CONVERT_MODEL_TO_FROM_DIRECTION,
    )

    model = pd.DataFrame(
        {
            "datetime": model_time,
            "model_hs": model_hs,
            "model_tp": model_tp,
            "model_dir": model_dir,
        }
    ).sort_values("datetime").reset_index(drop=True)

    print("\nResultados SWAN no nó escolhido:")
    print(model.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    model.to_csv(OUTPUT_DIR / "swan_at_buoy_node.csv", index=False, float_format="%.6f")

    buoy = load_buoy_data()
    comparison = align_buoy_to_model_times(buoy, model)
    metrics = build_metrics_table(comparison)

    comparison_file = OUTPUT_DIR / "buoy_swan_comparison.csv"
    metrics_file = OUTPUT_DIR / "validation_metrics.csv"
    node_file = OUTPUT_DIR / "selected_node.csv"

    comparison.to_csv(comparison_file, index=False, float_format="%.6f")
    metrics.to_csv(metrics_file, index=False, float_format="%.6f")
    pd.DataFrame(
        [
            {
                "buoy_name": BUOY_NAME,
                "buoy_longitude": buoy_longitude,
                "buoy_latitude": buoy_latitude,
                "node_index_python": node_index,
                "node_id": int(node_ids[node_index]),
                "node_longitude": float(node_lon[node_index]),
                "node_latitude": float(node_lat[node_index]),
                "distance_km": distance_km,
            }
        ]
    ).to_csv(node_file, index=False, float_format="%.8f")

    plot_time_series(
        comparison,
        "buoy_hs",
        "model_hs",
        "Altura significativa Hs (m)",
        f"{BUOY_NAME}: Hs observado × SWAN",
        "timeseries_hs.png",
    )
    plot_time_series(
        comparison,
        "buoy_tp",
        "model_tp",
        "Período de pico Tp (s)",
        f"{BUOY_NAME}: Tp observado × SWAN",
        "timeseries_tp.png",
    )
    plot_time_series(
        comparison,
        "buoy_dir",
        "model_dir",
        "Direção (°)",
        f"{BUOY_NAME}: direção observada × SWAN",
        "timeseries_direction.png",
    )
    plot_scatter(
        comparison,
        "buoy_hs",
        "model_hs",
        "Hs observado (m)",
        "Hs SWAN (m)",
        "Dispersão de Hs",
        "scatter_hs.png",
    )
    plot_scatter(
        comparison,
        "buoy_tp",
        "model_tp",
        "Tp observado (s)",
        "Tp SWAN (s)",
        "Dispersão de Tp",
        "scatter_tp.png",
    )
    plot_direction_error(comparison)

    print("\nComparação alinhada:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMétricas:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nArquivos gerados:")
    for path in [
        OUTPUT_DIR / "swan_at_buoy_node.csv",
        comparison_file,
        metrics_file,
        node_file,
        OUTPUT_DIR / "timeseries_hs.png",
        OUTPUT_DIR / "timeseries_tp.png",
        OUTPUT_DIR / "timeseries_direction.png",
        OUTPUT_DIR / "scatter_hs.png",
        OUTPUT_DIR / "scatter_tp.png",
        OUTPUT_DIR / "direction_error.png",
    ]:
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
