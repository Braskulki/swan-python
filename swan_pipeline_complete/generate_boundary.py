import json
from pathlib import Path

import numpy as np
import xarray as xr

from config import (
    BOUNDARY_OUTPUT,
    PROCESSED_DIR,
    WAVES_FILE,
    ensure_directories,
)
from generate_depth import generate_depth


DIRECTIONAL_SPREAD_DEGREES = 30.0

EAST_BOUNDARY_OUTPUT = PROCESSED_DIR / "boundary_east.txt"
SOUTH_BOUNDARY_OUTPUT = PROCESSED_DIR / "boundary_south.txt"
METADATA_OUTPUT = PROCESSED_DIR / "boundary_metadata.json"


def _find_name(
    dataset: xr.Dataset,
    candidates: tuple[str, ...],
) -> str:
    """
    Retorna o primeiro nome encontrado no Dataset.

    Procura em variáveis, coordenadas e dimensões.
    """
    for name in candidates:
        if (
            name in dataset.variables
            or name in dataset.coords
            or name in dataset.dims
        ):
            return name

    raise KeyError(
        f"Nenhum nome encontrado entre: {candidates}. "
        f"Disponíveis: {list(dataset.variables)}"
    )


def _load_grid() -> dict:
    """
    Carrega grid.json. Caso ainda não exista, gera a batimetria
    e os metadados da grade.
    """
    grid_file = PROCESSED_DIR / "grid.json"

    if not grid_file.exists():
        return generate_depth()

    return json.loads(
        grid_file.read_text(encoding="utf-8")
    )


def _format_swan_time(value: np.datetime64) -> str:
    """
    Converte numpy.datetime64 para o formato ISO usado pelo SWAN:

        YYYYMMDD.HHMMSS
    """
    text = np.datetime_as_string(value, unit="s")

    if "T" not in text:
        raise ValueError(
            f"Data inválida ou inesperada: {text}"
        )

    date, clock = text.split("T")

    return (
        f"{date.replace('-', '')}."
        f"{clock.replace(':', '')}"
    )


def _circular_mean_degrees(
    values: np.ndarray,
) -> float:
    """
    Calcula a média circular de direções em graus.

    Evita o problema da média aritmética entre, por exemplo,
    359° e 1°, que deveria resultar próximo de 0°.
    """
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]

    if valid.size == 0:
        raise ValueError(
            "Não há direções válidas para calcular a média circular."
        )

    radians = np.deg2rad(valid)

    mean_sine = np.mean(np.sin(radians))
    mean_cosine = np.mean(np.cos(radians))

    direction = np.rad2deg(
        np.arctan2(mean_sine, mean_cosine)
    )

    return float(direction % 360.0)


def _valid_wave_mask(
    hs: np.ndarray,
    period: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """
    Retorna máscara para valores de ondas fisicamente utilizáveis.
    """
    return (
        np.isfinite(hs)
        & np.isfinite(period)
        & np.isfinite(direction)
        & (hs >= 0.0)
        & (period > 0.0)
    )


def _summarize_wave_values(
    hs: np.ndarray,
    period: np.ndarray,
    direction: np.ndarray,
) -> tuple[float, float, float, int]:
    """
    Resume um conjunto de pontos de onda em um único estado TPAR.

    Retorna:
        Hs médio,
        período médio,
        direção média circular,
        quantidade de pontos válidos.
    """
    hs_array = np.asarray(hs, dtype=float).reshape(-1)
    period_array = np.asarray(period, dtype=float).reshape(-1)
    direction_array = np.asarray(
        direction,
        dtype=float,
    ).reshape(-1)

    mask = _valid_wave_mask(
        hs_array,
        period_array,
        direction_array,
    )

    valid_count = int(np.count_nonzero(mask))

    if valid_count == 0:
        raise ValueError(
            "Nenhum valor válido de onda foi encontrado."
        )

    hs_mean = float(np.mean(hs_array[mask]))
    period_mean = float(np.mean(period_array[mask]))
    direction_mean = _circular_mean_degrees(
        direction_array[mask]
    )

    # Evita valores nulos que podem causar um espectro degenerado.
    hs_mean = max(hs_mean, 0.01)
    period_mean = max(period_mean, 0.1)

    return (
        hs_mean,
        period_mean,
        direction_mean,
        valid_count,
    )


def _extract_summary_with_fallback(
    hs_side: np.ndarray,
    period_side: np.ndarray,
    direction_side: np.ndarray,
    hs_domain: np.ndarray,
    period_domain: np.ndarray,
    direction_domain: np.ndarray,
    hs_native: np.ndarray,
    period_native: np.ndarray,
    direction_native: np.ndarray,
) -> tuple[float, float, float, int, str]:
    """
    Tenta resumir os valores na seguinte ordem:

    1. lado interpolado;
    2. domínio interpolado;
    3. domínio nativo do ERA5.

    Isso é necessário porque o ERA5 pode ter NaN sobre terra.
    """
    candidates = (
        (
            hs_side,
            period_side,
            direction_side,
            "interpolated_side",
        ),
        (
            hs_domain,
            period_domain,
            direction_domain,
            "interpolated_domain",
        ),
        (
            hs_native,
            period_native,
            direction_native,
            "native_era5_domain",
        ),
    )

    errors: list[str] = []

    for (
        candidate_hs,
        candidate_period,
        candidate_direction,
        source,
    ) in candidates:
        try:
            (
                hs_mean,
                period_mean,
                direction_mean,
                valid_count,
            ) = _summarize_wave_values(
                candidate_hs,
                candidate_period,
                candidate_direction,
            )

            return (
                hs_mean,
                period_mean,
                direction_mean,
                valid_count,
                source,
            )

        except ValueError as exc:
            errors.append(f"{source}: {exc}")

    raise ValueError(
        "Não foi possível obter valores de onda válidos. "
        + " | ".join(errors)
    )


def _build_tpar(
    times: np.ndarray,
    hs_values: np.ndarray,
    period_values: np.ndarray,
    direction_values: np.ndarray,
    native_hs_values: np.ndarray,
    native_period_values: np.ndarray,
    native_direction_values: np.ndarray,
    side: str,
) -> tuple[str, list[dict]]:
    """
    Constrói o conteúdo TPAR para EAST ou SOUTH.
    """
    lines = ["TPAR"]
    records: list[dict] = []

    for time_index, time_value in enumerate(times):
        hs_domain = hs_values[time_index]
        period_domain = period_values[time_index]
        direction_domain = direction_values[time_index]

        if side == "east":
            # Longitude máxima da grade.
            hs_side = hs_domain[:, -1]
            period_side = period_domain[:, -1]
            direction_side = direction_domain[:, -1]

        elif side == "south":
            # Latitude mínima, pois a grade foi ordenada
            # crescentemente em latitude.
            hs_side = hs_domain[0, :]
            period_side = period_domain[0, :]
            direction_side = direction_domain[0, :]

        else:
            raise ValueError(
                f"Lado de contorno não suportado: {side}"
            )

        (
            hs_mean,
            period_mean,
            direction_mean,
            valid_count,
            source,
        ) = _extract_summary_with_fallback(
            hs_side=hs_side,
            period_side=period_side,
            direction_side=direction_side,
            hs_domain=hs_domain,
            period_domain=period_domain,
            direction_domain=direction_domain,
            hs_native=native_hs_values[time_index],
            period_native=native_period_values[time_index],
            direction_native=native_direction_values[time_index],
        )

        swan_time = _format_swan_time(time_value)

        lines.append(
            f"{swan_time} "
            f"{hs_mean:.4f} "
            f"{period_mean:.4f} "
            f"{direction_mean:.2f} "
            f"{DIRECTIONAL_SPREAD_DEGREES:.2f}"
        )

        records.append(
            {
                "time": np.datetime_as_string(
                    time_value,
                    unit="s",
                ),
                "swan_time": swan_time,
                "hs_m": hs_mean,
                "period_s": period_mean,
                "direction_degrees": direction_mean,
                "directional_spread_degrees": (
                    DIRECTIONAL_SPREAD_DEGREES
                ),
                "valid_points": valid_count,
                "source": source,
            }
        )

    return "\n".join(lines) + "\n", records


def generate_boundary() -> None:
    """
    Gera condições de contorno TPAR nos lados leste e sul.
    """
    ensure_directories()

    if not WAVES_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo ERA5 de ondas não encontrado: "
            f"{WAVES_FILE}"
        )

    grid = _load_grid()

    target_lon = np.asarray(
        grid["lon"],
        dtype=float,
    )
    target_lat = np.asarray(
        grid["lat"],
        dtype=float,
    )

    if target_lon.size < 2 or target_lat.size < 2:
        raise ValueError(
            "A grade SWAN precisa ter pelo menos dois pontos "
            "em longitude e latitude."
        )

    with xr.open_dataset(WAVES_FILE) as dataset:
        lon_name = _find_name(
            dataset,
            ("longitude", "lon"),
        )
        lat_name = _find_name(
            dataset,
            ("latitude", "lat"),
        )
        time_name = _find_name(
            dataset,
            (
                "valid_time",
                "time",
                "forecast_time",
                "datetime",
            ),
        )

        hs_name = _find_name(
            dataset,
            (
                "swh",
                "significant_height_of_combined_wind_waves_and_swell",
            ),
        )
        direction_name = _find_name(
            dataset,
            (
                "mwd",
                "mean_wave_direction",
            ),
        )
        period_name = _find_name(
            dataset,
            (
                "mwp",
                "mean_wave_period",
            ),
        )

        # O xarray interp funciona de forma mais previsível
        # com coordenadas crescentes.
        dataset = (
            dataset
            .sortby(lon_name)
            .sortby(lat_name)
            .sortby(time_name)
        )

        hs_native = dataset[hs_name].squeeze(drop=True)
        direction_native = dataset[
            direction_name
        ].squeeze(drop=True)
        period_native = dataset[
            period_name
        ].squeeze(drop=True)

        required_dims = {
            time_name,
            lat_name,
            lon_name,
        }

        for variable_name, variable in (
            (hs_name, hs_native),
            (direction_name, direction_native),
            (period_name, period_native),
        ):
            missing = required_dims.difference(
                variable.dims
            )

            if missing:
                raise ValueError(
                    f"A variável {variable_name!r} não possui "
                    f"as dimensões necessárias: {missing}. "
                    f"Dimensões encontradas: {variable.dims}"
                )

        interpolation_coordinates = {
            lon_name: xr.DataArray(
                target_lon,
                dims=(lon_name,),
            ),
            lat_name: xr.DataArray(
                target_lat,
                dims=(lat_name,),
            ),
        }

        hs_interp = hs_native.interp(
            interpolation_coordinates,
            method="linear",
        ).transpose(
            time_name,
            lat_name,
            lon_name,
        )

        period_interp = period_native.interp(
            interpolation_coordinates,
            method="linear",
        ).transpose(
            time_name,
            lat_name,
            lon_name,
        )

        # Direções não devem ser interpoladas diretamente,
        # por causa da descontinuidade em 0°/360°.
        direction_radians = np.deg2rad(
            direction_native
        )

        direction_sine = np.sin(
            direction_radians
        ).interp(
            interpolation_coordinates,
            method="linear",
        )

        direction_cosine = np.cos(
            direction_radians
        ).interp(
            interpolation_coordinates,
            method="linear",
        )

        direction_interp = (
            np.rad2deg(
                np.arctan2(
                    direction_sine,
                    direction_cosine,
                )
            )
            % 360.0
        ).transpose(
            time_name,
            lat_name,
            lon_name,
        )

        times = np.asarray(
            dataset[time_name].values
        )

        hs_values = np.asarray(
            hs_interp.values,
            dtype=float,
        )
        period_values = np.asarray(
            period_interp.values,
            dtype=float,
        )
        direction_values = np.asarray(
            direction_interp.values,
            dtype=float,
        )

        native_hs_values = np.asarray(
            hs_native.transpose(
                time_name,
                lat_name,
                lon_name,
            ).values,
            dtype=float,
        )
        native_period_values = np.asarray(
            period_native.transpose(
                time_name,
                lat_name,
                lon_name,
            ).values,
            dtype=float,
        )
        native_direction_values = np.asarray(
            direction_native.transpose(
                time_name,
                lat_name,
                lon_name,
            ).values,
            dtype=float,
        )

    expected_shape = (
        times.size,
        int(grid["ny"]),
        int(grid["nx"]),
    )

    for name, values in (
        ("Hs", hs_values),
        ("período", period_values),
        ("direção", direction_values),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"Forma interpolada inesperada para {name}. "
                f"Esperado: {expected_shape}; "
                f"encontrado: {values.shape}"
            )

    east_content, east_records = _build_tpar(
        times=times,
        hs_values=hs_values,
        period_values=period_values,
        direction_values=direction_values,
        native_hs_values=native_hs_values,
        native_period_values=native_period_values,
        native_direction_values=native_direction_values,
        side="east",
    )

    south_content, south_records = _build_tpar(
        times=times,
        hs_values=hs_values,
        period_values=period_values,
        direction_values=direction_values,
        native_hs_values=native_hs_values,
        native_period_values=native_period_values,
        native_direction_values=native_direction_values,
        side="south",
    )

    EAST_BOUNDARY_OUTPUT.write_text(
        east_content,
        encoding="ascii",
        newline="\n",
    )

    SOUTH_BOUNDARY_OUTPUT.write_text(
        south_content,
        encoding="ascii",
        newline="\n",
    )

    # Mantém boundary.txt por compatibilidade com scripts antigos.
    # Ele recebe o mesmo conteúdo do lado leste.
    BOUNDARY_OUTPUT.write_text(
        east_content,
        encoding="ascii",
        newline="\n",
    )

    metadata = {
        "time_count": int(times.size),
        "source_file": str(WAVES_FILE),
        "variables": {
            "significant_wave_height": hs_name,
            "wave_period": period_name,
            "wave_direction": direction_name,
        },
        "directional_spread_degrees": (
            DIRECTIONAL_SPREAD_DEGREES
        ),
        "boundary_files": {
            "east": str(EAST_BOUNDARY_OUTPUT),
            "south": str(SOUTH_BOUNDARY_OUTPUT),
            "legacy": str(BOUNDARY_OUTPUT),
        },
        "east_records": east_records,
        "south_records": south_records,
    }

    METADATA_OUTPUT.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Condição de contorno leste gerada: "
        f"{EAST_BOUNDARY_OUTPUT}"
    )
    print(
        f"Condição de contorno sul gerada: "
        f"{SOUTH_BOUNDARY_OUTPUT}"
    )
    print(
        f"Arquivo de compatibilidade gerado: "
        f"{BOUNDARY_OUTPUT}"
    )
    print(
        f"Quantidade de horários: {times.size}"
    )
    print(
        f"Metadados: {METADATA_OUTPUT}"
    )


if __name__ == "__main__":
    generate_boundary()