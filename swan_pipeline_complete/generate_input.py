import json
import numpy as np
import xarray as xr

from config import (
    BOUNDARY_OUTPUT,
    DEPTH_OUTPUT,
    DIRECTION_BINS,
    MAX_FREQUENCY_HZ,
    MIN_FREQUENCY_HZ,
    PROJECT_NAME,
    PROJECT_NUMBER,
    PROCESSED_DIR,
    SWAN_INPUT_OUTPUT,
    WIND_FILE,
    WIND_OUTPUT,
    ensure_directories,
)

from generate_boundary import generate_boundary
from generate_depth import generate_depth
from generate_wind import generate_wind


def _find_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords or name in dataset.dims:
            return name

    raise KeyError(f"Nenhum nome encontrado entre: {candidates}")


def _format_swan_time(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="s")
    date, clock = text.split("T")

    return f"{date.replace('-', '')}.{clock.replace(':', '')}"


def _time_step_hours(times: np.ndarray) -> float:
    if times.size < 2:
        raise ValueError(
            "São necessários pelo menos dois horários para o modo não estacionário."
        )

    seconds = np.diff(
        times.astype("datetime64[s]").astype(np.int64)
    )

    if np.any(seconds <= 0):
        raise ValueError("Os horários não estão em ordem crescente.")

    first = int(seconds[0])

    if not np.all(seconds == first):
        raise ValueError(
            "O arquivo de vento possui intervalo temporal irregular."
        )

    return first / 3600.0


def _format_number(value: float) -> str:
    return f"{value:.10g}"

def _find_name(dataset, candidates):
    for n in candidates:
        if n in dataset.coords or n in dataset.dims: return n
    raise KeyError(f"Nenhum nome encontrado entre: {candidates}")

def _time(v):
    d,c=np.datetime_as_string(v,unit='s').split('T'); return f"{d.replace('-','')}.{c.replace(':','')}"

def _dt_hours(times):
    sec=np.diff(times.astype('datetime64[s]').astype(np.int64))
    if len(sec)==0 or np.any(sec<=0) or not np.all(sec==sec[0]): raise ValueError('Horários inválidos ou irregulares.')
    return sec[0]/3600.0

def fmt(v): return f"{v:.10g}"

def generate_input() -> None:
    ensure_directories()

    grid_file = PROCESSED_DIR / "grid.json"

    if not grid_file.exists() or not DEPTH_OUTPUT.exists():
        grid = generate_depth()
    else:
        grid = json.loads(
            grid_file.read_text(encoding="utf-8")
        )


    if not WIND_OUTPUT.exists():
        generate_wind()

    if not BOUNDARY_OUTPUT.exists():
        generate_boundary()


    # =============================
    # TEMPO DO ERA5
    # =============================
    with xr.open_dataset(WIND_FILE) as dataset:

        time_name = _find_name(
            dataset,
            (
                "valid_time",
                "time",
                "forecast_time",
                "datetime",
            ),
        )

        times = np.asarray(
            dataset[time_name].values
        )

        times = np.sort(times)


    start = _format_swan_time(times[0])
    end = _format_swan_time(times[-1])


    # intervalo dos arquivos ERA5
    dt_hours = _time_step_hours(times)

    input_dt_text = _format_number(
        dt_hours
    )


    # intervalo interno SWAN
    compute_dt_value = 15
    compute_dt_unit = "MIN"



    # =============================
    # GRADE SWAN
    # =============================

    x0 = _format_number(
        grid["x0"]
    )

    y0 = _format_number(
        grid["y0"]
    )

    dx = _format_number(
        grid["dx"]
    )

    dy = _format_number(
        grid["dy"]
    )

    x_length = _format_number(
        grid["x_length"]
    )

    y_length = _format_number(
        grid["y_length"]
    )


    mx = int(
        grid["mx"]
    )

    my = int(
        grid["my"]
    )

    content = f"""PROJECT '{PROJECT_NAME}' '{PROJECT_NUMBER}'

SET LEVEL 0.0 MAXERR=99 NAUTICAL

COORDINATES SPHERICAL CCM

MODE NONSTATIONARY

CGRID REGULAR {x0} {y0} 0.0 {x_length} {y_length} {mx} {my} \
CIRCLE {DIRECTION_BINS} {MIN_FREQUENCY_HZ} {MAX_FREQUENCY_HZ}

INPGRID BOTTOM REGULAR {x0} {y0} 0.0 {mx} {my} {dx} {dy}
READINP BOTTOM 1.0 'depth.bot' 3 0 FREE

INPGRID WIND REGULAR {x0} {y0} 0.0 {mx} {my} {dx} {dy} \
NONSTATIONARY {start} {input_dt_text} HR {end}
READINP WIND 1.0 'wind.txt' 3 0 FREE

BOUND SHAPESPEC JONSWAP 3.3 MEAN DSPR DEGREES

BOUNDSPEC SIDE EAST CONSTANT FILE 'boundary_east.txt' 1
BOUNDSPEC SIDE SOUTH CONSTANT FILE 'boundary_south.txt' 1

NUMERIC ACCUR 0.02 0.02 0.02 99.5 NONSTAT 5

PROP BSBT

BLOCK 'COMPGRID' NOHEAD 'output.mat' LAY 3 \
HSIGN TM01 TPS DIR WIND OUTPUT {start} {input_dt_text} HR

COMPUTE NONSTATIONARY \
{start} {compute_dt_value} {compute_dt_unit} {end}

STOP
"""
    SWAN_INPUT_OUTPUT.write_text(content,encoding='ascii')
    print(f"INPUT gerado: {SWAN_INPUT_OUTPUT}")

if __name__=='__main__':
    generate_input()
