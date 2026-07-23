from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
GEBCO_FILE = RAW_DIR / "gebco.nc"
WIND_FILE = RAW_DIR / "wind.nc"
WAVES_FILE = RAW_DIR / "waves.nc"
DEPTH_OUTPUT = PROCESSED_DIR / "depth.bot"
WIND_OUTPUT = PROCESSED_DIR / "wind.txt"
BOUNDARY_OUTPUT = PROCESSED_DIR / "boundary.txt"
SWAN_INPUT_OUTPUT = PROCESSED_DIR / "INPUT"
BASE_DIR = Path(__file__).resolve().parent
UNSTRUCTURED_DIR = BASE_DIR / "data" / "unstructured_research"
UNSTRUCTURED_CASE_DIR = UNSTRUCTURED_DIR
UNSTRUCTURED_PUBLICATION_DIR = UNSTRUCTURED_DIR / "publication"
ERA5_YEAR = "2023"
ERA5_MONTH = "01"
ERA5_DAYS = [
    "01",
    "02",
    "03",
]

ERA5_TIMES = [
    "00:00",
    "06:00",
    "12:00",
    "18:00",
]
## LON_MIN = -48.12
## LON_MAX = -44.65
## LAT_MIN = -25.36
## LAT_MAX = -23.18
LON_MIN = -36.80
LON_MAX = -34.80

LAT_MIN = -10.70
LAT_MAX = -8.30
ERA5_MARGIN_DEGREES = 0.30
GRID_DX = 0.025
GRID_DY = 0.025
GEBCO_STEP = 9
DIRECTION_BINS = 36
MIN_FREQUENCY_HZ = 0.04
MAX_FREQUENCY_HZ = 1.0
SWAN_DOCKER_IMAGE = os.getenv(
    "SWAN_DOCKER_IMAGE",
    "openeuler/swan:latest",
)

SWAN_EXECUTABLE = os.getenv(
    "SWAN_EXECUTABLE",
    "/opt/swan/swan.exe",
)
DOCKER_PLATFORM = os.getenv("SWAN_DOCKER_PLATFORM", "")
PROJECT_NAME = "TEST"
PROJECT_NUMBER = "01"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
