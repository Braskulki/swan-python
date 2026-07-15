# SWAN Python Pipeline

A Python pipeline for preparing and running a **non-stationary SWAN wave simulation** using:

- **GEBCO** bathymetry
- **ERA5** wind fields
- **ERA5** wave parameters for boundary conditions
- **Docker** to run SWAN 41.51A

The pipeline generates all SWAN input files, runs the model inside a Docker container, and stores the resulting MATLAB output file.

---

## run anyfile

Open terminal on file folder:

Copy-Item .\FILENAME.swn INPUT

docker run --rm `
    -v ${PWD}:/work `
    -w /work `
    openeuler/swan:latest `
    /opt/swan/swan.exe

## Project Structure

```text
swan_pipeline_complete/
├── config.py
├── generate_depth.py
├── generate_wind.py
├── generate_boundary.py
├── generate_input.py
├── run_swan.py
├── check-output.py
└── data/
    ├── raw/
    │   ├── gebco.nc
    │   ├── wind.nc
    │   └── waves.nc
    └── processed/
        ├── depth.bot
        ├── wind.txt
        ├── boundary_east.txt
        ├── boundary_south.txt
        ├── INPUT
        ├── PRINT
        └── output.mat
```

---

## Scripts

### `config.py`

Central configuration file.

It defines:

- input and output directories
- geographic domain
- GEBCO subsampling step
- SWAN spectral settings
- Docker image name
- SWAN executable path
- project metadata

Recommended Docker defaults:

```python
SWAN_DOCKER_IMAGE = "openeuler/swan:latest"
SWAN_EXECUTABLE = "/opt/swan/swan.exe"
```

These values may also be overridden through environment variables.

### `generate_depth.py`

Reads `gebco.nc`, crops the configured geographic domain, converts GEBCO elevation into positive water depth, and writes:

```text
data/processed/depth.bot
```

It also generates:

```text
data/processed/grid.json
```

`grid.json` stores the grid origin, spacing, dimensions, longitude values, and latitude values shared by the remaining scripts.

### `generate_wind.py`

Reads ERA5 10 m wind components from:

```text
data/raw/wind.nc
```

The script:

1. interpolates `u10` and `v10` to the SWAN grid;
2. validates the resulting dimensions;
3. writes one complete U field followed by one complete V field for every timestamp.

Output:

```text
data/processed/wind.txt
```

The timestamps are defined by the SWAN `INPGRID WIND ... NONSTATIONARY` command rather than being written into `wind.txt`.

### `generate_boundary.py`

Reads ERA5 wave parameters from:

```text
data/raw/waves.nc
```

It uses:

- significant wave height
- mean wave period
- mean wave direction

The fields are interpolated to the SWAN grid and summarized along the open boundaries.

Outputs:

```text
data/processed/boundary_east.txt
data/processed/boundary_south.txt
```

Each file uses SWAN TPAR format:

```text
TPAR
20230101.000000 1.4400 8.2000 120.00 30.00
```

The columns are:

```text
timestamp  significant_wave_height  period  direction  directional_spread
```

NaN values over land are ignored. If a boundary has no valid wave cells, the script falls back to valid ocean cells from the interpolated or native ERA5 domain.

### `generate_input.py`

Builds the SWAN `INPUT` file automatically from:

- `grid.json`
- ERA5 timestamps
- generated bathymetry
- generated wind fields
- generated boundary files

Output:

```text
data/processed/INPUT
```

The generated configuration includes:

- spherical coordinates
- non-stationary mode
- GEBCO bathymetry
- spatially varying ERA5 wind
- TPAR boundaries on the east and south sides
- BSBT propagation scheme
- 15-minute computational step
- 6-hour forcing/output interval
- MATLAB block output

### `run_swan.py`

Runs the full pipeline in order:

1. generate bathymetry;
2. generate wind input;
3. generate wave boundary files;
4. generate the SWAN `INPUT` file;
5. run SWAN through Docker.

The processed directory is mounted inside the container as:

```text
/work
```

The resulting Docker command is equivalent to:

```powershell
docker run --rm `
  -v "D:\path\to\swan_pipeline_complete\data\processed:/work" `
  -w /work `
  openeuler/swan:latest `
  /opt/swan/swan.exe
```

### `check-output.py`

Reads:

```text
data/processed/output.mat
```

and reports, for each variable and timestamp:

- matrix shape
- total number of cells
- finite cells
- NaN cells
- valid percentage
- minimum
- maximum
- mean
- median
- zero count

SWAN stores dry or inactive cells as `NaN`, so NaN-aware NumPy functions must be used.

---

## Requirements

### Python

Recommended:

```text
Python 3.11+
```

Install the required packages:

```bash
pip -m install pandas cdsapi numpy xarray netcdf4 scipy matplotlib imageio

```

Optional plotting dependency:

```bash
pip install matplotlib imageio
```

### Docker

Docker Desktop must be installed and running.

Confirm that the SWAN image exists locally:

```powershell
docker images
```

Expected image:

```text
openeuler/swan:latest
```

Confirm the SWAN executable inside the image:

```powershell
docker run --rm -it `
  --entrypoint /bin/bash `
  openeuler/swan:latest
```

Inside the container:

```bash
ls -l /opt/swan/swan.exe
```

---

## Input Data

Place all source files under:

```text
data/raw/
```

### `gebco.nc`

Expected fields:

```text
lon
lat
elevation
```

### `wind.nc`

Expected ERA5 fields:

```text
u10
v10
time or valid_time
latitude
longitude
```

### `waves.nc`

Expected ERA5 fields:

```text
swh
mwp
mwd
time or valid_time
latitude
longitude
```

Variable names may vary slightly depending on the NetCDF export. The scripts include aliases for common ERA5 names.

---

## Running the Application

From the repository root:

```powershell
py .\swan_pipeline_complete\run_swan.py
```

Or enter the project directory first:

```powershell
cd .\swan_pipeline_complete
py .\run_swan.py
```

Generated files are stored in:

```text
data/processed/
```

The main result files are:

```text
PRINT
output.mat
```

---

## Docker Configuration

The Docker image and executable may be configured with environment variables.

PowerShell:

```powershell
$env:SWAN_DOCKER_IMAGE="openeuler/swan:latest"
$env:SWAN_EXECUTABLE="/opt/swan/swan.exe"

py .\swan_pipeline_complete\run_swan.py
```

Linux or WSL:

```bash
export SWAN_DOCKER_IMAGE="openeuler/swan:latest"
export SWAN_EXECUTABLE="/opt/swan/swan.exe"

python swan_pipeline_complete/run_swan.py
```

---

## Checking the Results

Run:

```powershell
py .\swan_pipeline_complete\check-output.py
```

The MATLAB file contains variables named by parameter and timestamp, for example:

```text
Hsig_20230103_180000
Tm01_20230103_180000
TPsmoo_20230103_180000
Dir_20230103_180000
Windv_x_20230103_180000
Windv_y_20230103_180000
```

| Variable | Description |
|---|---|
| `Hsig` | Significant wave height |
| `Tm01` | Mean wave period |
| `TPsmoo` | Smoothed peak period |
| `Dir` | Mean wave direction |
| `Windv_x` | X component of wind |
| `Windv_y` | Y component of wind |

---

## Spin-Up Period

The first output timestep should normally be treated as a **spin-up timestep**.

At the beginning of a non-stationary simulation, SWAN may contain little or no wave energy inside the domain, even though boundary spectra are already imposed. Wave energy needs time to propagate from the open boundaries into the computational grid.

For this reason, analyses should normally start from the second output timestep:

```python
hs_keys = sorted(
    key
    for key in data
    if key.startswith("Hsig_")
)

hs_keys_without_spinup = hs_keys[1:]
```

For scientific applications, a better approach is to start the simulation 12 to 24 hours before the period of interest and discard the full warm-up interval.

---

## Notes

- `output.mat` is a MATLAB binary file, not a plain text matrix.
- Use `scipy.io.loadmat` to read it.
- `NaN` values normally represent land or dry cells.
- The first timestep may not represent a fully developed wave field.
- Boundary conditions are spatially constant along each boundary side but vary over time.
- The current configuration applies wave boundaries to the east and south sides.
- The computational timestep is 15 minutes.
- ERA5 forcing and SWAN output are configured at 6-hour intervals.

---

## Troubleshooting

### Docker image not found

```text
Unable to find image ... locally
```

Check local images:

```powershell
docker images
```

Then update `SWAN_DOCKER_IMAGE`.

### SWAN executable not found

```text
exec: "swan": executable file not found in $PATH
```

Use the full executable path:

```text
/opt/swan/swan.exe
```

### `output.mat` cannot be read with `numpy.loadtxt`

Use:

```python
from scipy.io import loadmat

data = loadmat("data/processed/output.mat")
```

### Minimum and maximum are shown as NaN

Use NaN-aware NumPy functions:

```python
np.nanmin(values)
np.nanmax(values)
np.nanmean(values)
```

### Boundary warnings during the first hours

This is often related to model spin-up. Verify that the differences decrease over time and exclude the warm-up interval from the final analysis.

---

## License

Add the license appropriate for your project here.