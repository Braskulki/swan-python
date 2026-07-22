from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


# ============================================================
# CONFIGURAÇÕES
# ============================================================

XYZ_FILES = [
    "NORTE.xyz",
    "CENTRO.xyz",
    "SUL.xyz",
]

UNIFIED_XYZ_FILE = "COSTA_COMPLETA.xyz"
OUTPUT_IMAGE = "contornos_costa_completa.png"

# True: usa automaticamente os limites dos três arquivos.
# False: utiliza os limites definidos manualmente abaixo.
USE_AUTOMATIC_LIMITS = True

LON_MIN = -38.0
LON_MAX = -34.0
LAT_MIN = -13.0
LAT_MAX = -5.0

# Resolução da imagem interpolada.
#
# Para um domínio grande, comece com 1000 x 1400.
# Aumentar demais consome bastante memória.
NX = 1000
NY = 1400

CONTOUR_LEVELS = [
    5,
    10,
    20,
    30,
    50,
    100,
    200,
    500,
    1000,
    2000,
    3000,
    4000,
]

# Distância máxima entre uma célula e um ponto real do XYZ.
# Células mais distantes são tratadas como sem dados.
#
# Este valor é em graus:
# 0.025° corresponde aproximadamente a 2,7 km.
MAX_DISTANCE_TO_DATA = 0.025

# Limite usado para identificar mar:
# profundidade > 0 = oceano
OCEAN_THRESHOLD = 0.0

# Fração mínima de pontos oceânicos ao redor da célula.
OCEAN_FRACTION_THRESHOLD = 0.60

# Suavização leve das linhas.
# Use 0 para desativar.
SMOOTH_SIGMA = 0.8

# Número de casas decimais usado para identificar coordenadas
# repetidas nas áreas de sobreposição.
COORDINATE_DECIMALS = 6

# Caso os arquivos sejam muito densos, você pode reduzir os pontos.
# 1 mantém todos; 2 usa um de cada dois; 3 usa um de cada três etc.
POINT_STEP = 1


# ============================================================
# FUNÇÃO DE LEITURA
# ============================================================

def read_xyz(filename: str) -> pd.DataFrame:
    """
    Lê um arquivo XYZ no formato:

        longitude latitude profundidade

    O separador pode ser espaço ou tabulação.
    """

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path.resolve()}"
        )

    data = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "depth"],
        usecols=[0, 1, 2],
        comment="#",
        dtype=np.float64,
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["lon", "lat", "depth"])

    data["source"] = path.stem

    print(
        f"{filename}: {len(data):,} pontos | "
        f"lon {data['lon'].min():.4f} a {data['lon'].max():.4f} | "
        f"lat {data['lat'].min():.4f} a {data['lat'].max():.4f} | "
        f"profundidade {data['depth'].min():.2f} a "
        f"{data['depth'].max():.2f} m"
    )

    return data


# ============================================================
# LEITURA E UNIÃO DOS ARQUIVOS
# ============================================================

datasets = [
    read_xyz(filename)
    for filename in XYZ_FILES
]

all_data = pd.concat(
    datasets,
    ignore_index=True,
)

print()
print(f"Total antes da unificação: {len(all_data):,} pontos")


# ============================================================
# TRATAMENTO DE PONTOS DUPLICADOS
# ============================================================

# Arredondar as coordenadas permite localizar pontos coincidentes
# mesmo quando há pequenas diferenças numéricas entre os arquivos.

all_data["lon_key"] = all_data["lon"].round(
    COORDINATE_DECIMALS
)

all_data["lat_key"] = all_data["lat"].round(
    COORDINATE_DECIMALS
)

# Para coordenadas repetidas:
#
# - longitude e latitude: média;
# - profundidade: mediana.
#
# A mediana é mais resistente a valores anômalos nas áreas de
# sobreposição entre NORTE, CENTRO e SUL.

unified = (
    all_data
    .groupby(
        ["lon_key", "lat_key"],
        as_index=False,
        sort=False,
    )
    .agg(
        lon=("lon", "mean"),
        lat=("lat", "mean"),
        depth=("depth", "median"),
    )
)

unified = unified[
    ["lon", "lat", "depth"]
].copy()

unified = unified.sort_values(
    ["lat", "lon"],
    ascending=[True, True],
).reset_index(drop=True)

print(f"Total depois da unificação: {len(unified):,} pontos")
print(
    f"Duplicados removidos: "
    f"{len(all_data) - len(unified):,}"
)


# ============================================================
# SALVA O XYZ UNIFICADO
# ============================================================

unified.to_csv(
    UNIFIED_XYZ_FILE,
    sep=" ",
    header=False,
    index=False,
    float_format="%.8f",
)

print(f"Arquivo unificado salvo em: {UNIFIED_XYZ_FILE}")


# ============================================================
# LIMITES DO DOMÍNIO
# ============================================================

if USE_AUTOMATIC_LIMITS:
    lon_min = float(unified["lon"].min())
    lon_max = float(unified["lon"].max())
    lat_min = float(unified["lat"].min())
    lat_max = float(unified["lat"].max())
else:
    lon_min = LON_MIN
    lon_max = LON_MAX
    lat_min = LAT_MIN
    lat_max = LAT_MAX

    unified = unified[
        (unified["lon"] >= lon_min)
        & (unified["lon"] <= lon_max)
        & (unified["lat"] >= lat_min)
        & (unified["lat"] <= lat_max)
    ].copy()

if unified.empty:
    raise ValueError(
        "Nenhum ponto ficou dentro dos limites definidos."
    )

print()
print("Limites utilizados:")
print(f"  Longitude: {lon_min:.4f} até {lon_max:.4f}")
print(f"  Latitude:  {lat_min:.4f} até {lat_max:.4f}")


# ============================================================
# REDUÇÃO OPCIONAL DOS PONTOS
# ============================================================

if POINT_STEP > 1:
    interpolation_data = unified.iloc[
        ::POINT_STEP
    ].copy()
else:
    interpolation_data = unified.copy()

print(
    f"Pontos utilizados na interpolação: "
    f"{len(interpolation_data):,}"
)


# ============================================================
# IDENTIFICAÇÃO DE MAR E TERRA
# ============================================================

# Mantemos os pontos de terra durante a interpolação.
#
# Removê-los antes da interpolação faria o griddata criar
# profundidades falsas sobre o continente.

interpolation_data["is_ocean"] = (
    interpolation_data["depth"] > OCEAN_THRESHOLD
)

ocean_count = int(
    interpolation_data["is_ocean"].sum()
)

land_count = (
    len(interpolation_data) - ocean_count
)

print(f"Pontos oceânicos: {ocean_count:,}")
print(f"Pontos terrestres/sem batimetria: {land_count:,}")

if ocean_count == 0:
    raise ValueError(
        "Nenhuma profundidade positiva foi encontrada."
    )


# ============================================================
# CRIAÇÃO DA GRADE
# ============================================================

grid_lon = np.linspace(
    lon_min,
    lon_max,
    NX,
)

grid_lat = np.linspace(
    lat_min,
    lat_max,
    NY,
)

lon_grid, lat_grid = np.meshgrid(
    grid_lon,
    grid_lat,
)

source_coordinates = interpolation_data[
    ["lon", "lat"]
].to_numpy()

source_depth = interpolation_data[
    "depth"
].to_numpy()


# ============================================================
# INTERPOLAÇÃO DA PROFUNDIDADE
# ============================================================

print()
print("Interpolando a batimetria...")

depth_grid = griddata(
    points=source_coordinates,
    values=source_depth,
    xi=(lon_grid, lat_grid),
    method="linear",
)


# ============================================================
# MÁSCARA DE OCEANO
# ============================================================

print("Criando máscara de oceano e continente...")

ocean_indicator = (
    interpolation_data["is_ocean"]
    .astype(np.float64)
    .to_numpy()
)

ocean_fraction = griddata(
    points=source_coordinates,
    values=ocean_indicator,
    xi=(lon_grid, lat_grid),
    method="linear",
)

ocean_mask = (
    ocean_fraction
    >= OCEAN_FRACTION_THRESHOLD
)


# ============================================================
# MÁSCARA DE DISTÂNCIA
# ============================================================

# A triangulação Delaunay usada pelo griddata pode ligar pontos
# muito distantes e criar grandes polígonos artificiais.
#
# Esta máscara mantém apenas células próximas de um ponto real.

print("Calculando distância até os dados originais...")

tree = cKDTree(source_coordinates)

grid_coordinates = np.column_stack(
    [
        lon_grid.ravel(),
        lat_grid.ravel(),
    ]
)

distance_to_data, _ = tree.query(
    grid_coordinates,
    k=1,
    workers=-1,
)

distance_to_data = distance_to_data.reshape(
    lon_grid.shape
)

near_data_mask = (
    distance_to_data
    <= MAX_DISTANCE_TO_DATA
)


# ============================================================
# MÁSCARA FINAL
# ============================================================

valid_mask = (
    ocean_mask
    & near_data_mask
    & np.isfinite(depth_grid)
    & (depth_grid > OCEAN_THRESHOLD)
)

depth_masked = np.where(
    valid_mask,
    depth_grid,
    np.nan,
)


# ============================================================
# SUAVIZAÇÃO PRESERVANDO A LINHA DE COSTA
# ============================================================

if SMOOTH_SIGMA > 0:
    print("Aplicando suavização leve...")

    valid_cells = np.isfinite(
        depth_masked
    )

    depth_values = np.where(
        valid_cells,
        depth_masked,
        0.0,
    )

    weights = valid_cells.astype(
        np.float64
    )

    filtered_depth = gaussian_filter(
        depth_values,
        sigma=SMOOTH_SIGMA,
    )

    filtered_weights = gaussian_filter(
        weights,
        sigma=SMOOTH_SIGMA,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        smoothed_depth = (
            filtered_depth
            / filtered_weights
        )

    depth_masked = np.where(
        valid_mask
        & (filtered_weights > 0.20),
        smoothed_depth,
        np.nan,
    )


# ============================================================
# NÍVEIS DISPONÍVEIS
# ============================================================

if not np.any(np.isfinite(depth_masked)):
    raise ValueError(
        "Nenhuma célula válida restou após a aplicação das máscaras. "
        "Tente aumentar MAX_DISTANCE_TO_DATA."
    )

maximum_depth = float(
    np.nanmax(depth_masked)
)

levels = [
    level
    for level in CONTOUR_LEVELS
    if level <= maximum_depth
]

if not levels:
    raise ValueError(
        "Nenhum nível de contorno está dentro do intervalo "
        "de profundidades."
    )

print(f"Profundidade máxima no mapa: {maximum_depth:.2f} m")
print(f"Níveis desenhados: {levels}")


# ============================================================
# GERAÇÃO DA IMAGEM
# ============================================================

print("Gerando a imagem...")

fig, ax = plt.subplots(
    figsize=(11, 15),
    dpi=150,
)

contours = ax.contour(
    lon_grid,
    lat_grid,
    depth_masked,
    levels=levels,
    cmap="viridis_r",
    linewidths=1.05,
)

ax.clabel(
    contours,
    inline=True,
    inline_spacing=7,
    fontsize=8,
    fmt=lambda value: f"{value:g} m",
)

colorbar = fig.colorbar(
    contours,
    ax=ax,
    pad=0.025,
    shrink=0.90,
)

colorbar.set_label(
    "Profundidade (m)",
    fontsize=11,
)

colorbar.set_ticks(levels)

ax.set_xlim(
    lon_min,
    lon_max,
)

ax.set_ylim(
    lat_min,
    lat_max,
)

ax.set_title(
    "Contornos batimétricos da costa completa",
    fontsize=18,
    pad=18,
)

ax.set_xlabel(
    "Longitude (°)",
    fontsize=12,
)

ax.set_ylabel(
    "Latitude (°)",
    fontsize=12,
)

ax.grid(
    visible=True,
    linestyle="--",
    linewidth=0.55,
    alpha=0.30,
)

# Corrige visualmente a redução da distância longitudinal
# conforme a latitude.
mean_latitude = (
    lat_min + lat_max
) / 2.0

ax.set_aspect(
    1.0 / np.cos(
        np.radians(mean_latitude)
    ),
    adjustable="box",
)

fig.text(
    0.5,
    0.025,
    "Contornos derivados de NORTE.xyz, CENTRO.xyz e SUL.xyz",
    ha="center",
    fontsize=9,
    color="dimgray",
)

plt.tight_layout(
    rect=[0, 0.04, 1, 1]
)

plt.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print(f"Imagem salva em: {OUTPUT_IMAGE}")