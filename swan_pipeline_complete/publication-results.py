"""
Publication-ready SWAN post-processing.

Inputs:
    data/processed/output.mat
    data/processed/grid.json
    data/processed/depth.bot

Outputs:
    data/publication/
    ├── figures/
    │   ├── wave/
    │   ├── wind/
    │   └── scalar/
    ├── animations/
    ├── csv/
    └── summary.csv

Main products:
    1. Significant wave height with wave-direction vectors
    2. Wind speed with wind vectors
    3. Scalar maps for Hsig, Tm01, TPsmoo and Dir
    4. PNG and PDF figures
    5. GIF animations
    6. Time-series CSV files

Wave-direction convention:
    The SWAN input uses NAUTICAL convention. SWAN wave directions are treated
    here as directions FROM which waves arrive. For propagation arrows, 180°
    is added so arrows point TOWARD the propagation direction.

Usage:
    py publication-results.py
    py publication-results.py --skip-first
    py publication-results.py --skip-first --vector-step 6
    py publication-results.py --no-gif
    py publication-results.py --formats png pdf

py .\swan_pipeline_complete\publication-results.py `
  --skip-first `
  --vector-step 7 `
  --dpi 300 `
  --formats png pdf
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PUBLICATION_DIR = BASE_DIR / "data" / "publication"

MAT_FILE = PROCESSED_DIR / "output.mat"
GRID_FILE = PROCESSED_DIR / "grid.json"
DEPTH_FILE = PROCESSED_DIR / "depth.bot"

FIGURES_DIR = PUBLICATION_DIR / "figures"
ANIMATIONS_DIR = PUBLICATION_DIR / "animations"
CSV_DIR = PUBLICATION_DIR / "csv"

KEY_PATTERN = re.compile(
    r"^(?P<variable>.+)_(?P<timestamp>\d{8}_\d{6})$"
)

SCALAR_CONFIG = {
    "Hsig": {
        "label": "Significant wave height",
        "unit": "m",
    },
    "Tm01": {
        "label": "Mean wave period",
        "unit": "s",
    },
    "TPsmoo": {
        "label": "Smoothed peak period",
        "unit": "s",
    },
    "Dir": {
        "label": "Mean wave direction",
        "unit": "°",
    },
}


@dataclass(frozen=True)
class Grid:
    longitude: np.ndarray
    latitude: np.ndarray
    depth: np.ndarray

    @property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.longitude, self.latitude)

    @property
    def wet_mask(self) -> np.ndarray:
        return np.isfinite(self.depth) & (self.depth > 0)

    @property
    def land_mask(self) -> np.ndarray:
        return ~self.wet_mask


@dataclass(frozen=True)
class Field:
    variable: str
    timestamp_text: str
    timestamp: datetime
    values: np.ndarray
    key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready SWAN maps and animations."
    )

    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="Skip the first timestep as model spin-up.",
    )

    parser.add_argument(
        "--vector-step",
        type=int,
        default=5,
        help="Plot one vector every N grid points. Default: 5.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster figure resolution. Default: 300 DPI.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Figure formats. Default: png pdf.",
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Disable GIF generation.",
    )

    parser.add_argument(
        "--gif-duration",
        type=float,
        default=0.8,
        help="Duration of each GIF frame in seconds.",
    )

    parser.add_argument(
        "--wave-arrow-mode",
        choices=("toward", "from"),
        default="toward",
        help=(
            "Direction represented by wave arrows. "
            "'toward' shows propagation; 'from' shows SWAN nautical direction."
        ),
    )

    return parser.parse_args()


def ensure_files_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]

    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Required files were not found:\n{listing}"
        )


def load_grid() -> Grid:
    metadata = json.loads(
        GRID_FILE.read_text(encoding="utf-8")
    )

    longitude = np.asarray(metadata["lon"], dtype=np.float64)
    latitude = np.asarray(metadata["lat"], dtype=np.float64)
    depth = np.loadtxt(DEPTH_FILE, dtype=np.float64)

    expected_shape = (latitude.size, longitude.size)

    if depth.shape != expected_shape:
        raise ValueError(
            f"Depth shape {depth.shape} does not match grid {expected_shape}."
        )

    return Grid(
        longitude=longitude,
        latitude=latitude,
        depth=depth,
    )


def safe_array(raw_values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        values = np.array(
            raw_values,
            dtype=np.float64,
            copy=True,
        )

    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1.0e20] = np.nan

    return values


def load_fields() -> dict[str, dict[str, Field]]:
    raw = loadmat(MAT_FILE)
    fields: dict[str, dict[str, Field]] = {}

    for key, raw_values in raw.items():
        if key.startswith("__"):
            continue

        match = KEY_PATTERN.match(key)

        if not match:
            continue

        variable = match.group("variable")
        timestamp_text = match.group("timestamp")
        timestamp = datetime.strptime(
            timestamp_text,
            "%Y%m%d_%H%M%S",
        )

        fields.setdefault(variable, {})[timestamp_text] = Field(
            variable=variable,
            timestamp_text=timestamp_text,
            timestamp=timestamp,
            values=safe_array(raw_values),
            key=key,
        )

    return fields


def validate_shapes(
    fields: dict[str, dict[str, Field]],
    grid: Grid,
) -> None:
    expected = grid.depth.shape

    for variable_fields in fields.values():
        for field in variable_fields.values():
            if field.values.shape != expected:
                raise ValueError(
                    f"{field.key}: shape {field.values.shape}; expected {expected}."
                )


def mask_field(values: np.ndarray, grid: Grid) -> np.ndarray:
    result = values.astype(np.float64, copy=True)
    result[grid.land_mask] = np.nan
    return result


def statistics(values: np.ndarray) -> dict[str, float | int]:
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return {
            "valid_points": 0,
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "standard_deviation": np.nan,
        }

    return {
        "valid_points": int(valid.size),
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "standard_deviation": float(np.std(valid)),
    }


def robust_limits(
    fields: list[Field],
    grid: Grid,
    variable: str,
) -> tuple[float, float]:
    chunks: list[np.ndarray] = []

    for field in fields:
        values = mask_field(field.values, grid)
        valid = values[np.isfinite(values)]

        if valid.size:
            chunks.append(valid)

    if not chunks:
        return 0.0, 1.0

    combined = np.concatenate(chunks)

    if variable == "Dir":
        return 0.0, 360.0

    lower = float(np.percentile(combined, 1))
    upper = float(np.percentile(combined, 99))

    if variable in {"Hsig", "Tm01", "TPsmoo"}:
        lower = max(0.0, lower)

    if np.isclose(lower, upper):
        upper = lower + 1.0

    return lower, upper


def draw_land_and_coast(axis: plt.Axes, grid: Grid) -> None:
    lon_mesh, lat_mesh = grid.mesh
    wet = grid.wet_mask.astype(float)

    axis.contourf(
        lon_mesh,
        lat_mesh,
        wet,
        levels=[-0.5, 0.5],
        alpha=0.22,
        zorder=4,
    )

    axis.contour(
        lon_mesh,
        lat_mesh,
        wet,
        levels=[0.5],
        linewidths=0.8,
        zorder=5,
    )


def configure_axis(
    axis: plt.Axes,
    grid: Grid,
    title: str,
) -> None:
    axis.set_title(title, pad=12, fontsize=12)
    axis.set_xlabel("")
    axis.set_ylabel("Latitude (°)", labelpad=8)
    axis.set_xlim(float(grid.longitude.min()), float(grid.longitude.max()))
    axis.set_ylim(float(grid.latitude.min()), float(grid.latitude.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.35, alpha=0.30, linestyle="--")
    axis.tick_params(axis="both", direction="out", length=4, width=0.8)

def wave_direction_to_vectors(
    direction_degrees: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts nautical directions into Cartesian unit vectors.

    Nautical convention:
        0° = North
        90° = East
        Direction is interpreted as FROM.

    For propagation arrows:
        direction_toward = direction_from + 180°
    """
    direction = direction_degrees.astype(np.float64, copy=True)

    if mode == "toward":
        direction = (direction + 180.0) % 360.0

    radians = np.deg2rad(direction)

    u = np.sin(radians)
    v = np.cos(radians)

    return u, v


def vector_subset(
    grid: Grid,
    u: np.ndarray,
    v: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_mesh, lat_mesh = grid.mesh

    return (
        lon_mesh[::step, ::step],
        lat_mesh[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
    )


def add_statistics_box(
    axis: plt.Axes,
    values: np.ndarray,
    unit: str,
) -> None:
    stats = statistics(values)

    if stats["valid_points"] == 0:
        return

    text = (
        f"Mean: {stats['mean']:.2f} {unit}\n"
        f"Min: {stats['minimum']:.2f} {unit}\n"
        f"Max: {stats['maximum']:.2f} {unit}"
    )

    axis.text(
        0.015,
        0.02,
        text,
        transform=axis.transAxes,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "none",
        },
        zorder=10,
    )


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    saved: list[Path] = []
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for extension in formats:
        path = base_path.with_suffix(f".{extension}")

        kwargs = {
            "facecolor": "white",
        }

        if extension == "png":
            kwargs["dpi"] = dpi

        figure.savefig(
            path,
            **kwargs,
        )

        saved.append(path)

    return saved


def plot_wave_map(
    hs_field: Field,
    dir_field: Field,
    grid: Grid,
    limits: tuple[float, float],
    vector_step: int,
    arrow_mode: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> Path:
    hs = mask_field(hs_field.values, grid)
    direction = mask_field(dir_field.values, grid)

    u, v = wave_direction_to_vectors(direction, arrow_mode)
    invalid = ~np.isfinite(hs) | ~np.isfinite(direction)
    u[invalid] = np.nan
    v[invalid] = np.nan

    lon_mesh, lat_mesh = grid.mesh

    figure = plt.figure(figsize=(9.2, 10.2))
    gs = figure.add_gridspec(
        5, 1,
        height_ratios=[16.0, 1.00, 1.35, 1.50, 0.90],
        hspace=0.42,
    )

    axis = figure.add_subplot(gs[0])
    longitude_axis = figure.add_subplot(gs[1])
    colorbar_axis = figure.add_subplot(gs[2])
    legend_axis = figure.add_subplot(gs[3])
    footer_axis = figure.add_subplot(gs[4])

    image = axis.pcolormesh(
        lon_mesh, lat_mesh, hs,
        shading="auto",
        vmin=limits[0],
        vmax=limits[1],
        zorder=1,
    )

    qlon, qlat, qu, qv = vector_subset(grid, u, v, vector_step)

    axis.quiver(
        qlon, qlat, qu, qv,
        angles="xy",
        scale_units="xy",
        scale=10.0,
        width=0.0022,
        headwidth=3.5,
        headlength=4.5,
        zorder=6,
    )

    draw_land_and_coast(axis, grid)
    configure_axis(
        axis,
        grid,
        f"Significant wave height and direction — {hs_field.timestamp:%Y-%m-%d %H:%M UTC}",
    )
    add_statistics_box(axis, hs, "m")

    longitude_axis.axis("off")
    longitude_axis.text(0.5, 0.35, "Longitude (°)", ha="center", va="center", fontsize=11)

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Significant wave height (m)", labelpad=8)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(axis="x", pad=4)

    legend_axis.axis("off")
    label = (
        "Vectors indicate wave propagation direction"
        if arrow_mode == "toward"
        else "Vectors indicate the nautical direction from which waves arrive"
    )
    legend_axis.text(0.5, 0.55, label, ha="center", va="center", fontsize=10)

    footer_axis.axis("off")
    footer_axis.text(
        0.5, 0.30,
        "SWAN 41.51A • GEBCO 2023 • ERA5",
        ha="center", va="center", fontsize=8, color="0.4",
    )

    save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return output_base.with_suffix(".png")

def plot_wind_map(
    u_field: Field,
    v_field: Field,
    grid: Grid,
    limits: tuple[float, float],
    vector_step: int,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> Path:
    u = mask_field(u_field.values, grid)
    v = mask_field(v_field.values, grid)
    speed = np.sqrt(u**2 + v**2)

    lon_mesh, lat_mesh = grid.mesh

    figure = plt.figure(figsize=(9.2, 10.2))
    gs = figure.add_gridspec(
        5, 1,
        height_ratios=[16.0, 1.00, 1.35, 1.80, 0.90],
        hspace=0.42,
    )

    axis = figure.add_subplot(gs[0])
    longitude_axis = figure.add_subplot(gs[1])
    colorbar_axis = figure.add_subplot(gs[2])
    legend_axis = figure.add_subplot(gs[3])
    footer_axis = figure.add_subplot(gs[4])

    image = axis.pcolormesh(
        lon_mesh, lat_mesh, speed,
        shading="auto",
        vmin=limits[0],
        vmax=limits[1],
        zorder=1,
    )

    qlon, qlat, qu, qv = vector_subset(grid, u, v, vector_step)

    axis.quiver(
        qlon, qlat, qu, qv,
        angles="xy",
        scale_units="xy",
        scale=55.0,
        width=0.0022,
        headwidth=3.5,
        headlength=4.5,
        zorder=6,
    )

    draw_land_and_coast(axis, grid)
    configure_axis(
        axis,
        grid,
        f"Wind speed and direction — {u_field.timestamp:%Y-%m-%d %H:%M UTC}",
    )
    add_statistics_box(axis, speed, "m/s")

    longitude_axis.axis("off")
    longitude_axis.text(0.5, 0.35, "Longitude (°)", ha="center", va="center", fontsize=11)

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Wind speed (m/s)", labelpad=8)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(axis="x", pad=4)

    legend_axis.set_xlim(0, 1)
    legend_axis.set_ylim(0, 1)
    legend_axis.axis("off")

    legend_axis.annotate(
        "",
        xy=(0.43, 0.72),
        xytext=(0.28, 0.72),
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.4, "color": "black"},
    )
    legend_axis.text(
        0.46, 0.72,
        "5 m/s reference vector",
        ha="left", va="center", fontsize=10,
    )
    legend_axis.text(
        0.5, 0.22,
        "Background colors show wind speed; arrows show wind direction and relative magnitude.",
        ha="center", va="center", fontsize=9,
    )

    footer_axis.axis("off")
    footer_axis.text(
        0.5, 0.30,
        "SWAN 41.51A • GEBCO 2023 • ERA5",
        ha="center", va="center", fontsize=8, color="0.4",
    )

    save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return output_base.with_suffix(".png")

def plot_scalar_map(
    field: Field,
    grid: Grid,
    limits: tuple[float, float],
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> Path:
    config = SCALAR_CONFIG[field.variable]
    values = mask_field(field.values, grid)

    if field.variable == "Dir":
        finite = np.isfinite(values)
        values[finite] %= 360.0

    lon_mesh, lat_mesh = grid.mesh

    figure = plt.figure(figsize=(9.2, 9.4))
    gs = figure.add_gridspec(
        4, 1,
        height_ratios=[16.0, 1.00, 1.35, 0.90],
        hspace=0.42,
    )

    axis = figure.add_subplot(gs[0])
    longitude_axis = figure.add_subplot(gs[1])
    colorbar_axis = figure.add_subplot(gs[2])
    footer_axis = figure.add_subplot(gs[3])

    image = axis.pcolormesh(
        lon_mesh, lat_mesh, values,
        shading="auto",
        vmin=limits[0],
        vmax=limits[1],
        zorder=1,
    )

    draw_land_and_coast(axis, grid)
    configure_axis(
        axis,
        grid,
        f"{config['label']} — {field.timestamp:%Y-%m-%d %H:%M UTC}",
    )
    add_statistics_box(axis, values, config["unit"])

    longitude_axis.axis("off")
    longitude_axis.text(0.5, 0.35, "Longitude (°)", ha="center", va="center", fontsize=11)

    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        f"{config['label']} ({config['unit']})",
        labelpad=8,
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(axis="x", pad=4)

    footer_axis.axis("off")
    footer_axis.text(
        0.5, 0.30,
        "SWAN 41.51A • GEBCO 2023 • ERA5",
        ha="center", va="center", fontsize=8, color="0.4",
    )

    save_figure(figure, output_base, formats, dpi)
    plt.close(figure)
    return output_base.with_suffix(".png")

def normalize_frames(image_paths: list[Path]) -> list[np.ndarray]:
    images: list[Image.Image] = []

    for path in image_paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB").copy())

    if not images:
        return []

    width = max(image.width for image in images)
    height = max(image.height for image in images)

    frames: list[np.ndarray] = []

    for image in images:
        canvas = Image.new(
            "RGB",
            (width, height),
            "white",
        )

        x = (width - image.width) // 2
        y = (height - image.height) // 2

        canvas.paste(image, (x, y))
        frames.append(np.asarray(canvas, dtype=np.uint8))

    return frames


def write_gif(
    image_paths: list[Path],
    output_path: Path,
    duration: float,
) -> None:
    frames = normalize_frames(image_paths)

    if not frames:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(
        output_path,
        frames,
        duration=duration,
        loop=0,
    )


def shared_timestamps(
    fields: dict[str, dict[str, Field]],
    variables: Iterable[str],
) -> list[str]:
    variable_sets = [
        set(fields.get(variable, {}))
        for variable in variables
    ]

    if not variable_sets:
        return []

    return sorted(set.intersection(*variable_sets))


def wind_limits(
    fields: dict[str, dict[str, Field]],
    timestamps: list[str],
    grid: Grid,
) -> tuple[float, float]:
    chunks: list[np.ndarray] = []

    for timestamp in timestamps:
        u = mask_field(
            fields["Windv_x"][timestamp].values,
            grid,
        )
        v = mask_field(
            fields["Windv_y"][timestamp].values,
            grid,
        )

        speed = np.sqrt(u**2 + v**2)
        valid = speed[np.isfinite(speed)]

        if valid.size:
            chunks.append(valid)

    if not chunks:
        return 0.0, 1.0

    combined = np.concatenate(chunks)

    return (
        0.0,
        float(np.percentile(combined, 99)),
    )


def main() -> None:
    args = parse_args()

    if args.vector_step < 1:
        raise ValueError("--vector-step must be at least 1.")

    ensure_files_exist(
        (MAT_FILE, GRID_FILE, DEPTH_FILE)
    )

    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading grid: {GRID_FILE}")
    grid = load_grid()

    print(f"Reading MATLAB output: {MAT_FILE}")
    fields = load_fields()

    validate_shapes(fields, grid)

    wave_times = shared_timestamps(
        fields,
        ("Hsig", "Dir"),
    )

    wind_times = shared_timestamps(
        fields,
        ("Windv_x", "Windv_y"),
    )

    scalar_variables = [
        variable
        for variable in ("Hsig", "Tm01", "TPsmoo", "Dir")
        if variable in fields
    ]

    if args.skip_first:
        wave_times = wave_times[1:]
        wind_times = wind_times[1:]

    summary_rows: list[dict[str, object]] = []

    # -------------------------
    # Wave maps with vectors
    # -------------------------
    if wave_times:
        wave_hs_fields = [
            fields["Hsig"][timestamp]
            for timestamp in wave_times
        ]

        hs_limits = robust_limits(
            wave_hs_fields,
            grid,
            "Hsig",
        )

        wave_pngs: list[Path] = []

        print("\nGenerating wave maps with vectors...")

        for timestamp in wave_times:
            hs_field = fields["Hsig"][timestamp]
            dir_field = fields["Dir"][timestamp]

            png = plot_wave_map(
                hs_field=hs_field,
                dir_field=dir_field,
                grid=grid,
                limits=hs_limits,
                vector_step=args.vector_step,
                arrow_mode=args.wave_arrow_mode,
                output_base=(
                    FIGURES_DIR
                    / "wave"
                    / f"wave_{timestamp}"
                ),
                formats=args.formats,
                dpi=args.dpi,
            )

            wave_pngs.append(png)

            hs_values = mask_field(
                hs_field.values,
                grid,
            )
            stats = statistics(hs_values)

            summary_rows.append(
                {
                    "product": "wave",
                    "variable": "Hsig",
                    "timestamp": hs_field.timestamp.isoformat(),
                    **stats,
                }
            )

            print(
                f"  {timestamp}: "
                f"mean Hs={stats['mean']:.2f} m, "
                f"max Hs={stats['maximum']:.2f} m"
            )

        if not args.no_gif and "png" in args.formats:
            write_gif(
                wave_pngs,
                ANIMATIONS_DIR / "wave_height_direction.gif",
                args.gif_duration,
            )

    # -------------------------
    # Wind maps with vectors
    # -------------------------
    if wind_times:
        limits = wind_limits(
            fields,
            wind_times,
            grid,
        )

        wind_pngs: list[Path] = []

        print("\nGenerating wind maps with vectors...")

        for timestamp in wind_times:
            u_field = fields["Windv_x"][timestamp]
            v_field = fields["Windv_y"][timestamp]

            png = plot_wind_map(
                u_field=u_field,
                v_field=v_field,
                grid=grid,
                limits=limits,
                vector_step=args.vector_step,
                output_base=(
                    FIGURES_DIR
                    / "wind"
                    / f"wind_{timestamp}"
                ),
                formats=args.formats,
                dpi=args.dpi,
            )

            wind_pngs.append(png)

            u = mask_field(u_field.values, grid)
            v = mask_field(v_field.values, grid)
            speed = np.sqrt(u**2 + v**2)
            stats = statistics(speed)

            summary_rows.append(
                {
                    "product": "wind",
                    "variable": "WindSpeed",
                    "timestamp": u_field.timestamp.isoformat(),
                    **stats,
                }
            )

            print(
                f"  {timestamp}: "
                f"mean wind={stats['mean']:.2f} m/s, "
                f"max wind={stats['maximum']:.2f} m/s"
            )

        if not args.no_gif and "png" in args.formats:
            write_gif(
                wind_pngs,
                ANIMATIONS_DIR / "wind_speed_direction.gif",
                args.gif_duration,
            )

    # -------------------------
    # Scalar maps
    # -------------------------
    print("\nGenerating scalar maps...")

    for variable in scalar_variables:
        timestamps = sorted(fields[variable])

        if args.skip_first:
            timestamps = timestamps[1:]

        scalar_fields = [
            fields[variable][timestamp]
            for timestamp in timestamps
        ]

        limits = robust_limits(
            scalar_fields,
            grid,
            variable,
        )

        pngs: list[Path] = []

        for timestamp in timestamps:
            field = fields[variable][timestamp]

            png = plot_scalar_map(
                field=field,
                grid=grid,
                limits=limits,
                output_base=(
                    FIGURES_DIR
                    / "scalar"
                    / variable
                    / f"{variable}_{timestamp}"
                ),
                formats=args.formats,
                dpi=args.dpi,
            )

            pngs.append(png)

            values = mask_field(field.values, grid)
            stats = statistics(values)

            summary_rows.append(
                {
                    "product": "scalar",
                    "variable": variable,
                    "timestamp": field.timestamp.isoformat(),
                    **stats,
                }
            )

        if not args.no_gif and "png" in args.formats:
            write_gif(
                pngs,
                ANIMATIONS_DIR / f"{variable}.gif",
                args.gif_duration,
            )

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        raise RuntimeError(
            "No compatible SWAN output variables were found."
        )

    summary.sort_values(
        ["product", "variable", "timestamp"],
        inplace=True,
    )

    summary.to_csv(
        PUBLICATION_DIR / "summary.csv",
        index=False,
    )

    for (product, variable), table in summary.groupby(
        ["product", "variable"]
    ):
        table.to_csv(
            CSV_DIR / f"{product}_{variable}.csv",
            index=False,
        )

    print("\nPublication post-processing completed.")
    print(f"Figures:    {FIGURES_DIR}")
    print(f"Animations: {ANIMATIONS_DIR}")
    print(f"CSV files:  {CSV_DIR}")
    print(f"Summary:    {PUBLICATION_DIR / 'summary.csv'}")
    print(
        "Publication layout uses separate GridSpec rows for the map, "
        "horizontal colorbar and vector legend."
    )


if __name__ == "__main__":
    main()