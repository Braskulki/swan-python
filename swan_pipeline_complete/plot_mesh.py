"""
Plot and export the SWAN regular computational grid and bathymetry.

Inputs:
    data/processed/grid.json
    data/processed/depth.bot

Outputs:
    data/domain/
    ├── bathymetry.png
    ├── bathymetry.pdf
    ├── regular_grid.png
    ├── regular_grid.pdf
    ├── bathymetry_grid.png
    ├── bathymetry_grid.pdf
    ├── bathymetry_contours.png
    ├── bathymetry_contours.pdf
    └── grid_nodes.csv

Usage:
    py plot_mesh.py
    py plot_mesh.py --formats png pdf
    py plot_mesh.py --dpi 300
    py plot_mesh.py --grid-step 4
    py plot_mesh.py --no-grid-overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "domain"

GRID_FILE = PROCESSED_DIR / "grid.json"
DEPTH_FILE = PROCESSED_DIR / "depth.bot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-style bathymetry and regular-grid figures "
            "from grid.json and depth.bot."
        )
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Output figure formats. Default: png pdf.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300 DPI.",
    )

    parser.add_argument(
        "--grid-step",
        type=int,
        default=4,
        help=(
            "Draw one grid line every N grid cells. "
            "Use 1 to draw the full grid. Default: 4."
        ),
    )

    parser.add_argument(
        "--no-grid-overlay",
        action="store_true",
        help="Do not overlay grid lines on the bathymetry-grid figure.",
    )

    parser.add_argument(
        "--contour-levels",
        nargs="+",
        type=float,
        default=(10, 20, 50, 100, 200, 500, 1000, 2000),
        help=(
            "Bathymetric contour levels in metres. "
            "Default: 10 20 50 100 200 500 1000 2000."
        ),
    )

    return parser.parse_args()


def ensure_inputs() -> None:
    missing = [
        path
        for path in (GRID_FILE, DEPTH_FILE)
        if not path.exists()
    ]

    if missing:
        listing = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Required input files were not found:\n{listing}"
        )


def load_domain() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = json.loads(
        GRID_FILE.read_text(encoding="utf-8")
    )

    longitude = np.asarray(
        metadata["lon"],
        dtype=np.float64,
    )
    latitude = np.asarray(
        metadata["lat"],
        dtype=np.float64,
    )
    depth = np.loadtxt(
        DEPTH_FILE,
        dtype=np.float64,
    )

    expected_shape = (
        latitude.size,
        longitude.size,
    )

    if depth.shape != expected_shape:
        raise ValueError(
            "depth.bot shape does not match grid.json. "
            f"Expected {expected_shape}, found {depth.shape}."
        )

    if not np.all(np.diff(longitude) > 0):
        raise ValueError(
            "Longitude coordinates must be strictly increasing."
        )

    if not np.all(np.diff(latitude) > 0):
        raise ValueError(
            "Latitude coordinates must be strictly increasing."
        )

    return longitude, latitude, depth


def water_depth(depth: np.ndarray) -> np.ndarray:
    """
    Returns positive water depths and masks dry/land cells.

    In the current pipeline:
        depth > 0  -> wet point
        depth <= 0 -> land/dry point
    """
    result = depth.astype(np.float64, copy=True)
    result[~np.isfinite(result)] = np.nan
    result[result <= 0] = np.nan
    return result


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    base_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in formats:
        output_path = base_path.with_suffix(
            f".{extension}"
        )

        kwargs = {
            "facecolor": "white",
        }

        if extension == "png":
            kwargs["dpi"] = dpi

        figure.savefig(
            output_path,
            **kwargs,
        )


def configure_map_axis(
    axis: plt.Axes,
    longitude: np.ndarray,
    latitude: np.ndarray,
    title: str,
) -> None:
    axis.set_title(
        title,
        pad=12,
        fontsize=12,
    )
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")

    axis.set_xlim(
        float(longitude.min()),
        float(longitude.max()),
    )
    axis.set_ylim(
        float(latitude.min()),
        float(latitude.max()),
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.grid(
        True,
        linewidth=0.35,
        linestyle="--",
        alpha=0.3,
    )

    axis.tick_params(
        direction="out",
        length=4,
        width=0.8,
    )


def add_footer(
    figure: plt.Figure,
    text: str,
) -> None:
    figure.text(
        0.5,
        0.018,
        text,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=8,
        color="0.4",
    )


def add_statistics(
    axis: plt.Axes,
    depth: np.ndarray,
) -> None:
    valid = depth[np.isfinite(depth)]

    if valid.size == 0:
        return

    text = (
        f"Wet cells: {valid.size}\n"
        f"Min depth: {np.min(valid):.2f} m\n"
        f"Max depth: {np.max(valid):.2f} m\n"
        f"Mean depth: {np.mean(valid):.2f} m"
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


def draw_coastline(
    axis: plt.Axes,
    longitude_mesh: np.ndarray,
    latitude_mesh: np.ndarray,
    depth: np.ndarray,
) -> None:
    wet_mask = np.isfinite(depth).astype(float)

    axis.contour(
        longitude_mesh,
        latitude_mesh,
        wet_mask,
        levels=[0.5],
        linewidths=0.9,
        zorder=6,
    )


def draw_regular_grid(
    axis: plt.Axes,
    longitude: np.ndarray,
    latitude: np.ndarray,
    step: int,
    linewidth: float = 0.25,
    alpha: float = 0.45,
) -> None:
    step = max(1, int(step))

    for x in longitude[::step]:
        axis.plot(
            [x, x],
            [latitude.min(), latitude.max()],
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
        )

    for y in latitude[::step]:
        axis.plot(
            [longitude.min(), longitude.max()],
            [y, y],
            linewidth=linewidth,
            alpha=alpha,
            zorder=5,
        )


def plot_bathymetry(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    formats: Iterable[str],
    dpi: int,
) -> None:
    longitude_mesh, latitude_mesh = np.meshgrid(
        longitude,
        latitude,
    )

    figure, axis = plt.subplots(
        figsize=(9.2, 7.6),
        constrained_layout=True,
    )

    valid = depth[np.isfinite(depth)]
    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99))

    image = axis.pcolormesh(
        longitude_mesh,
        latitude_mesh,
        depth,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    draw_coastline(
        axis,
        longitude_mesh,
        latitude_mesh,
        depth,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        pad=0.10,
        fraction=0.055,
        aspect=35,
    )
    colorbar.set_label("Water depth (m)")

    configure_map_axis(
        axis,
        longitude,
        latitude,
        "SWAN bathymetry",
    )

    add_statistics(axis, depth)
    add_footer(
        figure,
        "SWAN regular grid • GEBCO bathymetry",
    )

    save_figure(
        figure,
        OUTPUT_DIR / "bathymetry",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_grid_only(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    formats: Iterable[str],
    dpi: int,
    grid_step: int,
) -> None:
    longitude_mesh, latitude_mesh = np.meshgrid(
        longitude,
        latitude,
    )

    figure, axis = plt.subplots(
        figsize=(9.2, 7.6),
        constrained_layout=True,
    )

    draw_regular_grid(
        axis,
        longitude,
        latitude,
        step=grid_step,
        linewidth=0.35,
        alpha=0.7,
    )

    draw_coastline(
        axis,
        longitude_mesh,
        latitude_mesh,
        depth,
    )

    configure_map_axis(
        axis,
        longitude,
        latitude,
        "SWAN regular computational grid",
    )

    total_cells = (
        (longitude.size - 1)
        * (latitude.size - 1)
    )

    axis.text(
        0.015,
        0.02,
        (
            f"Grid points: {longitude.size} × {latitude.size}\n"
            f"Grid cells: {total_cells}\n"
            f"Displayed every {grid_step} cell(s)"
        ),
        transform=axis.transAxes,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "none",
        },
    )

    add_footer(
        figure,
        "SWAN CGRID REGULAR",
    )

    save_figure(
        figure,
        OUTPUT_DIR / "regular_grid",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_bathymetry_grid(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    formats: Iterable[str],
    dpi: int,
    grid_step: int,
    overlay_grid: bool,
) -> None:
    longitude_mesh, latitude_mesh = np.meshgrid(
        longitude,
        latitude,
    )

    figure, axis = plt.subplots(
        figsize=(9.2, 7.6),
        constrained_layout=True,
    )

    valid = depth[np.isfinite(depth)]
    vmin = float(np.percentile(valid, 1))
    vmax = float(np.percentile(valid, 99))

    image = axis.pcolormesh(
        longitude_mesh,
        latitude_mesh,
        depth,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    if overlay_grid:
        draw_regular_grid(
            axis,
            longitude,
            latitude,
            step=grid_step,
            linewidth=0.2,
            alpha=0.4,
        )

    draw_coastline(
        axis,
        longitude_mesh,
        latitude_mesh,
        depth,
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        pad=0.10,
        fraction=0.055,
        aspect=35,
    )
    colorbar.set_label("Water depth (m)")

    configure_map_axis(
        axis,
        longitude,
        latitude,
        "Bathymetry and SWAN regular grid",
    )

    add_statistics(axis, depth)

    add_footer(
        figure,
        (
            f"SWAN CGRID REGULAR • Grid line interval: "
            f"{grid_step} cell(s)"
        ),
    )

    save_figure(
        figure,
        OUTPUT_DIR / "bathymetry_grid",
        formats,
        dpi,
    )

    plt.close(figure)


def plot_contours(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    formats: Iterable[str],
    dpi: int,
    contour_levels: Iterable[float],
) -> None:
    longitude_mesh, latitude_mesh = np.meshgrid(
        longitude,
        latitude,
    )

    valid = depth[np.isfinite(depth)]
    minimum = float(np.min(valid))
    maximum = float(np.max(valid))

    levels = sorted(
        {
            float(level)
            for level in contour_levels
            if minimum <= float(level) <= maximum
        }
    )

    if not levels:
        levels = np.linspace(
            minimum,
            maximum,
            8,
        ).tolist()

    figure, axis = plt.subplots(
        figsize=(9.2, 7.6),
        constrained_layout=True,
    )

    contour = axis.contour(
        longitude_mesh,
        latitude_mesh,
        depth,
        levels=levels,
        linewidths=0.8,
    )

    axis.clabel(
        contour,
        inline=True,
        fontsize=8,
        fmt=lambda value: f"{value:g} m",
    )

    draw_coastline(
        axis,
        longitude_mesh,
        latitude_mesh,
        depth,
    )

    configure_map_axis(
        axis,
        longitude,
        latitude,
        "Bathymetric contours",
    )

    add_footer(
        figure,
        "Bathymetric contours derived from depth.bot",
    )

    save_figure(
        figure,
        OUTPUT_DIR / "bathymetry_contours",
        formats,
        dpi,
    )

    plt.close(figure)


def export_grid_nodes(
    longitude: np.ndarray,
    latitude: np.ndarray,
    raw_depth: np.ndarray,
) -> None:
    longitude_mesh, latitude_mesh = np.meshgrid(
        longitude,
        latitude,
    )

    table = pd.DataFrame(
        {
            "row": np.repeat(
                np.arange(latitude.size),
                longitude.size,
            ),
            "column": np.tile(
                np.arange(longitude.size),
                latitude.size,
            ),
            "longitude": longitude_mesh.reshape(-1),
            "latitude": latitude_mesh.reshape(-1),
            "depth": raw_depth.reshape(-1),
            "is_wet": (
                np.isfinite(raw_depth)
                & (raw_depth > 0)
            ).reshape(-1),
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        OUTPUT_DIR / "grid_nodes.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()

    if args.grid_step < 1:
        raise ValueError(
            "--grid-step must be at least 1."
        )

    ensure_inputs()

    print(f"Reading grid: {GRID_FILE}")
    longitude, latitude, raw_depth = load_domain()
    depth = water_depth(raw_depth)

    if not np.isfinite(depth).any():
        raise RuntimeError(
            "No wet cells were found in depth.bot."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_grid_nodes(
        longitude,
        latitude,
        raw_depth,
    )

    print("Generating bathymetry map...")
    plot_bathymetry(
        longitude,
        latitude,
        depth,
        args.formats,
        args.dpi,
    )

    print("Generating regular-grid map...")
    plot_grid_only(
        longitude,
        latitude,
        depth,
        args.formats,
        args.dpi,
        args.grid_step,
    )

    print("Generating bathymetry-grid map...")
    plot_bathymetry_grid(
        longitude,
        latitude,
        depth,
        args.formats,
        args.dpi,
        args.grid_step,
        not args.no_grid_overlay,
    )

    print("Generating bathymetric contours...")
    plot_contours(
        longitude,
        latitude,
        depth,
        args.formats,
        args.dpi,
        args.contour_levels,
    )

    print("\nDomain inspection completed.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        f"Grid shape: {latitude.size} × {longitude.size} points"
    )


if __name__ == "__main__":
    main()