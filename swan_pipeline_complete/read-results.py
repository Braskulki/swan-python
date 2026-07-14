"""
Read and post-process SWAN MATLAB output using geographic coordinates.

Inputs:
    data/processed/output.mat
    data/processed/grid.json
    data/processed/depth.bot

Outputs:
    data/results/summary.csv
    data/results/timeseries_<variable>.csv
    data/results/maps/<variable>/<variable>_<timestamp>.png
    data/results/animations/<variable>.gif

Usage examples:
    py read-results.py
    py read-results.py --skip-first
    py read-results.py --variables Hsig TPsmoo Dir --skip-first
    py read-results.py --no-gif
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
RESULTS_DIR = BASE_DIR / "data" / "results"

MAT_FILE = PROCESSED_DIR / "output.mat"
GRID_FILE = PROCESSED_DIR / "grid.json"
DEPTH_FILE = PROCESSED_DIR / "depth.bot"

MAPS_DIR = RESULTS_DIR / "maps"
ANIMATIONS_DIR = RESULTS_DIR / "animations"

VARIABLE_CONFIG = {
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
    "Windv_x": {
        "label": "Wind U component",
        "unit": "m/s",
    },
    "Windv_y": {
        "label": "Wind V component",
        "unit": "m/s",
    },
}

KEY_PATTERN = re.compile(
    r"^(?P<variable>.+)_(?P<timestamp>\d{8}_\d{6})$"
)


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
class OutputField:
    variable: str
    timestamp_text: str
    timestamp: datetime
    values: np.ndarray
    key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate georeferenced maps, GIF animations and statistics "
            "from SWAN output.mat."
        )
    )

    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(VARIABLE_CONFIG),
        help=(
            "Variables to process. Default: "
            + ", ".join(VARIABLE_CONFIG)
        ),
    )

    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="Skip the first timestep as model spin-up.",
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Do not generate GIF animations.",
    )

    parser.add_argument(
        "--gif-duration",
        type=float,
        default=0.8,
        help="Duration of each GIF frame in seconds. Default: 0.8.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution. Default: 180 DPI.",
    )

    parser.add_argument(
        "--fixed-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a fixed color scale across timestamps of the same variable. "
            "Enabled by default."
        ),
    )

    return parser.parse_args()


def ensure_files_exist(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Required files were not found:\n{formatted}"
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
            "Grid/depth dimensions do not match. "
            f"Expected {expected_shape}, found {depth.shape}."
        )

    if longitude.size < 2 or latitude.size < 2:
        raise ValueError(
            "The grid must contain at least two points per axis."
        )

    if not np.all(np.diff(longitude) > 0):
        raise ValueError(
            "Longitude values must be strictly increasing."
        )

    if not np.all(np.diff(latitude) > 0):
        raise ValueError(
            "Latitude values must be strictly increasing."
        )

    return Grid(
        longitude=longitude,
        latitude=latitude,
        depth=depth,
    )


def safe_float_array(raw_values: np.ndarray) -> np.ndarray:
    """
    Converts MATLAB arrays to float64 while preserving invalid values as NaN.

    SWAN may use large integer sentinel values in MATLAB output.
    Those values are replaced with NaN.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        values = np.array(
            raw_values,
            dtype=np.float64,
            copy=True,
        )

    # Replace non-finite values.
    values[~np.isfinite(values)] = np.nan

    # SWAN/MATLAB missing-value sentinels can appear as very large magnitudes.
    values[np.abs(values) > 1.0e20] = np.nan

    return values


def load_output_fields(
    requested_variables: set[str],
) -> dict[str, list[OutputField]]:
    raw_data = loadmat(MAT_FILE)

    fields_by_variable: dict[str, list[OutputField]] = {
        variable: [] for variable in requested_variables
    }

    for key, raw_values in raw_data.items():
        if key.startswith("__"):
            continue

        match = KEY_PATTERN.match(key)

        if not match:
            continue

        variable = match.group("variable")
        timestamp_text = match.group("timestamp")

        if variable not in requested_variables:
            continue

        timestamp = datetime.strptime(
            timestamp_text,
            "%Y%m%d_%H%M%S",
        )

        values = safe_float_array(raw_values)

        fields_by_variable[variable].append(
            OutputField(
                variable=variable,
                timestamp_text=timestamp_text,
                timestamp=timestamp,
                values=values,
                key=key,
            )
        )

    for fields in fields_by_variable.values():
        fields.sort(key=lambda item: item.timestamp)

    return fields_by_variable


def validate_field_shapes(
    fields_by_variable: dict[str, list[OutputField]],
    grid: Grid,
) -> None:
    expected_shape = grid.depth.shape

    for variable, fields in fields_by_variable.items():
        for field in fields:
            if field.values.shape != expected_shape:
                raise ValueError(
                    f"{field.key} has shape {field.values.shape}; "
                    f"expected {expected_shape}."
                )


def apply_domain_mask(
    variable: str,
    values: np.ndarray,
    grid: Grid,
) -> np.ndarray:
    result = values.astype(np.float64, copy=True)

    # Mask dry/land cells consistently for all variables.
    result[grid.land_mask] = np.nan

    if variable == "Dir":
        finite = np.isfinite(result)
        result[finite] %= 360.0

    return result


def finite_statistics(
    values: np.ndarray,
) -> dict[str, float | int]:
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]

    if finite.size == 0:
        return {
            "valid_points": 0,
            "nan_points": int(np.isnan(values).sum()),
            "minimum": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "standard_deviation": np.nan,
            "zero_points": 0,
        }

    return {
        "valid_points": int(finite.size),
        "nan_points": int(np.isnan(values).sum()),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "standard_deviation": float(np.std(finite)),
        "zero_points": int(np.count_nonzero(finite == 0)),
    }


def robust_limits(
    fields: list[OutputField],
    variable: str,
    grid: Grid,
) -> tuple[float, float]:
    valid_chunks: list[np.ndarray] = []

    for field in fields:
        values = apply_domain_mask(
            variable,
            field.values,
            grid,
        )

        valid = values[np.isfinite(values)]

        if valid.size:
            valid_chunks.append(valid)

    if not valid_chunks:
        return 0.0, 1.0

    combined = np.concatenate(valid_chunks)

    if variable == "Dir":
        return 0.0, 360.0

    lower = float(np.percentile(combined, 1))
    upper = float(np.percentile(combined, 99))

    if np.isclose(lower, upper):
        lower = float(np.min(combined))
        upper = float(np.max(combined))

    if np.isclose(lower, upper):
        upper = lower + 1.0

    if variable in {"Hsig", "Tm01", "TPsmoo"}:
        lower = max(0.0, lower)

    return lower, upper


def draw_coastline(
    axis: plt.Axes,
    grid: Grid,
) -> None:
    longitude_mesh, latitude_mesh = grid.mesh

    wet_values = grid.wet_mask.astype(float)

    # Light land mask.
    axis.contourf(
        longitude_mesh,
        latitude_mesh,
        wet_values,
        levels=[-0.5, 0.5],
        alpha=0.20,
    )

    # Coastline from wet/dry transition.
    axis.contour(
        longitude_mesh,
        latitude_mesh,
        wet_values,
        levels=[0.5],
        linewidths=0.9,
    )


def plot_field(
    field: OutputField,
    grid: Grid,
    output_path: Path,
    dpi: int,
    fixed_limits: tuple[float, float] | None,
) -> None:
    variable_config = VARIABLE_CONFIG.get(
        field.variable,
        {
            "label": field.variable,
            "unit": "",
        },
    )

    values = apply_domain_mask(
        field.variable,
        field.values,
        grid,
    )

    longitude_mesh, latitude_mesh = grid.mesh

    figure, axis = plt.subplots(
        figsize=(10, 7),
        constrained_layout=True,
    )

    plot_kwargs: dict[str, float] = {}

    if fixed_limits is not None:
        plot_kwargs["vmin"] = fixed_limits[0]
        plot_kwargs["vmax"] = fixed_limits[1]

    image = axis.pcolormesh(
        longitude_mesh,
        latitude_mesh,
        values,
        shading="auto",
        **plot_kwargs,
    )

    draw_coastline(axis, grid)

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.02,
    )

    colorbar_label = variable_config["label"]

    if variable_config["unit"]:
        colorbar_label += f" ({variable_config['unit']})"

    colorbar.set_label(colorbar_label)

    axis.set_title(
        f"{variable_config['label']} — "
        f"{field.timestamp:%Y-%m-%d %H:%M UTC}"
    )

    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")

    axis.set_xlim(
        float(grid.longitude.min()),
        float(grid.longitude.max()),
    )

    axis.set_ylim(
        float(grid.latitude.min()),
        float(grid.latitude.max()),
    )

    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.35, alpha=0.4)

    stats = finite_statistics(values)

    if stats["valid_points"] > 0:
        unit = variable_config["unit"]

        annotation = (
            f"Mean: {stats['mean']:.2f} {unit}\n"
            f"Min: {stats['minimum']:.2f} {unit}\n"
            f"Max: {stats['maximum']:.2f} {unit}"
        )

        axis.text(
            0.015,
            0.02,
            annotation,
            transform=axis.transAxes,
            verticalalignment="bottom",
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.80,
                "edgecolor": "none",
            },
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Do not use bbox_inches="tight".
    # It can produce PNG files with different dimensions,
    # which causes GIF generation to fail.
    figure.savefig(
        output_path,
        dpi=dpi,
        facecolor="white",
    )

    plt.close(figure)


def normalize_gif_frames(
    image_paths: list[Path],
) -> list[np.ndarray]:
    """
    Ensures that all GIF frames have exactly the same dimensions and mode.
    """
    if not image_paths:
        return []

    loaded_images: list[Image.Image] = []

    for image_path in image_paths:
        with Image.open(image_path) as source:
            loaded_images.append(
                source.convert("RGB").copy()
            )

    max_width = max(image.width for image in loaded_images)
    max_height = max(image.height for image in loaded_images)

    frames: list[np.ndarray] = []

    for image in loaded_images:
        canvas = Image.new(
            mode="RGB",
            size=(max_width, max_height),
            color="white",
        )

        x_offset = (max_width - image.width) // 2
        y_offset = (max_height - image.height) // 2

        canvas.paste(
            image,
            (x_offset, y_offset),
        )

        frames.append(
            np.asarray(canvas, dtype=np.uint8)
        )

    return frames


def generate_animation(
    image_paths: list[Path],
    output_path: Path,
    duration: float,
) -> None:
    frames = normalize_gif_frames(image_paths)

    if not frames:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    imageio.mimsave(
        output_path,
        frames,
        duration=duration,
        loop=0,
    )


def process_variable(
    variable: str,
    fields: list[OutputField],
    grid: Grid,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    if args.skip_first and fields:
        fields = fields[1:]

    if not fields:
        print(f"[skip] No fields found for {variable}.")
        return []

    limits = (
        robust_limits(fields, variable, grid)
        if args.fixed_scale
        else None
    )

    image_paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []

    print(f"\n{variable}")
    print(f"{len(fields)} timestep(s)")

    for field in fields:
        values = apply_domain_mask(
            variable,
            field.values,
            grid,
        )

        stats = finite_statistics(values)

        image_path = (
            MAPS_DIR
            / variable
            / f"{field.key}.png"
        )

        plot_field(
            field=field,
            grid=grid,
            output_path=image_path,
            dpi=args.dpi,
            fixed_limits=limits,
        )

        image_paths.append(image_path)

        total_points = int(values.size)
        valid_percentage = (
            100.0
            * int(stats["valid_points"])
            / total_points
        )

        summary_rows.append(
            {
                "variable": variable,
                "timestamp": field.timestamp.isoformat(),
                "source_key": field.key,
                "total_points": total_points,
                "valid_percentage": valid_percentage,
                **stats,
            }
        )

        if stats["valid_points"] > 0:
            print(
                f"{field.timestamp_text} "
                f"mean={stats['mean']:.2f} "
                f"max={stats['maximum']:.2f}"
            )
        else:
            print(
                f"{field.timestamp_text} "
                "no valid values"
            )

    if not args.no_gif:
        gif_path = ANIMATIONS_DIR / f"{variable}.gif"

        generate_animation(
            image_paths=image_paths,
            output_path=gif_path,
            duration=args.gif_duration,
        )

        print(f"GIF generated: {gif_path}")

    variable_summary = pd.DataFrame(summary_rows)

    variable_summary.to_csv(
        RESULTS_DIR / f"timeseries_{variable}.csv",
        index=False,
    )

    return summary_rows


def main() -> None:
    args = parse_args()

    ensure_files_exist(
        (
            MAT_FILE,
            GRID_FILE,
            DEPTH_FILE,
        )
    )

    requested_variables = set(args.variables)

    unknown_variables = requested_variables.difference(
        VARIABLE_CONFIG
    )

    if unknown_variables:
        print(
            "Warning: variables without predefined labels: "
            + ", ".join(sorted(unknown_variables))
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Reading grid: {GRID_FILE}")
    grid = load_grid()

    print(f"Reading MATLAB file: {MAT_FILE}")
    fields_by_variable = load_output_fields(
        requested_variables
    )

    total_variables = sum(
        len(fields)
        for fields in fields_by_variable.values()
    )

    print(f"{total_variables} variables/timestamps found")

    validate_field_shapes(
        fields_by_variable,
        grid,
    )

    all_summary_rows: list[dict[str, object]] = []

    for variable in args.variables:
        rows = process_variable(
            variable=variable,
            fields=fields_by_variable.get(variable, []),
            grid=grid,
            args=args,
        )

        all_summary_rows.extend(rows)

    summary = pd.DataFrame(all_summary_rows)

    if summary.empty:
        raise RuntimeError(
            "No matching SWAN variables were found in output.mat."
        )

    summary.sort_values(
        ["variable", "timestamp"],
        inplace=True,
    )

    summary.to_csv(
        RESULTS_DIR / "summary.csv",
        index=False,
    )

    print("\nPost-processing completed.")
    print(f"Maps: {MAPS_DIR}")

    if not args.no_gif:
        print(f"Animations: {ANIMATIONS_DIR}")

    print(f"Statistics: {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()