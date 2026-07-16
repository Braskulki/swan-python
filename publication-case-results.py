"""
Publication-style post-processing for standalone SWAN cases.

Designed for folders containing files such as:

    f32har01.swn
    f32har01.mat
    f32har01.vtu
    f32har01.tab
    f32har01.tbl
    f32har01.spc
    f32har01.prt

The script prefers the VTU file because it normally preserves the
unstructured mesh coordinates and field arrays. If no usable VTU file is
available, it attempts to read the MATLAB output.

Outputs are written to:

    publication_<case_name>/
    ├── figures/
    ├── animations/
    ├── csv/
    └── summary.csv

Requirements:

    pip install numpy pandas scipy matplotlib pillow imageio meshio

Examples:

    py publication-case-results.py f32har01
    py publication-case-results.py f32har01 --formats png pdf
    py publication-case-results.py f32har01 --no-gif
    py publication-case-results.py f32har01 --vector-step 5
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat


try:
    import meshio
except ImportError:
    meshio = None


FIELD_ALIASES = {
    "hs": (
        "Hsig",
        "HSIGN",
        "Hs",
        "hs",
        "Significant wave height",
    ),
    "tm01": (
        "Tm01",
        "TM01",
        "tm01",
    ),
    "tp": (
        "TPsmoo",
        "TPS",
        "Tp",
        "tp",
    ),
    "direction": (
        "Dir",
        "DIR",
        "Direction",
        "direction",
    ),
    "wind_u": (
        "Windv_x",
        "Wind_x",
        "Uwind",
        "wind_u",
        "WindU",
    ),
    "wind_v": (
        "Windv_y",
        "Wind_y",
        "Vwind",
        "wind_v",
        "WindV",
    ),
}

DISPLAY_CONFIG = {
    "hs": {
        "label": "Significant wave height",
        "unit": "m",
    },
    "tm01": {
        "label": "Mean wave period",
        "unit": "s",
    },
    "tp": {
        "label": "Peak wave period",
        "unit": "s",
    },
    "direction": {
        "label": "Mean wave direction",
        "unit": "°",
    },
    "wind_speed": {
        "label": "Wind speed",
        "unit": "m/s",
    },
}


@dataclass(frozen=True)
class MeshData:
    x: np.ndarray
    y: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class FieldData:
    canonical_name: str
    original_name: str
    timestamp: str
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-style maps from a standalone SWAN case."
        )
    )

    parser.add_argument(
        "case",
        help=(
            "Case name without extension, for example f32har01. "
            "The script must be executed in the case directory."
        ),
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
        help="Figure formats. Default: png pdf.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300 DPI.",
    )

    parser.add_argument(
        "--vector-step",
        type=int,
        default=5,
        help="Plot one vector every N mesh points. Default: 5.",
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
        help="Duration of each GIF frame in seconds.",
    )

    parser.add_argument(
        "--wave-arrow-mode",
        choices=("toward", "from"),
        default="toward",
        help=(
            "For nautical SWAN direction, show propagation direction "
            "or the direction from which waves arrive."
        ),
    )

    return parser.parse_args()


def safe_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    result[~np.isfinite(result)] = np.nan
    result[np.abs(result) > 1.0e20] = np.nan
    return result


def canonical_name(name: str) -> str | None:
    normalized = name.strip().lower()

    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if normalized == alias.lower():
                return canonical

    return None


def extract_timestamp(name: str) -> str:
    match = re.search(r"(\d{8})[_-]?(\d{6})", name)

    if match:
        return f"{match.group(1)}_{match.group(2)}"

    return "static"


def find_triangles(cells: list) -> np.ndarray:
    for cell_block in cells:
        if cell_block.type == "triangle":
            return np.asarray(cell_block.data, dtype=int)

    raise ValueError("No triangle cells were found in the VTU mesh.")


def load_vtu(path: Path) -> tuple[MeshData, list[FieldData]]:
    if meshio is None:
        raise RuntimeError(
            "meshio is required to read VTU files. "
            "Install it with: pip install meshio"
        )

    mesh = meshio.read(path)

    points = np.asarray(mesh.points, dtype=float)

    if points.shape[1] < 2:
        raise ValueError("VTU points do not contain X/Y coordinates.")

    triangles = find_triangles(mesh.cells)

    mesh_data = MeshData(
        x=points[:, 0],
        y=points[:, 1],
        triangles=triangles,
    )

    fields: list[FieldData] = []

    for name, raw_values in mesh.point_data.items():
        canonical = canonical_name(name)

        if canonical is None:
            continue

        values = np.asarray(raw_values)

        if values.ndim > 1:
            if values.shape[1] == 1:
                values = values[:, 0]
            else:
                continue

        values = safe_array(values)

        if values.size != mesh_data.x.size:
            continue

        fields.append(
            FieldData(
                canonical_name=canonical,
                original_name=name,
                timestamp=extract_timestamp(name),
                values=values,
            )
        )

    return mesh_data, fields


def find_mat_coordinate(
    raw: dict,
    aliases: tuple[str, ...],
) -> np.ndarray | None:
    for alias in aliases:
        if alias in raw:
            return safe_array(raw[alias])

    return None


def triangulate_coordinates(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    triangulation = mtri.Triangulation(x, y)
    return np.asarray(triangulation.triangles, dtype=int)


def load_mat(path: Path) -> tuple[MeshData, list[FieldData]]:
    raw = loadmat(path)

    x = find_mat_coordinate(raw, ("Xp", "X", "x", "lon", "longitude"))
    y = find_mat_coordinate(raw, ("Yp", "Y", "y", "lat", "latitude"))

    fields: list[FieldData] = []

    for name, raw_values in raw.items():
        if name.startswith("__"):
            continue

        base_name = re.sub(r"_\d{8}_\d{6}$", "", name)
        canonical = canonical_name(base_name)

        if canonical is None:
            continue

        values = safe_array(raw_values)

        fields.append(
            FieldData(
                canonical_name=canonical,
                original_name=name,
                timestamp=extract_timestamp(name),
                values=values,
            )
        )

    if x is None or y is None:
        field_shapes = [
            np.asarray(value).shape
            for key, value in raw.items()
            if not key.startswith("__")
            and canonical_name(
                re.sub(r"_\d{8}_\d{6}$", "", key)
            ) is not None
        ]

        two_dimensional = [
            shape for shape in field_shapes if len(shape) == 2
        ]

        if not two_dimensional:
            raise ValueError(
                "MAT file does not contain coordinates or 2D SWAN fields."
            )

        rows, columns = two_dimensional[0]
        grid_x, grid_y = np.meshgrid(
            np.arange(columns, dtype=float),
            np.arange(rows, dtype=float),
        )

        x = grid_x.reshape(-1)
        y = grid_y.reshape(-1)

    if x.size != y.size:
        raise ValueError("MAT coordinate arrays have different sizes.")

    triangles = triangulate_coordinates(x, y)

    mesh_data = MeshData(
        x=x,
        y=y,
        triangles=triangles,
    )

    compatible_fields: list[FieldData] = []

    for field in fields:
        if field.values.size == x.size:
            compatible_fields.append(field)

    return mesh_data, compatible_fields


def group_fields(
    fields: list[FieldData],
) -> dict[str, dict[str, FieldData]]:
    grouped: dict[str, dict[str, FieldData]] = {}

    for field in fields:
        grouped.setdefault(
            field.canonical_name,
            {},
        )[field.timestamp] = field

    return grouped


def finite_statistics(values: np.ndarray) -> dict[str, float | int]:
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
    fields: Iterable[FieldData],
    nonnegative: bool = False,
) -> tuple[float, float]:
    chunks = [
        field.values[np.isfinite(field.values)]
        for field in fields
        if np.isfinite(field.values).any()
    ]

    if not chunks:
        return 0.0, 1.0

    combined = np.concatenate(chunks)

    lower = float(np.percentile(combined, 1))
    upper = float(np.percentile(combined, 99))

    if nonnegative:
        lower = max(0.0, lower)

    if np.isclose(lower, upper):
        upper = lower + 1.0

    return lower, upper


def wave_vectors(
    direction: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    angle = direction.copy()

    if mode == "toward":
        angle = (angle + 180.0) % 360.0

    radians = np.deg2rad(angle)

    return np.sin(radians), np.cos(radians)


def create_layout(
    title: str,
    footer: str,
) -> tuple[
    plt.Figure,
    plt.Axes,
    plt.Axes,
    plt.Axes,
    plt.Axes,
]:
    figure = plt.figure(figsize=(9.5, 9.2))

    grid = figure.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[16.0, 1.3, 1.6, 0.8],
        hspace=0.38,
    )

    map_axis = figure.add_subplot(grid[0])
    colorbar_axis = figure.add_subplot(grid[1])
    legend_axis = figure.add_subplot(grid[2])
    footer_axis = figure.add_subplot(grid[3])

    map_axis.set_title(title, pad=12)
    map_axis.set_xlabel("X / Longitude")
    map_axis.set_ylabel("Y / Latitude")
    map_axis.grid(
        True,
        linewidth=0.35,
        linestyle="--",
        alpha=0.3,
    )
    map_axis.set_aspect("equal", adjustable="box")

    legend_axis.axis("off")
    footer_axis.axis("off")
    footer_axis.text(
        0.5,
        0.5,
        footer,
        ha="center",
        va="center",
        fontsize=8,
        color="0.4",
    )

    return (
        figure,
        map_axis,
        colorbar_axis,
        legend_axis,
        footer_axis,
    )


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for extension in formats:
        output = base_path.with_suffix(f".{extension}")

        kwargs = {"facecolor": "white"}

        if extension == "png":
            kwargs["dpi"] = dpi

        figure.savefig(output, **kwargs)
        saved.append(output)

    return saved


def plot_scalar(
    mesh: MeshData,
    field: FieldData,
    limits: tuple[float, float],
    label: str,
    unit: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    footer: str,
) -> Path:
    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        f"{label} — {field.timestamp}",
        footer,
    )

    triangulation = mtri.Triangulation(
        mesh.x,
        mesh.y,
        mesh.triangles,
    )

    image = axis.tripcolor(
        triangulation,
        field.values,
        shading="gouraud",
        vmin=limits[0],
        vmax=limits[1],
    )

    colorbar = figure.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(f"{label} ({unit})")
    colorbar.ax.xaxis.set_label_position("top")

    stats = finite_statistics(field.values)

    axis.text(
        0.015,
        0.02,
        (
            f"Mean: {stats['mean']:.2f} {unit}\n"
            f"Min: {stats['minimum']:.2f} {unit}\n"
            f"Max: {stats['maximum']:.2f} {unit}"
        ),
        transform=axis.transAxes,
        va="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "none",
        },
    )

    legend_axis.text(
        0.5,
        0.5,
        f"Background colors represent {label.lower()}.",
        ha="center",
        va="center",
    )

    save_figure(
        figure,
        output_base,
        formats,
        dpi,
    )

    plt.close(figure)

    return output_base.with_suffix(".png")


def plot_wave(
    mesh: MeshData,
    hs: FieldData,
    direction: FieldData,
    limits: tuple[float, float],
    vector_step: int,
    arrow_mode: str,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    footer: str,
) -> Path:
    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        f"Significant wave height and direction — {hs.timestamp}",
        footer,
    )

    triangulation = mtri.Triangulation(
        mesh.x,
        mesh.y,
        mesh.triangles,
    )

    image = axis.tripcolor(
        triangulation,
        hs.values,
        shading="gouraud",
        vmin=limits[0],
        vmax=limits[1],
    )

    u, v = wave_vectors(
        direction.values,
        arrow_mode,
    )

    valid = (
        np.isfinite(mesh.x)
        & np.isfinite(mesh.y)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    indices = np.flatnonzero(valid)[::vector_step]

    axis.quiver(
        mesh.x[indices],
        mesh.y[indices],
        u[indices],
        v[indices],
        angles="xy",
        scale_units="xy",
        scale=10.0,
        width=0.0022,
    )

    colorbar = figure.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Significant wave height (m)")
    colorbar.ax.xaxis.set_label_position("top")

    legend = (
        "Vectors indicate wave propagation direction."
        if arrow_mode == "toward"
        else "Vectors indicate the nautical direction from which waves arrive."
    )

    legend_axis.text(
        0.5,
        0.5,
        legend,
        ha="center",
        va="center",
    )

    save_figure(
        figure,
        output_base,
        formats,
        dpi,
    )

    plt.close(figure)

    return output_base.with_suffix(".png")


def plot_wind(
    mesh: MeshData,
    wind_u: FieldData,
    wind_v: FieldData,
    limits: tuple[float, float],
    vector_step: int,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
    footer: str,
) -> Path:
    u = wind_u.values
    v = wind_v.values
    speed = np.sqrt(u**2 + v**2)

    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        f"Wind speed and direction — {wind_u.timestamp}",
        footer,
    )

    triangulation = mtri.Triangulation(
        mesh.x,
        mesh.y,
        mesh.triangles,
    )

    image = axis.tripcolor(
        triangulation,
        speed,
        shading="gouraud",
        vmin=limits[0],
        vmax=limits[1],
    )

    valid = (
        np.isfinite(mesh.x)
        & np.isfinite(mesh.y)
        & np.isfinite(u)
        & np.isfinite(v)
    )

    indices = np.flatnonzero(valid)[::vector_step]

    axis.quiver(
        mesh.x[indices],
        mesh.y[indices],
        u[indices],
        v[indices],
        angles="xy",
        scale_units="xy",
        scale=55.0,
        width=0.0022,
    )

    colorbar = figure.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Wind speed (m/s)")
    colorbar.ax.xaxis.set_label_position("top")

    legend_axis.set_xlim(0, 1)
    legend_axis.set_ylim(0, 1)

    legend_axis.annotate(
        "",
        xy=(0.43, 0.67),
        xytext=(0.28, 0.67),
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.4,
            "color": "black",
        },
    )

    legend_axis.text(
        0.46,
        0.67,
        "5 m/s reference vector",
        ha="left",
        va="center",
    )

    legend_axis.text(
        0.5,
        0.22,
        (
            "Background colors show wind speed; "
            "arrows show wind direction and relative magnitude."
        ),
        ha="center",
        va="center",
        fontsize=9,
    )

    save_figure(
        figure,
        output_base,
        formats,
        dpi,
    )

    plt.close(figure)

    return output_base.with_suffix(".png")


def normalize_frames(paths: list[Path]) -> list[np.ndarray]:
    images: list[Image.Image] = []

    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    if not images:
        return []

    width = max(image.width for image in images)
    height = max(image.height for image in images)

    frames: list[np.ndarray] = []

    for image in images:
        canvas = Image.new("RGB", (width, height), "white")
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        frames.append(np.asarray(canvas, dtype=np.uint8))

    return frames


def write_gif(
    paths: list[Path],
    output: Path,
    duration: float,
) -> None:
    frames = normalize_frames(paths)

    if not frames:
        return

    output.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(
        output,
        frames,
        duration=duration,
        loop=0,
    )


def main() -> None:
    args = parse_args()

    case_dir = Path.cwd()
    case_name = Path(args.case).stem

    vtu_path = case_dir / f"{case_name}.vtu"
    mat_path = case_dir / f"{case_name}.mat"

    mesh = None
    field_list = None
    source_type = None

    if vtu_path.exists():
        print(f"Reading VTU: {vtu_path}")

        try:
            mesh, field_list = load_vtu(vtu_path)
            source_type = "VTU"

        except Exception as exc:
            print()
            print("WARNING: the VTU file could not be read.")
            print(f"Reason: {exc}")
            print(
                "This usually means that the VTU appended binary/Base64 "
                "section is malformed, truncated or incompatible with meshio."
            )

            if mat_path.exists():
                print()
                print(f"Falling back to MATLAB output: {mat_path}")
                mesh, field_list = load_mat(mat_path)
                source_type = "MAT fallback"
            else:
                raise RuntimeError(
                    "The VTU file could not be read and no MAT fallback "
                    f"was found at {mat_path}."
                ) from exc

    elif mat_path.exists():
        print(f"Reading MATLAB output: {mat_path}")
        mesh, field_list = load_mat(mat_path)
        source_type = "MAT"

    else:
        raise FileNotFoundError(
            f"Neither {vtu_path.name} nor {mat_path.name} was found."
        )

    if not field_list:
        raise RuntimeError(
            "No compatible SWAN variables were found. "
            "Inspect the VTU/MAT field names and update FIELD_ALIASES."
        )

    fields = group_fields(field_list)

    output_dir = case_dir / f"publication_{case_name}"
    figures_dir = output_dir / "figures"
    animations_dir = output_dir / "animations"
    csv_dir = output_dir / "csv"

    figures_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    footer = f"SWAN case: {case_name} • Source: {source_type}"

    summary_rows: list[dict[str, object]] = []

    timestamps = sorted(
        set().union(
            *(set(variable_fields) for variable_fields in fields.values())
        )
    )

    hs_fields = list(fields.get("hs", {}).values())
    hs_limits = robust_limits(hs_fields, nonnegative=True)

    wind_speeds: list[FieldData] = []

    if "wind_u" in fields and "wind_v" in fields:
        common_wind_times = sorted(
            set(fields["wind_u"]) & set(fields["wind_v"])
        )

        for timestamp in common_wind_times:
            speed = np.sqrt(
                fields["wind_u"][timestamp].values**2
                + fields["wind_v"][timestamp].values**2
            )

            wind_speeds.append(
                FieldData(
                    canonical_name="wind_speed",
                    original_name="WindSpeed",
                    timestamp=timestamp,
                    values=speed,
                )
            )

    wind_limits_values = robust_limits(
        wind_speeds,
        nonnegative=True,
    )

    wave_pngs: list[Path] = []
    wind_pngs: list[Path] = []

    common_wave_times = sorted(
        set(fields.get("hs", {}))
        & set(fields.get("direction", {}))
    )

    for timestamp in common_wave_times:
        hs = fields["hs"][timestamp]
        direction = fields["direction"][timestamp]

        output_base = (
            figures_dir
            / "wave"
            / f"wave_{timestamp}"
        )

        png = plot_wave(
            mesh=mesh,
            hs=hs,
            direction=direction,
            limits=hs_limits,
            vector_step=args.vector_step,
            arrow_mode=args.wave_arrow_mode,
            output_base=output_base,
            formats=args.formats,
            dpi=args.dpi,
            footer=footer,
        )

        wave_pngs.append(png)

        stats = finite_statistics(hs.values)

        summary_rows.append(
            {
                "product": "wave",
                "variable": "Hsig",
                "timestamp": timestamp,
                **stats,
            }
        )

    common_wind_times = sorted(
        set(fields.get("wind_u", {}))
        & set(fields.get("wind_v", {}))
    )

    for timestamp in common_wind_times:
        wind_u = fields["wind_u"][timestamp]
        wind_v = fields["wind_v"][timestamp]

        output_base = (
            figures_dir
            / "wind"
            / f"wind_{timestamp}"
        )

        png = plot_wind(
            mesh=mesh,
            wind_u=wind_u,
            wind_v=wind_v,
            limits=wind_limits_values,
            vector_step=args.vector_step,
            output_base=output_base,
            formats=args.formats,
            dpi=args.dpi,
            footer=footer,
        )

        wind_pngs.append(png)

        speed = np.sqrt(
            wind_u.values**2 + wind_v.values**2
        )

        stats = finite_statistics(speed)

        summary_rows.append(
            {
                "product": "wind",
                "variable": "WindSpeed",
                "timestamp": timestamp,
                **stats,
            }
        )

    for canonical in ("hs", "tm01", "tp", "direction"):
        variable_fields = fields.get(canonical, {})

        if not variable_fields:
            continue

        config = DISPLAY_CONFIG[canonical]

        limits = (
            (0.0, 360.0)
            if canonical == "direction"
            else robust_limits(
                variable_fields.values(),
                nonnegative=canonical in {"hs", "tm01", "tp"},
            )
        )

        scalar_pngs: list[Path] = []

        for timestamp, field in sorted(variable_fields.items()):
            output_base = (
                figures_dir
                / "scalar"
                / canonical
                / f"{canonical}_{timestamp}"
            )

            png = plot_scalar(
                mesh=mesh,
                field=field,
                limits=limits,
                label=config["label"],
                unit=config["unit"],
                output_base=output_base,
                formats=args.formats,
                dpi=args.dpi,
                footer=footer,
            )

            scalar_pngs.append(png)

            stats = finite_statistics(field.values)

            summary_rows.append(
                {
                    "product": "scalar",
                    "variable": canonical,
                    "timestamp": timestamp,
                    **stats,
                }
            )

        if (
            not args.no_gif
            and "png" in args.formats
            and len(scalar_pngs) > 1
        ):
            write_gif(
                scalar_pngs,
                animations_dir / f"{canonical}.gif",
                args.gif_duration,
            )

    if (
        not args.no_gif
        and "png" in args.formats
    ):
        if len(wave_pngs) > 1:
            write_gif(
                wave_pngs,
                animations_dir / "wave_height_direction.gif",
                args.gif_duration,
            )

        if len(wind_pngs) > 1:
            write_gif(
                wind_pngs,
                animations_dir / "wind_speed_direction.gif",
                args.gif_duration,
            )

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        summary.sort_values(
            ["product", "variable", "timestamp"],
            inplace=True,
        )

        summary.to_csv(
            output_dir / "summary.csv",
            index=False,
        )

        for (product, variable), table in summary.groupby(
            ["product", "variable"]
        ):
            table.to_csv(
                csv_dir / f"{product}_{variable}.csv",
                index=False,
            )

    print("\nPost-processing completed.")
    print(f"Output directory: {output_dir}")
    print(f"Fields found: {sorted(fields)}")


if __name__ == "__main__":
    main()
