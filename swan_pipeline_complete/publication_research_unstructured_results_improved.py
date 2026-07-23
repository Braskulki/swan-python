"""
Publication-ready post-processing for the SWAN unstructured case.

The paths are read from config.py when the following optional constants exist:

    UNSTRUCTURED_DIR
    UNSTRUCTURED_CASE_DIR
    UNSTRUCTURED_PUBLICATION_DIR

Default structure:

    data/unstructured/
    ├── case/
    │   ├── output_unstructured.mat
    │   ├── mesh.node
    │   ├── mesh.ele
    │   └── bottom_unstructured.txt
    └── publication/
        ├── figures/
        ├── animations/
        ├── csv/
        └── summary.csv

Usage:
    py publication_research_unstructured_results.py
    py publication_research_unstructured_results.py --skip-first
    py publication_research_unstructured_results.py --formats png pdf --dpi 300
    py publication_research_unstructured_results.py --vector-step 8
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


def _as_path(value: object, fallback: Path) -> Path:
    """
    Convert a config value to Path.

    Relative paths are resolved from the project directory, making execution
    independent from the terminal's current working directory.
    """
    if value is None:
        return fallback

    path = Path(value)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


BASE_DIR = Path(__file__).resolve().parent

try:
    import config as project_config
except ImportError:
    project_config = None


if project_config is not None:
    CONFIG_BASE_DIR = _as_path(
        getattr(project_config, "BASE_DIR", None),
        BASE_DIR,
    )
else:
    CONFIG_BASE_DIR = BASE_DIR


BASE_DIR = CONFIG_BASE_DIR

DEFAULT_UNSTRUCTURED_DIR = (
    BASE_DIR
    / "data"
    / "unstructured"
)

UNSTRUCTURED_DIR = _as_path(
    (
        getattr(project_config, "UNSTRUCTURED_DIR", None)
        if project_config is not None
        else None
    ),
    DEFAULT_UNSTRUCTURED_DIR,
)

CASE_DIR = _as_path(
    (
        getattr(project_config, "UNSTRUCTURED_CASE_DIR", None)
        if project_config is not None
        else None
    ),
    UNSTRUCTURED_DIR / "case",
)

OUTPUT_DIR = _as_path(
    (
        getattr(project_config, "UNSTRUCTURED_PUBLICATION_DIR", None)
        if project_config is not None
        else None
    ),
    UNSTRUCTURED_DIR / "publication",
)

MAT_FILE = CASE_DIR / "output_unstructured.mat"
NODE_FILE = CASE_DIR / "mesh.node"
ELE_FILE = CASE_DIR / "mesh.ele"
BOTTOM_FILE = CASE_DIR / "bottom_unstructured.txt"

FIGURES_DIR = OUTPUT_DIR / "figures"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"
CSV_DIR = OUTPUT_DIR / "csv"

KEY_PATTERN = re.compile(
    r"^(?P<variable>.+)_(?P<timestamp>\d{8}_\d{6})$"
)

DISPLAY = {
    "Hsig": ("Significant wave height", "m"),
    "Tm01": ("Mean wave period", "s"),
    "TPsmoo": ("Smoothed peak period", "s"),
    "Dir": ("Mean wave direction", "°"),
}


@dataclass(frozen=True)
class Mesh:
    x: np.ndarray
    y: np.ndarray
    triangles: np.ndarray
    depth: np.ndarray

    @property
    def triangulation(self) -> mtri.Triangulation:
        return mtri.Triangulation(
            self.x,
            self.y,
            self.triangles,
        )


@dataclass(frozen=True)
class Field:
    variable: str
    timestamp_text: str
    timestamp: datetime
    values: np.ndarray
    key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready maps from SWAN unstructured output."
        )
    )

    parser.add_argument(
        "--skip-first",
        action="store_true",
        help="Skip the first timestep as spin-up.",
    )
    parser.add_argument(
        "--vector-step",
        type=int,
        default=7,
        help=(
            "Legacy node-stride sampling, used only when the vector grid "
            "dimensions are zero."
        ),
    )
    parser.add_argument(
        "--vector-grid-nx",
        type=int,
        default=18,
        help="Uniform vector columns. Default: 18.",
    )
    parser.add_argument(
        "--vector-grid-ny",
        type=int,
        default=24,
        help="Uniform vector rows. Default: 24.",
    )
    parser.add_argument(
        "--coast-smoothing-iterations",
        type=int,
        default=3,
        help=(
            "Chaikin iterations used only for coastline drawing. "
            "Default: 3."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf"),
        default=("png", "pdf"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--wave-arrow-mode",
        choices=("toward", "from"),
        default="toward",
    )

    return parser.parse_args()


def ensure_inputs() -> None:
    expected = (
        MAT_FILE,
        NODE_FILE,
        ELE_FILE,
        BOTTOM_FILE,
    )

    missing = [
        path
        for path in expected
        if not path.is_file()
    ]

    if missing:
        config_hint = (
            "\n\nConfigure these optional variables in config.py:\n"
            "UNSTRUCTURED_DIR = BASE_DIR / 'data' / 'unstructured'\n"
            "UNSTRUCTURED_CASE_DIR = UNSTRUCTURED_DIR / 'case'\n"
            "UNSTRUCTURED_PUBLICATION_DIR = "
            "UNSTRUCTURED_DIR / 'publication'"
        )

        raise FileNotFoundError(
            "Unstructured publication inputs were not found.\n"
            f"Resolved case directory: {CASE_DIR}\n\n"
            "Missing files:\n"
            + "\n".join(f"- {path}" for path in missing)
            + config_hint
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Unstructured publication paths:")
    print(f"  Case: {CASE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")


def numeric_rows(path: Path) -> list[list[float]]:
    rows = []

    for raw_line in path.read_text(
        encoding="ascii",
        errors="ignore",
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        rows.append(
            [float(value) for value in line.split()]
        )

    return rows


def load_mesh() -> Mesh:
    node_rows = numeric_rows(NODE_FILE)
    element_rows = numeric_rows(ELE_FILE)

    node_count = int(node_rows[0][0])
    element_count = int(element_rows[0][0])

    nodes = node_rows[1:1 + node_count]
    elements = element_rows[1:1 + element_count]

    node_ids = np.asarray(
        [int(row[0]) for row in nodes],
        dtype=int,
    )
    x = np.asarray(
        [row[1] for row in nodes],
        dtype=float,
    )
    y = np.asarray(
        [row[2] for row in nodes],
        dtype=float,
    )

    id_to_index = {
        int(node_id): index
        for index, node_id in enumerate(node_ids)
    }

    triangles = np.asarray(
        [
            [
                id_to_index[int(row[1])],
                id_to_index[int(row[2])],
                id_to_index[int(row[3])],
            ]
            for row in elements
        ],
        dtype=int,
    )

    depth = np.loadtxt(
        BOTTOM_FILE,
        dtype=float,
    ).reshape(-1)

    if depth.size != x.size:
        raise ValueError(
            f"Depth count {depth.size} does not match node count {x.size}."
        )

    return Mesh(
        x=x,
        y=y,
        triangles=triangles,
        depth=depth,
    )


def safe_array(raw: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", over="ignore"):
        values = np.asarray(
            raw,
            dtype=np.float64,
        ).reshape(-1).copy()

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
    fields: Iterable[Field],
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
    figure = plt.figure(figsize=(9.4, 9.4))
    grid = figure.add_gridspec(
        4,
        1,
        height_ratios=[16.0, 1.4, 1.8, 0.9],
        hspace=0.42,
    )

    map_axis = figure.add_subplot(grid[0])
    colorbar_axis = figure.add_subplot(grid[1])
    legend_axis = figure.add_subplot(grid[2])
    footer_axis = figure.add_subplot(grid[3])

    map_axis.set_title(title, pad=12, fontsize=12)
    map_axis.set_xlabel("Longitude (°)")
    map_axis.set_ylabel("Latitude (°)")
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.grid(
        True,
        linewidth=0.35,
        linestyle="--",
        alpha=0.3,
    )

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
) -> Path:
    base_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for extension in formats:
        output = base_path.with_suffix(f".{extension}")
        kwargs = {"facecolor": "white"}

        if extension == "png":
            kwargs["dpi"] = dpi

        figure.savefig(output, **kwargs)

    return base_path.with_suffix(".png")


def add_stats_box(
    axis: plt.Axes,
    values: np.ndarray,
    unit: str,
) -> None:
    stats = statistics(values)

    if stats["valid_points"] == 0:
        return

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



def boundary_edges(mesh: Mesh) -> np.ndarray:
    triangles = np.asarray(mesh.triangles, dtype=int)
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    ordered = np.sort(edges, axis=1)
    unique, counts = np.unique(ordered, axis=0, return_counts=True)
    return unique[counts == 1]


def ordered_boundary_chains(edges: np.ndarray) -> list[np.ndarray]:
    adjacency: dict[int, list[int]] = {}
    for node_a, node_b in edges:
        adjacency.setdefault(int(node_a), []).append(int(node_b))
        adjacency.setdefault(int(node_b), []).append(int(node_a))

    unused = {
        tuple(sorted((int(node_a), int(node_b))))
        for node_a, node_b in edges
    }
    chains: list[np.ndarray] = []

    while unused:
        seed = next(iter(unused))
        component = set(seed)
        stack = list(seed)

        while stack:
            node = stack.pop()
            for neighbour in adjacency.get(node, []):
                edge = tuple(sorted((node, neighbour)))
                if edge in unused and neighbour not in component:
                    component.add(neighbour)
                    stack.append(neighbour)

        endpoints = [
            node
            for node in component
            if sum(
                tuple(sorted((node, neighbour))) in unused
                for neighbour in adjacency.get(node, [])
            ) == 1
        ]
        start = endpoints[0] if endpoints else next(iter(component))
        chain = [start]
        previous = None
        current = start

        while True:
            candidates = [
                neighbour
                for neighbour in adjacency.get(current, [])
                if tuple(sorted((current, neighbour))) in unused
                and neighbour != previous
            ]
            if not candidates:
                break

            next_node = candidates[0]
            unused.remove(tuple(sorted((current, next_node))))
            chain.append(next_node)
            previous, current = current, next_node

            if current == start:
                break

        if len(chain) >= 2:
            chains.append(np.asarray(chain, dtype=int))

    return chains


def chaikin(
    coordinates: np.ndarray,
    iterations: int,
    closed: bool,
) -> np.ndarray:
    points = np.asarray(coordinates, dtype=float)

    if iterations <= 0 or points.shape[0] < 3:
        return points

    if closed and not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))

    for _ in range(iterations):
        refined: list[np.ndarray] = []

        if not closed:
            refined.append(points[0])

        for index in range(points.shape[0] - 1):
            point_a = points[index]
            point_b = points[index + 1]
            refined.append(0.75 * point_a + 0.25 * point_b)
            refined.append(0.25 * point_a + 0.75 * point_b)

        if closed:
            refined.append(refined[0])
        else:
            refined.append(points[-1])

        points = np.asarray(refined, dtype=float)

    return points


def add_domain_boundary(
    axis: plt.Axes,
    mesh: Mesh,
    smoothing_iterations: int,
) -> None:
    x_min = float(np.nanmin(mesh.x))
    x_max = float(np.nanmax(mesh.x))
    y_min = float(np.nanmin(mesh.y))
    y_max = float(np.nanmax(mesh.y))
    tolerance = max(x_max - x_min, y_max - y_min) * 1.0e-5

    for chain in ordered_boundary_chains(boundary_edges(mesh)):
        coordinates = np.column_stack((mesh.x[chain], mesh.y[chain]))
        closed = np.linalg.norm(coordinates[0] - coordinates[-1]) <= tolerance
        open_boundary = (
            np.all(np.abs(coordinates[:, 0] - x_max) <= tolerance)
            or np.all(np.abs(coordinates[:, 1] - y_min) <= tolerance)
        )

        plotted = (
            coordinates
            if open_boundary
            else chaikin(coordinates, smoothing_iterations, closed)
        )

        axis.plot(
            plotted[:, 0],
            plotted[:, 1],
            linewidth=0.8 if open_boundary else 1.35,
            alpha=0.95,
            zorder=8,
        )


def sample_vector_field(
    mesh: Mesh,
    u_values: np.ndarray,
    v_values: np.ndarray,
    grid_nx: int,
    grid_ny: int,
    vector_step: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_nodes = (
        np.isfinite(mesh.x)
        & np.isfinite(mesh.y)
        & np.isfinite(u_values)
        & np.isfinite(v_values)
    )

    if grid_nx <= 0 or grid_ny <= 0:
        indices = np.flatnonzero(valid_nodes)[::max(1, vector_step)]
        u = np.asarray(u_values[indices], dtype=float)
        v = np.asarray(v_values[indices], dtype=float)

        if normalize:
            magnitude = np.hypot(u, v)
            good = magnitude > 1.0e-12
            return (
                mesh.x[indices][good],
                mesh.y[indices][good],
                u[good] / magnitude[good],
                v[good] / magnitude[good],
            )

        return mesh.x[indices], mesh.y[indices], u, v

    x_axis = np.linspace(float(np.nanmin(mesh.x)), float(np.nanmax(mesh.x)), grid_nx)
    y_axis = np.linspace(float(np.nanmin(mesh.y)), float(np.nanmax(mesh.y)), grid_ny)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)

    u_interp = mtri.LinearTriInterpolator(
        mesh.triangulation,
        np.asarray(u_values, dtype=float),
    )
    v_interp = mtri.LinearTriInterpolator(
        mesh.triangulation,
        np.asarray(v_values, dtype=float),
    )

    sampled_u = np.ma.asarray(u_interp(grid_x, grid_y))
    sampled_v = np.ma.asarray(v_interp(grid_x, grid_y))
    mask = (
        np.ma.getmaskarray(sampled_u)
        | np.ma.getmaskarray(sampled_v)
        | ~np.isfinite(sampled_u.filled(np.nan))
        | ~np.isfinite(sampled_v.filled(np.nan))
    )

    x = grid_x[~mask]
    y = grid_y[~mask]
    u = sampled_u.filled(np.nan)[~mask]
    v = sampled_v.filled(np.nan)[~mask]

    if normalize:
        magnitude = np.hypot(u, v)
        good = magnitude > 1.0e-12
        x = x[good]
        y = y[good]
        u = u[good] / magnitude[good]
        v = v[good] / magnitude[good]

    return x, y, u, v


def draw_vectors(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normalized: bool,
) -> None:
    if x.size == 0:
        return

    domain_width = float(axis.get_xlim()[1] - axis.get_xlim()[0])
    domain_height = float(axis.get_ylim()[1] - axis.get_ylim()[0])
    reference = min(domain_width, domain_height)

    if normalized:
        factor = reference * 0.040
    else:
        magnitude = np.hypot(u, v)
        typical = float(np.nanpercentile(magnitude, 75))
        if not np.isfinite(typical) or typical <= 0:
            typical = 1.0
        factor = reference * 0.045 / typical

    axis.quiver(
        x,
        y,
        u * factor,
        v * factor,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0020,
        headwidth=3.4,
        headlength=4.4,
        headaxislength=3.8,
        pivot="middle",
        zorder=10,
    )


def plot_mesh(mesh: Mesh, formats: Iterable[str], dpi: int) -> None:
    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        "SWAN unstructured computational mesh",
        "SWAN 41.51A • GEBCO • ERA5",
    )

    axis.triplot(
        mesh.triangulation,
        linewidth=0.18,
        alpha=0.58,
    )
    add_domain_boundary(axis, mesh, smoothing_iterations=0)

    triangle_points = np.column_stack((mesh.x, mesh.y))[mesh.triangles]
    triangle_area = 0.5 * np.abs(
        (
            triangle_points[:, 1, 0] - triangle_points[:, 0, 0]
        )
        * (
            triangle_points[:, 2, 1] - triangle_points[:, 0, 1]
        )
        - (
            triangle_points[:, 1, 1] - triangle_points[:, 0, 1]
        )
        * (
            triangle_points[:, 2, 0] - triangle_points[:, 0, 0]
        )
    )
    positive_area = triangle_area[triangle_area > 0]
    area_ratio = (
        float(np.max(positive_area) / np.min(positive_area))
        if positive_area.size
        else float("nan")
    )

    cbar_axis.axis("off")
    legend_axis.text(
        0.5,
        0.5,
        (
            f"{mesh.x.size} nodes • "
            f"{mesh.triangles.shape[0]} triangular elements • "
            f"max/min area ratio: {area_ratio:.1f}"
        ),
        ha="center",
        va="center",
    )

    save_figure(
        figure,
        FIGURES_DIR / "mesh" / "triangular_mesh",
        formats,
        dpi,
    )
    plt.close(figure)


def plot_bathymetry(
    mesh: Mesh,
    formats: Iterable[str],
    dpi: int,
) -> None:
    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        "Bathymetry and triangular mesh",
        "SWAN 41.51A • GEBCO • ERA5",
    )

    valid = mesh.depth[np.isfinite(mesh.depth)]
    limits = (
        float(np.percentile(valid, 1)),
        float(np.percentile(valid, 99)),
    )

    image = axis.tripcolor(
        mesh.triangulation,
        mesh.depth,
        shading="flat",
        vmin=limits[0],
        vmax=limits[1],
    )

    axis.triplot(
        mesh.triangulation,
        linewidth=0.12,
        alpha=0.3,
    )

    colorbar = figure.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Water depth (m)")
    colorbar.ax.xaxis.set_label_position("top")

    legend_axis.text(
        0.5,
        0.5,
        "Filled colors represent depth; thin lines represent triangular elements.",
        ha="center",
        va="center",
    )

    add_domain_boundary(
        axis,
        mesh,
        smoothing_iterations=3,
    )
    add_stats_box(axis, mesh.depth, "m")

    save_figure(
        figure,
        FIGURES_DIR / "bathymetry" / "bathymetry_mesh",
        formats,
        dpi,
    )
    plt.close(figure)


def plot_wave(
    mesh: Mesh,
    hs: Field,
    direction: Field,
    limits: tuple[float, float],
    vector_step: int,
    arrow_mode: str,
    formats: Iterable[str],
    dpi: int,
    args: argparse.Namespace,
) -> Path:
    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        (
            "Significant wave height and direction — "
            f"{hs.timestamp:%Y-%m-%d %H:%M UTC}"
        ),
        "SWAN 41.51A • GEBCO • ERA5",
    )

    image = axis.tricontourf(
        mesh.triangulation,
        hs.values,
        levels=np.linspace(limits[0], limits[1], 40),
        extend="both",
    )

    u, v = wave_vectors(
        direction.values,
        arrow_mode,
    )

    vector_x, vector_y, vector_u, vector_v = sample_vector_field(
        mesh,
        u,
        v,
        args.vector_grid_nx,
        args.vector_grid_ny,
        vector_step,
        normalize=True,
    )
    draw_vectors(
        axis,
        vector_x,
        vector_y,
        vector_u,
        vector_v,
        normalized=True,
    )
    add_domain_boundary(
        axis,
        mesh,
        args.coast_smoothing_iterations,
    )

    colorbar = figure.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Significant wave height (m)")
    colorbar.ax.xaxis.set_label_position("top")

    legend_axis.text(
        0.5,
        0.5,
        (
            "Vectors indicate wave propagation direction."
            if arrow_mode == "toward"
            else "Vectors indicate the nautical direction from which waves arrive."
        ),
        ha="center",
        va="center",
    )

    add_stats_box(axis, hs.values, "m")

    output = save_figure(
        figure,
        FIGURES_DIR / "wave" / f"wave_{hs.timestamp_text}",
        formats,
        dpi,
    )
    plt.close(figure)

    return output


def plot_wind(
    mesh: Mesh,
    wind_u: Field,
    wind_v: Field,
    limits: tuple[float, float],
    vector_step: int,
    formats: Iterable[str],
    dpi: int,
    args: argparse.Namespace,
) -> Path:
    speed = np.sqrt(
        wind_u.values**2 + wind_v.values**2
    )

    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        (
            "Wind speed and direction — "
            f"{wind_u.timestamp:%Y-%m-%d %H:%M UTC}"
        ),
        "SWAN 41.51A • GEBCO • ERA5",
    )

    image = axis.tricontourf(
        mesh.triangulation,
        speed,
        levels=np.linspace(limits[0], limits[1], 40),
        extend="both",
    )

    vector_x, vector_y, vector_u, vector_v = sample_vector_field(
        mesh,
        wind_u.values,
        wind_v.values,
        args.vector_grid_nx,
        args.vector_grid_ny,
        vector_step,
        normalize=False,
    )
    draw_vectors(
        axis,
        vector_x,
        vector_y,
        vector_u,
        vector_v,
        normalized=False,
    )
    add_domain_boundary(
        axis,
        mesh,
        args.coast_smoothing_iterations,
    )

    colorbar = figure.colorbar(
        image,
        cax=cbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label("Wind speed (m/s)")
    colorbar.ax.xaxis.set_label_position("top")

    legend_axis.text(
        0.5,
        0.5,
        (
            "Background colors show wind speed; arrows show wind direction "
            "and relative magnitude."
        ),
        ha="center",
        va="center",
    )

    add_stats_box(axis, speed, "m/s")

    output = save_figure(
        figure,
        FIGURES_DIR / "wind" / f"wind_{wind_u.timestamp_text}",
        formats,
        dpi,
    )
    plt.close(figure)

    return output


def plot_scalar(
    mesh: Mesh,
    field: Field,
    limits: tuple[float, float],
    formats: Iterable[str],
    dpi: int,
) -> Path:
    label, unit = DISPLAY[field.variable]

    figure, axis, cbar_axis, legend_axis, _ = create_layout(
        f"{label} — {field.timestamp:%Y-%m-%d %H:%M UTC}",
        "SWAN 41.51A • GEBCO • ERA5",
    )

    values = field.values.copy()

    if field.variable == "Dir":
        finite = np.isfinite(values)
        values[finite] %= 360.0

    image = axis.tripcolor(
        mesh.triangulation,
        values,
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

    legend_axis.text(
        0.5,
        0.5,
        f"Background colors represent {label.lower()}.",
        ha="center",
        va="center",
    )

    add_domain_boundary(
        axis,
        mesh,
        smoothing_iterations=3,
    )
    add_stats_box(axis, values, unit)

    output = save_figure(
        figure,
        FIGURES_DIR / "scalar" / field.variable / field.key,
        formats,
        dpi,
    )
    plt.close(figure)

    return output


def normalize_frames(paths: list[Path]) -> list[np.ndarray]:
    images = []

    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())

    if not images:
        return []

    width = max(image.width for image in images)
    height = max(image.height for image in images)

    frames = []

    for image in images:
        canvas = Image.new(
            "RGB",
            (width, height),
            "white",
        )
        canvas.paste(
            image,
            (
                (width - image.width) // 2,
                (height - image.height) // 2,
            ),
        )
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

    if args.vector_step < 1:
        raise ValueError("--vector-step must be at least 1.")

    ensure_inputs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh()
    fields = load_fields()

    for variable_fields in fields.values():
        for field in variable_fields.values():
            if field.values.size != mesh.x.size:
                raise ValueError(
                    f"{field.key} has {field.values.size} values; "
                    f"mesh has {mesh.x.size} nodes."
                )

    plot_mesh(mesh, args.formats, args.dpi)
    plot_bathymetry(mesh, args.formats, args.dpi)

    summary_rows = []

    wave_times = sorted(
        set(fields.get("Hsig", {}))
        & set(fields.get("Dir", {}))
    )
    wind_times = sorted(
        set(fields.get("Windv_x", {}))
        & set(fields.get("Windv_y", {}))
    )

    if args.skip_first:
        wave_times = wave_times[1:]
        wind_times = wind_times[1:]

    hs_limits = robust_limits(
        [fields["Hsig"][time] for time in wave_times],
        nonnegative=True,
    )

    wave_pngs = []

    for time in wave_times:
        hs = fields["Hsig"][time]
        direction = fields["Dir"][time]

        wave_pngs.append(
            plot_wave(
                mesh,
                hs,
                direction,
                hs_limits,
                args.vector_step,
                args.wave_arrow_mode,
                args.formats,
                args.dpi,
                args,
            )
        )

        summary_rows.append(
            {
                "product": "wave",
                "variable": "Hsig",
                "timestamp": hs.timestamp.isoformat(),
                **statistics(hs.values),
            }
        )

    wind_speed_fields = []

    for time in wind_times:
        speed = np.sqrt(
            fields["Windv_x"][time].values**2
            + fields["Windv_y"][time].values**2
        )
        wind_speed_fields.append(
            Field(
                variable="WindSpeed",
                timestamp_text=time,
                timestamp=fields["Windv_x"][time].timestamp,
                values=speed,
                key=f"WindSpeed_{time}",
            )
        )

    wind_limits = robust_limits(
        wind_speed_fields,
        nonnegative=True,
    )

    wind_pngs = []

    for time in wind_times:
        wind_u = fields["Windv_x"][time]
        wind_v = fields["Windv_y"][time]

        wind_pngs.append(
            plot_wind(
                mesh,
                wind_u,
                wind_v,
                wind_limits,
                args.vector_step,
                args.formats,
                args.dpi,
                args,
            )
        )

        speed = np.sqrt(
            wind_u.values**2 + wind_v.values**2
        )

        summary_rows.append(
            {
                "product": "wind",
                "variable": "WindSpeed",
                "timestamp": wind_u.timestamp.isoformat(),
                **statistics(speed),
            }
        )

    for variable in ("Hsig", "Tm01", "TPsmoo", "Dir"):
        variable_fields = fields.get(variable, {})

        if not variable_fields:
            continue

        times = sorted(variable_fields)

        if args.skip_first:
            times = times[1:]

        selected = [
            variable_fields[time]
            for time in times
        ]

        limits = (
            (0.0, 360.0)
            if variable == "Dir"
            else robust_limits(
                selected,
                nonnegative=variable != "Dir",
            )
        )

        scalar_pngs = []

        for field in selected:
            scalar_pngs.append(
                plot_scalar(
                    mesh,
                    field,
                    limits,
                    args.formats,
                    args.dpi,
                )
            )

            summary_rows.append(
                {
                    "product": "scalar",
                    "variable": variable,
                    "timestamp": field.timestamp.isoformat(),
                    **statistics(field.values),
                }
            )

        if (
            not args.no_gif
            and "png" in args.formats
            and len(scalar_pngs) > 1
        ):
            write_gif(
                scalar_pngs,
                ANIMATIONS_DIR / f"{variable}.gif",
                args.gif_duration,
            )

    if (
        not args.no_gif
        and "png" in args.formats
    ):
        if len(wave_pngs) > 1:
            write_gif(
                wave_pngs,
                ANIMATIONS_DIR / "wave_height_direction.gif",
                args.gif_duration,
            )

        if len(wind_pngs) > 1:
            write_gif(
                wind_pngs,
                ANIMATIONS_DIR / "wind_speed_direction.gif",
                args.gif_duration,
            )

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        summary.sort_values(
            ["product", "variable", "timestamp"],
            inplace=True,
        )
        summary.to_csv(
            OUTPUT_DIR / "summary.csv",
            index=False,
        )

        for (product, variable), table in summary.groupby(
            ["product", "variable"]
        ):
            table.to_csv(
                CSV_DIR / f"{product}_{variable}.csv",
                index=False,
            )

    print("\nUnstructured publication post-processing completed.")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
