"""
Generate a conforming SWAN unstructured case from the existing regular
swan-pipeline-complete domain.

This version fixes non-manifold boundary nodes by triangulating the wet
regular cells instead of applying unconstrained Delaunay triangulation to
all wet points.

Inputs:
    data/processed/grid.json
    data/processed/depth.bot
    data/raw/wind.nc
    data/processed/boundary_east.txt
    data/processed/boundary_south.txt

Outputs:
    data/unstructured/
    ├── mesh.node
    ├── mesh.ele
    ├── mesh.edge
    ├── mesh.poly
    ├── bottom_unstructured.txt
    ├── wind_unstructured.txt
    ├── boundary_east.txt
    ├── boundary_south.txt
    ├── mesh_metadata.json
    ├── mesh_preview.png
    └── INPUT

Mesh strategy:
    * Select the regular grid at the requested subsampling interval.
    * Retain only quadrilateral cells whose four corner nodes are wet.
    * Divide each retained quadrilateral into two counterclockwise triangles.
    * Compact unused nodes.
    * Derive actual exterior edges from triangle connectivity.
    * Split the east and south open boundaries into connected edge components.
    * Write each component explicitly through BOUNDSPEC SEGMENT IJ.

Boundary marker convention:
    1       coastline / other exterior boundary
    20+     east open-boundary components
    30+     south open-boundary components

Usage:
    py generate_unstructured_pipeline.py --subsample 2
    py generate_unstructured_pipeline.py --subsample 2 --run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "unstructured"

GRID_FILE = PROCESSED_DIR / "grid.json"
DEPTH_FILE = PROCESSED_DIR / "depth.bot"
WIND_FILE = RAW_DIR / "wind.nc"

BOUNDARY_EAST_FILE = PROCESSED_DIR / "boundary_east.txt"
BOUNDARY_SOUTH_FILE = PROCESSED_DIR / "boundary_south.txt"

MESH_BASENAME = "mesh"
NODE_FILE = OUTPUT_DIR / "mesh.node"
ELE_FILE = OUTPUT_DIR / "mesh.ele"
EDGE_FILE = OUTPUT_DIR / "mesh.edge"
POLY_FILE = OUTPUT_DIR / "mesh.poly"
BOTTOM_FILE = OUTPUT_DIR / "bottom_unstructured.txt"
WIND_OUTPUT_FILE = OUTPUT_DIR / "wind_unstructured.txt"
INPUT_FILE = OUTPUT_DIR / "INPUT"
METADATA_FILE = OUTPUT_DIR / "mesh_metadata.json"
PREVIEW_FILE = OUTPUT_DIR / "mesh_preview.png"

DIRECTION_BINS = 36
MIN_FREQUENCY_HZ = 0.04
MAX_FREQUENCY_HZ = 1.0

COAST_MARKER = 1
EAST_MARKER_BASE = 20
SOUTH_MARKER_BASE = 30

DEFAULT_DOCKER_IMAGE = os.getenv(
    "SWAN_DOCKER_IMAGE",
    "openeuler/swan:latest",
)
DEFAULT_SWAN_EXECUTABLE = os.getenv(
    "SWAN_EXECUTABLE",
    "/opt/swan/swan.exe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a conforming triangular SWAN mesh and complete "
            "unstructured input files."
        )
    )

    parser.add_argument(
        "--subsample",
        type=int,
        default=2,
        help=(
            "Keep one regular-grid point every N points in both axes. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--compute-step-minutes",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--output-step-hours",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--run",
        action="store_true",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
    )
    parser.add_argument(
        "--swan-executable",
        default=DEFAULT_SWAN_EXECUTABLE,
    )

    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


def find_name(
    dataset: xr.Dataset,
    candidates: tuple[str, ...],
) -> str:
    for name in candidates:
        if (
            name in dataset.variables
            or name in dataset.coords
            or name in dataset.dims
        ):
            return name

    raise KeyError(
        f"None of {candidates} was found. "
        f"Available: {list(dataset.variables)}"
    )


def format_swan_time(value: np.datetime64) -> str:
    text = np.datetime_as_string(value, unit="s")
    date, clock = text.split("T")

    return (
        date.replace("-", "")
        + "."
        + clock.replace(":", "")
    )


def constant_time_step_hours(times: np.ndarray) -> float:
    seconds = np.diff(
        times.astype("datetime64[s]").astype(np.int64)
    )

    if seconds.size == 0:
        raise ValueError(
            "At least two wind timestamps are required."
        )

    if np.any(seconds <= 0) or not np.all(seconds == seconds[0]):
        raise ValueError(
            "Wind timestamps must be strictly increasing "
            "with a constant interval."
        )

    return float(seconds[0]) / 3600.0


def load_regular_domain() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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

    expected = (latitude.size, longitude.size)

    if depth.shape != expected:
        raise ValueError(
            f"depth.bot has shape {depth.shape}; expected {expected}."
        )

    return longitude, latitude, depth


def triangle_signed_area(
    points: np.ndarray,
    triangle: np.ndarray,
) -> float:
    a, b, c = points[triangle]

    return 0.5 * (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def build_conforming_mesh(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    subsample: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulates only fully wet regular cells.

    A retained rectangular cell is split using an alternating diagonal to
    avoid a systematic directional bias.
    """
    step = max(1, int(subsample))

    lon = longitude[::step]
    lat = latitude[::step]
    sampled_depth = depth[::step, ::step]

    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    wet = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0)
    )

    full_points = np.column_stack(
        (
            lon_mesh.reshape(-1),
            lat_mesh.reshape(-1),
        )
    )
    full_depth = sampled_depth.reshape(-1)

    rows, columns = sampled_depth.shape

    def node_index(row: int, column: int) -> int:
        return row * columns + column

    triangles: list[list[int]] = []

    for row in range(rows - 1):
        for column in range(columns - 1):
            corners_wet = (
                wet[row, column]
                and wet[row, column + 1]
                and wet[row + 1, column]
                and wet[row + 1, column + 1]
            )

            if not corners_wet:
                continue

            lower_left = node_index(row, column)
            lower_right = node_index(row, column + 1)
            upper_left = node_index(row + 1, column)
            upper_right = node_index(row + 1, column + 1)

            if (row + column) % 2 == 0:
                cell_triangles = (
                    [lower_left, lower_right, upper_right],
                    [lower_left, upper_right, upper_left],
                )
            else:
                cell_triangles = (
                    [lower_left, lower_right, upper_left],
                    [lower_right, upper_right, upper_left],
                )

            for triangle in cell_triangles:
                triangle_array = np.asarray(
                    triangle,
                    dtype=int,
                )

                if (
                    triangle_signed_area(
                        full_points,
                        triangle_array,
                    )
                    < 0
                ):
                    triangle_array[[1, 2]] = (
                        triangle_array[[2, 1]]
                    )

                triangles.append(
                    triangle_array.tolist()
                )

    if not triangles:
        raise RuntimeError(
            "No fully wet cells were available for triangulation."
        )

    triangle_array = np.asarray(
        triangles,
        dtype=int,
    )

    used_nodes = np.unique(triangle_array.reshape(-1))
    mapping = np.full(
        full_points.shape[0],
        -1,
        dtype=int,
    )
    mapping[used_nodes] = np.arange(used_nodes.size)

    points = full_points[used_nodes]
    node_depth = full_depth[used_nodes]
    triangle_array = mapping[triangle_array]

    return points, node_depth, triangle_array


def get_boundary_edges(
    triangles: np.ndarray,
) -> np.ndarray:
    counter: Counter[tuple[int, int]] = Counter()

    for n1, n2, n3 in triangles:
        counter.update(
            (
                tuple(sorted((int(n1), int(n2)))),
                tuple(sorted((int(n2), int(n3)))),
                tuple(sorted((int(n3), int(n1)))),
            )
        )

    edges = [
        edge
        for edge, count in counter.items()
        if count == 1
    ]

    if not edges:
        raise RuntimeError(
            "No exterior edges were found."
        )

    return np.asarray(edges, dtype=int)


def connected_edge_components(
    edges: np.ndarray,
) -> list[np.ndarray]:
    if edges.size == 0:
        return []

    node_to_edges: dict[int, list[int]] = defaultdict(list)

    for edge_index, (node_a, node_b) in enumerate(edges):
        node_to_edges[int(node_a)].append(edge_index)
        node_to_edges[int(node_b)].append(edge_index)

    remaining = set(range(edges.shape[0]))
    components: list[np.ndarray] = []

    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        indices = [seed]

        while queue:
            edge_index = queue.popleft()
            node_a, node_b = edges[edge_index]

            for node in (int(node_a), int(node_b)):
                for neighbour in node_to_edges[node]:
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        queue.append(neighbour)
                        indices.append(neighbour)

        components.append(
            edges[np.asarray(indices, dtype=int)]
        )

    return components


def classify_boundary_edges(
    points: np.ndarray,
    boundary_edges: np.ndarray,
    original_lon: np.ndarray,
    original_lat: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[int, np.ndarray],
]:
    """
    Assigns one marker to every connected east/south boundary component.

    Returns:
        edge_markers:
            marker for each row of boundary_edges.
        segment_nodes:
            ordered node arrays keyed by marker.
    """
    dx = abs(float(np.median(np.diff(original_lon))))
    dy = abs(float(np.median(np.diff(original_lat))))

    lon_max = float(original_lon.max())
    lat_min = float(original_lat.min())

    edge_points_a = points[boundary_edges[:, 0]]
    edge_points_b = points[boundary_edges[:, 1]]

    east_mask = (
        np.abs(edge_points_a[:, 0] - lon_max) <= dx * 1.25
    ) & (
        np.abs(edge_points_b[:, 0] - lon_max) <= dx * 1.25
    )

    south_mask = (
        np.abs(edge_points_a[:, 1] - lat_min) <= dy * 1.25
    ) & (
        np.abs(edge_points_b[:, 1] - lat_min) <= dy * 1.25
    )

    # Southeast corner edge belongs to the east group first.
    south_mask &= ~east_mask

    edge_markers = np.full(
        boundary_edges.shape[0],
        COAST_MARKER,
        dtype=int,
    )
    segment_nodes: dict[int, np.ndarray] = {}

    for component_index, component in enumerate(
        connected_edge_components(
            boundary_edges[east_mask]
        )
    ):
        marker = EAST_MARKER_BASE + component_index
        component_set = {
            tuple(sorted(map(int, edge)))
            for edge in component
        }

        for edge_index, edge in enumerate(boundary_edges):
            if tuple(sorted(map(int, edge))) in component_set:
                edge_markers[edge_index] = marker

        nodes = np.unique(component.reshape(-1))
        # Counterclockwise order on the east side: south -> north.
        nodes = nodes[
            np.argsort(points[nodes, 1])
        ]
        segment_nodes[marker] = nodes

    for component_index, component in enumerate(
        connected_edge_components(
            boundary_edges[south_mask]
        )
    ):
        marker = SOUTH_MARKER_BASE + component_index
        component_set = {
            tuple(sorted(map(int, edge)))
            for edge in component
        }

        for edge_index, edge in enumerate(boundary_edges):
            if tuple(sorted(map(int, edge))) in component_set:
                edge_markers[edge_index] = marker

        nodes = np.unique(component.reshape(-1))
        # Counterclockwise order on the south side: west -> east.
        nodes = nodes[
            np.argsort(points[nodes, 0])
        ]
        segment_nodes[marker] = nodes

    east_markers = [
        marker
        for marker in segment_nodes
        if EAST_MARKER_BASE <= marker < SOUTH_MARKER_BASE
    ]
    south_markers = [
        marker
        for marker in segment_nodes
        if marker >= SOUTH_MARKER_BASE
    ]

    if not east_markers:
        raise ValueError(
            "No east open-boundary component was found."
        )

    if not south_markers:
        raise ValueError(
            "No south open-boundary component was found."
        )

    return edge_markers, segment_nodes


def node_markers_from_edges(
    node_count: int,
    boundary_edges: np.ndarray,
    edge_markers: np.ndarray,
) -> np.ndarray:
    markers = np.zeros(node_count, dtype=int)

    for edge, marker in zip(boundary_edges, edge_markers):
        for node in edge:
            current = markers[int(node)]

            if current == 0 or current == COAST_MARKER:
                markers[int(node)] = int(marker)

    return markers


def write_triangle_files(
    points: np.ndarray,
    triangles: np.ndarray,
    boundary_edges: np.ndarray,
    edge_markers: np.ndarray,
    node_markers: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with NODE_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        stream.write(
            f"{points.shape[0]} 2 0 1\n"
        )

        for node_id, ((x, y), marker) in enumerate(
            zip(points, node_markers),
            start=1,
        ):
            stream.write(
                f"{node_id} {x:.10f} {y:.10f} {int(marker)}\n"
            )

    with ELE_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        stream.write(
            f"{triangles.shape[0]} 3 0\n"
        )

        for element_id, triangle in enumerate(
            triangles,
            start=1,
        ):
            n1, n2, n3 = triangle + 1
            stream.write(
                f"{element_id} {n1} {n2} {n3}\n"
            )

    with EDGE_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        stream.write(
            f"{boundary_edges.shape[0]} 1\n"
        )

        for edge_id, (edge, marker) in enumerate(
            zip(boundary_edges, edge_markers),
            start=1,
        ):
            n1, n2 = edge + 1
            stream.write(
                f"{edge_id} {n1} {n2} {int(marker)}\n"
            )

    # Triangle .poly format using the existing vertices by reference.
    with POLY_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        # Zero points means: read points from the associated .node file.
        stream.write("0 2 0 1\n")
        stream.write(
            f"{boundary_edges.shape[0]} 1\n"
        )

        for segment_id, (edge, marker) in enumerate(
            zip(boundary_edges, edge_markers),
            start=1,
        ):
            n1, n2 = edge + 1
            stream.write(
                f"{segment_id} {n1} {n2} {int(marker)}\n"
            )

        # No holes and no regional attributes.
        stream.write("0\n")
        stream.write("0\n")


def write_bottom_file(depth: np.ndarray) -> None:
    np.savetxt(
        BOTTOM_FILE,
        depth,
        fmt="%.6f",
    )


def generate_unstructured_wind(
    points: np.ndarray,
) -> tuple[np.ndarray, float]:
    with xr.open_dataset(WIND_FILE) as dataset:
        lon_name = find_name(
            dataset,
            ("longitude", "lon"),
        )
        lat_name = find_name(
            dataset,
            ("latitude", "lat"),
        )
        time_name = find_name(
            dataset,
            ("valid_time", "time", "forecast_time"),
        )
        u_name = find_name(
            dataset,
            ("u10", "10u"),
        )
        v_name = find_name(
            dataset,
            ("v10", "10v"),
        )

        dataset = (
            dataset
            .sortby(lon_name)
            .sortby(lat_name)
            .sortby(time_name)
        )

        target_lon = xr.DataArray(
            points[:, 0],
            dims=("node",),
        )
        target_lat = xr.DataArray(
            points[:, 1],
            dims=("node",),
        )

        u = (
            dataset[u_name]
            .squeeze(drop=True)
            .interp(
                {
                    lon_name: target_lon,
                    lat_name: target_lat,
                },
                method="linear",
            )
            .transpose(time_name, "node")
        )
        v = (
            dataset[v_name]
            .squeeze(drop=True)
            .interp(
                {
                    lon_name: target_lon,
                    lat_name: target_lat,
                },
                method="linear",
            )
            .transpose(time_name, "node")
        )

        times = np.asarray(
            dataset[time_name].values
        )
        u_values = np.asarray(
            u.values,
            dtype=np.float64,
        )
        v_values = np.asarray(
            v.values,
            dtype=np.float64,
        )

    if (
        not np.isfinite(u_values).all()
        or not np.isfinite(v_values).all()
    ):
        raise ValueError(
            "Wind interpolation generated NaN values."
        )

    with WIND_OUTPUT_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        for time_index in range(times.size):
            for value in u_values[time_index]:
                stream.write(f"{value:.6f}\n")

            for value in v_values[time_index]:
                stream.write(f"{value:.6f}\n")

    return times, constant_time_step_hours(times)


def copy_boundary_files() -> None:
    shutil.copy2(
        BOUNDARY_EAST_FILE,
        OUTPUT_DIR / BOUNDARY_EAST_FILE.name,
    )
    shutil.copy2(
        BOUNDARY_SOUTH_FILE,
        OUTPUT_DIR / BOUNDARY_SOUTH_FILE.name,
    )


def format_segment_command(
    node_indices: np.ndarray,
    filename: str,
    nodes_per_line: int = 10,
) -> str:
    node_ids = [
        str(int(node) + 1)
        for node in node_indices
    ]

    chunks = [
        node_ids[index:index + nodes_per_line]
        for index in range(0, len(node_ids), nodes_per_line)
    ]

    lines: list[str] = []

    for chunk_index, chunk in enumerate(chunks):
        prefix = (
            "BOUNDSPEC SEGMENT IJ "
            if chunk_index == 0
            else "    "
        )
        lines.append(
            prefix + " ".join(chunk) + " &"
        )

    lines.append(
        f"    CONSTANT FILE '{filename}' 1"
    )

    return "\n".join(lines)


def generate_input(
    times: np.ndarray,
    wind_step_hours: float,
    compute_step_minutes: int,
    output_step_hours: float,
    segment_nodes: dict[int, np.ndarray],
) -> None:
    start = format_swan_time(times[0])
    end = format_swan_time(times[-1])

    boundary_commands: list[str] = []

    for marker in sorted(segment_nodes):
        filename = (
            "boundary_east.txt"
            if EAST_MARKER_BASE <= marker < SOUTH_MARKER_BASE
            else "boundary_south.txt"
        )

        boundary_commands.append(
            format_segment_command(
                segment_nodes[marker],
                filename,
            )
        )

    boundary_block = "\n\n".join(
        boundary_commands
    )

    content = f"""PROJECT 'UNSTRUCTURED' '01'

SET LEVEL 0.0 MAXERR=99 NAUTICAL

COORDINATES SPHERICAL CCM

MODE NONSTATIONARY

CGRID UNSTRUCTURED CIRCLE {DIRECTION_BINS} {MIN_FREQUENCY_HZ} {MAX_FREQUENCY_HZ}

READGRID UNSTRUCTURED TRIANGLE '{MESH_BASENAME}'

INPGRID BOTTOM UNSTRUCTURED EXCEPTION -999.0
READINP BOTTOM 1.0 'bottom_unstructured.txt' 1 0 FREE

INPGRID WIND UNSTRUCTURED NONSTATIONARY {start} {wind_step_hours:g} HR {end}
READINP WIND 1.0 'wind_unstructured.txt' 1 0 0 FREE

BOUND SHAPESPEC JONSWAP 3.3 MEAN DSPR DEGREES

{boundary_block}

NUMERIC ACCUR 0.02 0.02 0.02 99.5 NONSTAT 5

PROP BSBT

BLOCK 'COMPGRID' NOHEAD 'output_unstructured.mat' LAY 3 HSIGN TM01 TPS DIR WIND OUTPUT {start} {output_step_hours:g} HR

COMPUTE NONSTATIONARY {start} {compute_step_minutes} MIN {end}

STOP
"""

    INPUT_FILE.write_text(
        content,
        encoding="ascii",
        newline="\n",
    )


def write_metadata(
    points: np.ndarray,
    triangles: np.ndarray,
    boundary_edges: np.ndarray,
    segment_nodes: dict[int, np.ndarray],
    subsample: int,
) -> None:
    metadata = {
        "mesh_method": (
            "fully-wet regular cells split into conforming triangles"
        ),
        "node_count": int(points.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "boundary_edge_count": int(boundary_edges.shape[0]),
        "subsample": int(subsample),
        "segment_node_ids": {
            str(marker): [
                int(node) + 1
                for node in nodes
            ]
            for marker, nodes in segment_nodes.items()
        },
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def plot_preview(
    points: np.ndarray,
    triangles: np.ndarray,
    depth: np.ndarray,
    segment_nodes: dict[int, np.ndarray],
) -> None:
    triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    image = axis.tripcolor(
        triangulation,
        depth,
        shading="flat",
    )
    axis.triplot(
        triangulation,
        linewidth=0.15,
        alpha=0.35,
    )

    for marker, nodes in sorted(segment_nodes.items()):
        side = (
            "east"
            if marker < SOUTH_MARKER_BASE
            else "south"
        )

        axis.plot(
            points[nodes, 0],
            points[nodes, 1],
            linewidth=2.3,
            label=f"{side} segment {marker}",
        )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        pad=0.08,
    )
    colorbar.set_label("Water depth (m)")

    axis.set_title(
        "Conforming SWAN unstructured mesh and open boundaries"
    )
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")

    figure.savefig(
        PREVIEW_FILE,
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)


def run_swan(
    docker_image: str,
    executable: str,
) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{OUTPUT_DIR.resolve()}:/work",
        "-w",
        "/work",
        docker_image,
        executable,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )

    if result.stdout:
        print("\n--- STDOUT ---")
        print(result.stdout)

    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"SWAN finished with code {result.returncode}."
        )


def main() -> int:
    args = parse_args()

    try:
        if args.subsample < 1:
            raise ValueError(
                "--subsample must be at least 1."
            )

        require_files(
            [
                GRID_FILE,
                DEPTH_FILE,
                WIND_FILE,
                BOUNDARY_EAST_FILE,
                BOUNDARY_SOUTH_FILE,
            ]
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        longitude, latitude, regular_depth = (
            load_regular_domain()
        )

        points, node_depth, triangles = (
            build_conforming_mesh(
                longitude,
                latitude,
                regular_depth,
                args.subsample,
            )
        )

        boundary_edges = get_boundary_edges(
            triangles
        )

        edge_markers, segment_nodes = (
            classify_boundary_edges(
                points,
                boundary_edges,
                longitude,
                latitude,
            )
        )

        node_markers = node_markers_from_edges(
            points.shape[0],
            boundary_edges,
            edge_markers,
        )

        write_triangle_files(
            points,
            triangles,
            boundary_edges,
            edge_markers,
            node_markers,
        )
        write_bottom_file(node_depth)

        times, wind_step_hours = (
            generate_unstructured_wind(points)
        )

        copy_boundary_files()

        generate_input(
            times,
            wind_step_hours,
            args.compute_step_minutes,
            args.output_step_hours,
            segment_nodes,
        )

        write_metadata(
            points,
            triangles,
            boundary_edges,
            segment_nodes,
            args.subsample,
        )

        plot_preview(
            points,
            triangles,
            node_depth,
            segment_nodes,
        )

        print("\nUnstructured case generated.")
        print(f"Nodes:          {points.shape[0]}")
        print(f"Triangles:      {triangles.shape[0]}")
        print(f"Boundary edges: {boundary_edges.shape[0]}")
        print(
            "Segments:       "
            + ", ".join(
                f"{marker} ({nodes.size} nodes)"
                for marker, nodes in sorted(
                    segment_nodes.items()
                )
            )
        )
        print(f"Output:         {OUTPUT_DIR}")

        if args.run:
            run_swan(
                args.docker_image,
                args.swan_executable,
            )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())