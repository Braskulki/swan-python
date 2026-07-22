"""
Research-grade SWAN unstructured mesh and input generator.

The script creates a coastline-following unstructured mesh instead of
triangulating the pixels of the regular raster. It is intended as a solid
starting point for research workflows and publication-quality domain figures.

Workflow
--------
1. Read grid.json and depth.bot.
2. Extract the 0 m (or selected) bathymetric contour as vector lines.
3. Polygonize the contour together with the rectangular model limits.
4. Select the water polygon connected to the open ocean.
5. Simplify and optionally smooth the polygon while preserving topology.
6. Generate a quality triangular mesh with Gmsh.
7. Refine the mesh near the coastline using Distance + Threshold fields.
8. Export Triangle-compatible .node, .ele, .edge and .poly files.
9. Interpolate bathymetry and ERA5 wind to the mesh nodes.
10. Create BOUNDSPEC SEGMENT commands from ordered unstructured vertices.
11. Write a complete SWAN INPUT file.
12. Generate QA figures and metadata.

Inputs
------
data/processed/grid.json
data/processed/depth.bot
data/raw/wind.nc
data/processed/boundary_east.txt
data/processed/boundary_south.txt

Outputs
-------
data/unstructured_research/
├── INPUT
├── mesh.node
├── mesh.ele
├── mesh.edge
├── mesh.poly
├── mesh.msh
├── bottom_unstructured.txt
├── wind_unstructured.txt
├── boundary_east.txt
├── boundary_south.txt
├── domain_polygon.geojson
├── mesh_metadata.json
├── coastline_and_domain.png
├── mesh_preview.png
└── quality_report.csv

Dependencies
------------
pip install numpy scipy xarray netcdf4 matplotlib pandas shapely contourpy gmsh

Examples
--------
py generate_research_unstructured_pipeline.py
py generate_research_unstructured_pipeline.py --coast-level 0
py generate_research_unstructured_pipeline.py --coast-size 0.015 --offshore-size 0.08
py generate_research_unstructured_pipeline.py --run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import contourpy
import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
    mapping,
)
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "unstructured_research"

GRID_FILE = PROCESSED_DIR / "grid.json"
DEPTH_FILE = PROCESSED_DIR / "depth.bot"
WIND_FILE = RAW_DIR / "wind.nc"

BOUNDARY_EAST_FILE = PROCESSED_DIR / "boundary_east.txt"
BOUNDARY_SOUTH_FILE = PROCESSED_DIR / "boundary_south.txt"

NODE_FILE = OUTPUT_DIR / "mesh.node"
ELE_FILE = OUTPUT_DIR / "mesh.ele"
EDGE_FILE = OUTPUT_DIR / "mesh.edge"
POLY_FILE = OUTPUT_DIR / "mesh.poly"
MSH_FILE = OUTPUT_DIR / "mesh.msh"

BOTTOM_FILE = OUTPUT_DIR / "bottom_unstructured.txt"
WIND_OUTPUT_FILE = OUTPUT_DIR / "wind_unstructured.txt"
INPUT_FILE = OUTPUT_DIR / "INPUT"

DOMAIN_GEOJSON = OUTPUT_DIR / "domain_polygon.geojson"
METADATA_FILE = OUTPUT_DIR / "mesh_metadata.json"
QUALITY_REPORT = OUTPUT_DIR / "quality_report.csv"
DOMAIN_PREVIEW = OUTPUT_DIR / "coastline_and_domain.png"
MESH_PREVIEW = OUTPUT_DIR / "mesh_preview.png"

MESH_BASENAME = "mesh"

COAST_MARKER = 1
EAST_MARKER_BASE = 20
SOUTH_MARKER_BASE = 30

DIRECTION_BINS = 36
MIN_FREQUENCY_HZ = 0.04
MAX_FREQUENCY_HZ = 1.0

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
            "Generate a coastline-following, quality-controlled "
            "unstructured SWAN case."
        )
    )

    parser.add_argument(
        "--coast-level",
        type=float,
        default=0.0,
        help=(
            "Bathymetric contour used as coastline/domain boundary in metres. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--minimum-water-depth",
        type=float,
        default=0.5,
        help=(
            "Minimum positive depth assigned to mesh nodes that lie on or "
            "very close to the selected coastal contour. Default: 0.5 m."
        ),
    )
    parser.add_argument(
        "--depth-repair-tolerance",
        type=float,
        default=2.0,
        help=(
            "Maximum depth mismatch, in metres, that may be repaired by "
            "applying --minimum-water-depth. Nodes farther landward than "
            "coast_level - tolerance cause an error. Default: 2 m."
        ),
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.006,
        help=(
            "Topology-preserving coastline simplification tolerance in degrees. "
            "Default: 0.006."
        ),
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help=(
            "Optional polygon smoothing distance in degrees using "
            "buffer(+d).buffer(-d). Default: 0."
        ),
    )
    parser.add_argument(
        "--topology-cleanup",
        type=float,
        default=0.0005,
        help=(
            "Initial erosion/dilation distance in degrees used to remove "
            "point contacts and very narrow wet-domain necks that SWAN "
            "cannot traverse. The generator retries with larger values when "
            "necessary. Default: 0.0005."
        ),
    )
    parser.add_argument(
        "--topology-cleanup-attempts",
        type=int,
        default=5,
        help=(
            "Maximum number of automatic remeshing attempts used to remove "
            "branched boundary vertices. Default: 5."
        ),
    )
    parser.add_argument(
        "--minimum-hole-area",
        type=float,
        default=1.0e-6,
        help=(
            "Polygon holes smaller than this area in square degrees are "
            "removed before meshing. Default: 1e-6."
        ),
    )
    parser.add_argument(
        "--coast-size",
        type=float,
        default=0.018,
        help="Target Gmsh element size near the coastline in degrees.",
    )
    parser.add_argument(
        "--offshore-size",
        type=float,
        default=0.075,
        help="Target Gmsh element size offshore in degrees.",
    )
    parser.add_argument(
        "--refine-distance-min",
        type=float,
        default=0.05,
        help=(
            "Distance from coastline over which coast-size is retained, "
            "in degrees."
        ),
    )
    parser.add_argument(
        "--refine-distance-max",
        type=float,
        default=0.45,
        help=(
            "Distance from coastline at which offshore-size is reached, "
            "in degrees."
        ),
    )
    parser.add_argument(
        "--minimum-angle",
        type=float,
        default=28.0,
        help=(
            "Target minimum triangle angle used for QA. Gmsh does not "
            "guarantee this exact value. Default: 28 degrees."
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
        help="Run SWAN through Docker after generation.",
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


def require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(f"- {path}" for path in missing)
        )


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

    expected_shape = (latitude.size, longitude.size)

    if depth.shape != expected_shape:
        raise ValueError(
            f"depth.bot shape is {depth.shape}; "
            f"expected {expected_shape}."
        )

    if not np.all(np.diff(longitude) > 0):
        raise ValueError("Longitude must be strictly increasing.")

    if not np.all(np.diff(latitude) > 0):
        raise ValueError("Latitude must be strictly increasing.")

    return longitude, latitude, depth


def depth_interpolator(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    fill_value: float = -999.0,
) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (latitude, longitude),
        depth,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )


def contour_lines(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    level: float,
) -> list[LineString]:
    generator = contourpy.contour_generator(
        x=longitude,
        y=latitude,
        z=depth,
        name="serial",
        line_type=contourpy.LineType.Separate,
    )

    raw_lines = generator.lines(level)
    lines: list[LineString] = []

    for coordinates in raw_lines:
        if coordinates.shape[0] < 2:
            continue

        line = LineString(coordinates)

        if line.length > 0:
            lines.append(line)

    if not lines:
        raise ValueError(
            f"No bathymetric contour was found at level {level} m."
        )

    return lines


def polygon_parts(geometry) -> list[Polygon]:
    if geometry.is_empty:
        return []

    geometry = make_valid(geometry)

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    return [
        item
        for item in getattr(geometry, "geoms", [])
        if isinstance(item, Polygon)
    ]


def build_water_polygon(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    level: float,
) -> tuple[Polygon, list[LineString]]:
    """
    Polygonizes the isobath and bounding-box edges, then retains cells whose
    representative point has depth greater than the selected level.
    """
    bbox = box(
        float(longitude.min()),
        float(latitude.min()),
        float(longitude.max()),
        float(latitude.max()),
    )

    coast_lines = contour_lines(
        longitude,
        latitude,
        depth,
        level,
    )

    linework = unary_union(
        coast_lines
        + [LineString(bbox.exterior.coords)]
    )

    candidates = list(polygonize(linework))

    if not candidates:
        raise RuntimeError(
            "The coastline and bounding-box linework did not polygonize."
        )

    interpolate = depth_interpolator(
        longitude,
        latitude,
        depth,
        fill_value=-999.0,
    )

    wet_polygons = []

    for candidate in candidates:
        sample = candidate.representative_point()
        sample_depth = float(
            interpolate([[sample.y, sample.x]])[0]
        )

        if sample_depth > level:
            wet_polygons.append(candidate)

    if not wet_polygons:
        raise RuntimeError(
            "No water polygon was identified after polygonization."
        )

    merged = make_valid(
        unary_union(wet_polygons)
    )

    parts = polygon_parts(merged)

    if not parts:
        raise RuntimeError(
            "The resulting water geometry contains no polygons."
        )

    # Prefer the polygon connected to the southeast/open-ocean corner.
    southeast_probe = Point(
        float(longitude.max()) - 1e-8,
        float(latitude.min()) + 1e-8,
    )

    connected = [
        polygon
        for polygon in parts
        if polygon.buffer(1e-7).contains(southeast_probe)
    ]

    if connected:
        domain = max(connected, key=lambda polygon: polygon.area)
    else:
        domain = max(parts, key=lambda polygon: polygon.area)

    return make_valid(domain), coast_lines


def process_domain_polygon(
    polygon: Polygon,
    simplify: float,
    smooth: float,
    topology_cleanup: float = 0.0,
    minimum_hole_area: float = 0.0,
) -> Polygon:
    """
    Prepare a SWAN-safe wet-domain polygon.

    Besides simplification, this applies a small erosion followed by dilation.
    That operation removes point contacts and extremely narrow wet-domain
    necks, which otherwise create boundary vertices with degree four and make
    SwanBpntlist fail.

    Small polygon holes are removed to avoid isolated one-to-three-triangle
    components after meshing. The final geometry is always intersected with
    the original wet polygon, so processing cannot expand the domain over land.
    """
    original = make_valid(polygon)
    processed = original

    if simplify > 0:
        processed = processed.simplify(
            simplify,
            preserve_topology=True,
        )

    if smooth > 0:
        processed = (
            processed
            .buffer(smooth, join_style="round")
            .buffer(-smooth, join_style="round")
        )

    if topology_cleanup > 0:
        processed = (
            processed
            .buffer(-topology_cleanup, join_style="round")
            .buffer(topology_cleanup, join_style="round")
        )

    processed = make_valid(
        processed.intersection(original)
    )

    parts = polygon_parts(processed)

    if not parts:
        raise ValueError(
            "Polygon processing removed the model domain."
        )

    result = max(parts, key=lambda item: item.area)

    retained_holes = []

    for interior in result.interiors:
        hole = Polygon(interior)

        if hole.area >= minimum_hole_area:
            retained_holes.append(list(interior.coords))

    result = make_valid(
        Polygon(
            list(result.exterior.coords),
            retained_holes,
        )
    )

    parts = polygon_parts(result)

    if not parts:
        raise ValueError(
            "Hole cleanup removed the model domain."
        )

    result = max(parts, key=lambda item: item.area)

    if not result.exterior.is_ccw:
        result = Polygon(
            list(result.exterior.coords)[::-1],
            [
                list(interior.coords)[::-1]
                for interior in result.interiors
            ],
        )

    return result

def classify_ring_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
    tolerance_x: float,
    tolerance_y: float,
) -> str:
    min_lon, min_lat, max_lon, max_lat = bounds

    if (
        abs(start[0] - max_lon) <= tolerance_x
        and abs(end[0] - max_lon) <= tolerance_x
    ):
        return "east"

    if (
        abs(start[1] - min_lat) <= tolerance_y
        and abs(end[1] - min_lat) <= tolerance_y
    ):
        return "south"

    return "coast"


def create_gmsh_geometry(
    domain: Polygon,
    longitude: np.ndarray,
    latitude: np.ndarray,
    coast_size: float,
    offshore_size: float,
    distance_min: float,
    distance_max: float,
) -> tuple[
    int,
    dict[str, list[int]],
]:
    """
    Creates the domain geometry in Gmsh.

    Returns:
        surface_tag
        curve_tags_by_class
    """
    bounds = domain.bounds
    tolerance_x = abs(float(np.median(np.diff(longitude)))) * 0.55
    tolerance_y = abs(float(np.median(np.diff(latitude)))) * 0.55

    curve_tags: dict[str, list[int]] = {
        "coast": [],
        "east": [],
        "south": [],
    }

    def add_ring(
        coordinates: list[tuple[float, float]],
    ) -> int:
        coordinates = coordinates[:-1]
        point_tags = [
            gmsh.model.geo.addPoint(
                float(x),
                float(y),
                0.0,
                offshore_size,
            )
            for x, y in coordinates
        ]

        line_tags = []

        for index, point_tag in enumerate(point_tags):
            next_index = (index + 1) % len(point_tags)
            next_tag = point_tags[next_index]

            line_tag = gmsh.model.geo.addLine(
                point_tag,
                next_tag,
            )
            line_tags.append(line_tag)

            segment_class = classify_ring_segment(
                coordinates[index],
                coordinates[next_index],
                bounds,
                tolerance_x,
                tolerance_y,
            )
            curve_tags[segment_class].append(line_tag)

        return gmsh.model.geo.addCurveLoop(line_tags)

    exterior_loop = add_ring(
        list(domain.exterior.coords)
    )

    hole_loops = [
        add_ring(list(interior.coords))
        for interior in domain.interiors
    ]

    surface = gmsh.model.geo.addPlaneSurface(
        [exterior_loop, *hole_loops]
    )

    gmsh.model.geo.synchronize()

    if curve_tags["coast"]:
        distance_field = gmsh.model.mesh.field.add(
            "Distance"
        )
        gmsh.model.mesh.field.setNumbers(
            distance_field,
            "CurvesList",
            curve_tags["coast"],
        )
        gmsh.model.mesh.field.setNumber(
            distance_field,
            "Sampling",
            250,
        )

        threshold_field = gmsh.model.mesh.field.add(
            "Threshold"
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "InField",
            distance_field,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMin",
            coast_size,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMax",
            offshore_size,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "DistMin",
            distance_min,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "DistMax",
            distance_max,
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(
            threshold_field
        )

    gmsh.option.setNumber(
        "Mesh.MeshSizeFromPoints",
        0,
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeFromCurvature",
        12,
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeExtendFromBoundary",
        0,
    )
    gmsh.option.setNumber(
        "Mesh.Algorithm",
        6,
    )
    gmsh.option.setNumber(
        "Mesh.Optimize",
        1,
    )
    gmsh.option.setNumber(
        "Mesh.OptimizeNetgen",
        1,
    )

    return surface, curve_tags


def gmsh_mesh(
    domain: Polygon,
    longitude: np.ndarray,
    latitude: np.ndarray,
    args: argparse.Namespace,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, list[tuple[int, int]]],
]:
    gmsh.initialize()
    gmsh.option.setNumber(
        "General.Terminal",
        1,
    )
    gmsh.model.add(
        "swan_research_domain"
    )

    try:
        _, curve_tags = create_gmsh_geometry(
            domain=domain,
            longitude=longitude,
            latitude=latitude,
            coast_size=args.coast_size,
            offshore_size=args.offshore_size,
            distance_min=args.refine_distance_min,
            distance_max=args.refine_distance_max,
        )

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize(
            "Netgen"
        )
        gmsh.write(str(MSH_FILE))

        node_tags, coordinates, _ = (
            gmsh.model.mesh.getNodes()
        )

        coordinates = np.asarray(
            coordinates,
            dtype=float,
        ).reshape(-1, 3)

        node_tags = np.asarray(
            node_tags,
            dtype=np.int64,
        )
        tag_to_index = {
            int(tag): index
            for index, tag in enumerate(node_tags)
        }

        points = coordinates[:, :2]

        element_types, _, element_node_tags = (
            gmsh.model.mesh.getElements(dim=2)
        )

        triangles = []

        for element_type, node_data in zip(
            element_types,
            element_node_tags,
        ):
            properties = gmsh.model.mesh.getElementProperties(
                element_type
            )
            element_name = properties[0]
            nodes_per_element = properties[3]

            if "Triangle" not in element_name:
                continue

            raw = np.asarray(
                node_data,
                dtype=np.int64,
            ).reshape(-1, nodes_per_element)

            raw = raw[:, :3]

            for element in raw:
                triangle = np.asarray(
                    [
                        tag_to_index[int(tag)]
                        for tag in element
                    ],
                    dtype=int,
                )

                if triangle_signed_area(
                    points,
                    triangle,
                ) < 0:
                    triangle[[1, 2]] = triangle[[2, 1]]

                triangles.append(triangle)

        if not triangles:
            raise RuntimeError(
                "Gmsh generated no triangular elements."
            )

        boundary_edges: dict[str, list[tuple[int, int]]] = {
            "coast": [],
            "east": [],
            "south": [],
        }

        for boundary_class, tags in curve_tags.items():
            for curve_tag in tags:
                types, _, curve_nodes = (
                    gmsh.model.mesh.getElements(
                        dim=1,
                        tag=curve_tag,
                    )
                )

                for element_type, node_data in zip(
                    types,
                    curve_nodes,
                ):
                    properties = gmsh.model.mesh.getElementProperties(
                        element_type
                    )
                    element_name = properties[0]
                    nodes_per_element = properties[3]

                    if "Line" not in element_name:
                        continue

                    raw = np.asarray(
                        node_data,
                        dtype=np.int64,
                    ).reshape(-1, nodes_per_element)

                    for line in raw:
                        node_a = tag_to_index[
                            int(line[0])
                        ]
                        node_b = tag_to_index[
                            int(line[1])
                        ]

                        boundary_edges[boundary_class].append(
                            (node_a, node_b)
                        )

        return (
            points,
            np.asarray(triangles, dtype=int),
            boundary_edges,
        )

    finally:
        gmsh.finalize()


def triangle_signed_area(
    points: np.ndarray,
    triangle: np.ndarray,
) -> float:
    a, b, c = points[triangle]

    return 0.5 * (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def connected_edge_components(
    edges: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    if not edges:
        return []

    adjacency: dict[int, list[int]] = defaultdict(list)

    for edge_index, (node_a, node_b) in enumerate(edges):
        adjacency[node_a].append(edge_index)
        adjacency[node_b].append(edge_index)

    remaining = set(range(len(edges)))
    components = []

    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component = [edges[seed]]

        while queue:
            edge_index = queue.popleft()
            node_a, node_b = edges[edge_index]

            for node in (node_a, node_b):
                for neighbour in adjacency[node]:
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        queue.append(neighbour)
                        component.append(
                            edges[neighbour]
                        )

        components.append(component)

    return components


def ordered_edge_chain(
    edges: list[tuple[int, int]],
) -> np.ndarray:
    adjacency: dict[int, list[int]] = defaultdict(list)

    for node_a, node_b in edges:
        adjacency[node_a].append(node_b)
        adjacency[node_b].append(node_a)

    endpoints = [
        node
        for node, neighbours in adjacency.items()
        if len(neighbours) == 1
    ]

    if len(endpoints) == 2:
        start = endpoints[0]
    elif not endpoints:
        start = min(adjacency)
    else:
        raise ValueError(
            "Boundary component is branched and cannot be ordered."
        )

    ordered = [start]
    previous = None
    current = start

    while True:
        neighbours = [
            node
            for node in adjacency[current]
            if node != previous
        ]

        if not neighbours:
            break

        next_node = neighbours[0]

        if next_node == start:
            break

        ordered.append(next_node)
        previous, current = current, next_node

        if len(ordered) > len(adjacency) + 1:
            raise RuntimeError(
                "Boundary ordering exceeded component size."
            )

    return np.asarray(ordered, dtype=int)


def orient_open_segment(
    points: np.ndarray,
    nodes: np.ndarray,
    side: str,
) -> np.ndarray:
    """
    Orient an outer-domain boundary segment counterclockwise.

    For a counterclockwise exterior ring, the wet domain remains on the left
    side of the traversal:

    - east boundary: north -> south
    - south boundary: east -> west
    """
    if side == "east":
        if points[nodes[0], 1] < points[nodes[-1], 1]:
            nodes = nodes[::-1].copy()
    elif side == "south":
        if points[nodes[0], 0] < points[nodes[-1], 0]:
            nodes = nodes[::-1].copy()
    else:
        raise ValueError(
            f"Unsupported open-boundary side: {side}"
        )

    return nodes


def validate_ordered_boundary_segment(
    points: np.ndarray,
    nodes: np.ndarray,
    component_edges: list[tuple[int, int]],
    side: str,
) -> None:
    """
    Ensure every consecutive pair is an actual exterior edge and orientation
    is counterclockwise for the requested side.
    """
    if nodes.size < 2:
        raise ValueError(
            f"Boundary segment '{side}' has fewer than two vertices."
        )

    edge_set = {
        tuple(sorted((int(a), int(b))))
        for a, b in component_edges
    }

    missing_pairs = []

    for node_a, node_b in zip(nodes[:-1], nodes[1:]):
        edge = tuple(sorted((int(node_a), int(node_b))))

        if edge not in edge_set:
            missing_pairs.append(
                (int(node_a) + 1, int(node_b) + 1)
            )

    if missing_pairs:
        preview = ", ".join(
            f"{a}->{b}"
            for a, b in missing_pairs[:10]
        )

        raise ValueError(
            f"Boundary segment '{side}' is not continuous. "
            f"Missing exterior edges: {preview}"
        )

    first = points[int(nodes[0])]
    last = points[int(nodes[-1])]

    if side == "east" and first[1] < last[1]:
        raise ValueError(
            "East boundary must be ordered north-to-south for SWAN."
        )

    if side == "south" and first[0] < last[0]:
        raise ValueError(
            "South boundary must be ordered east-to-west for SWAN."
        )


def build_boundary_segments(
    points: np.ndarray,
    boundary_edges: dict[str, list[tuple[int, int]]],
) -> tuple[
    list[tuple[int, int, int]],
    dict[int, np.ndarray],
]:
    """
    Returns all boundary edges with markers and ordered open-boundary segments.
    """
    marked_edges: list[tuple[int, int, int]] = []
    open_segments: dict[int, np.ndarray] = {}

    for node_a, node_b in boundary_edges["coast"]:
        marked_edges.append(
            (node_a, node_b, COAST_MARKER)
        )

    for side, marker_base in (
        ("east", EAST_MARKER_BASE),
        ("south", SOUTH_MARKER_BASE),
    ):
        components = connected_edge_components(
            boundary_edges[side]
        )

        for component_index, component in enumerate(
            components
        ):
            marker = marker_base + component_index
            nodes = orient_open_segment(
                points,
                ordered_edge_chain(component),
                side,
            )

            validate_ordered_boundary_segment(
                points=points,
                nodes=nodes,
                component_edges=component,
                side=side,
            )

            open_segments[marker] = nodes

            for node_a, node_b in component:
                marked_edges.append(
                    (node_a, node_b, marker)
                )

    if not any(
        EAST_MARKER_BASE <= marker < SOUTH_MARKER_BASE
        for marker in open_segments
    ):
        raise ValueError(
            "No east open boundary was generated."
        )

    if not any(
        marker >= SOUTH_MARKER_BASE
        for marker in open_segments
    ):
        raise ValueError(
            "No south open boundary was generated."
        )

    return marked_edges, open_segments




def keep_largest_triangle_component(
    points: np.ndarray,
    triangles: np.ndarray,
    node_values: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    dict[str, object],
]:
    """
    Keep only the largest edge-connected triangle component and compact nodes.
    """
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)

    for triangle_index, (node_a, node_b, node_c) in enumerate(triangles):
        for edge in (
            tuple(sorted((int(node_a), int(node_b)))),
            tuple(sorted((int(node_b), int(node_c)))),
            tuple(sorted((int(node_c), int(node_a)))),
        ):
            edge_to_triangles[edge].append(triangle_index)

    adjacency: list[set[int]] = [
        set() for _ in range(triangles.shape[0])
    ]

    for owners in edge_to_triangles.values():
        if len(owners) == 2:
            first, second = owners
            adjacency[first].add(second)
            adjacency[second].add(first)

    unseen = set(range(triangles.shape[0]))
    components: list[list[int]] = []

    while unseen:
        start = unseen.pop()
        queue = deque([start])
        component = [start]

        while queue:
            current = queue.popleft()

            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)

        components.append(component)

    components.sort(key=len, reverse=True)
    keep_indices = np.asarray(components[0], dtype=int)
    kept_triangles = triangles[keep_indices]

    used_nodes = np.unique(kept_triangles.reshape(-1))
    mapping = np.full(points.shape[0], -1, dtype=int)
    mapping[used_nodes] = np.arange(used_nodes.size)

    compact_points = points[used_nodes]
    compact_triangles = mapping[kept_triangles]
    compact_values = (
        np.asarray(node_values)[used_nodes]
        if node_values is not None
        else None
    )

    qa = {
        "component_count_before_cleanup": len(components),
        "component_triangle_counts": [len(item) for item in components],
        "removed_disconnected_triangles": int(
            triangles.shape[0] - kept_triangles.shape[0]
        ),
        "removed_unused_nodes": int(
            points.shape[0] - compact_points.shape[0]
        ),
    }

    return compact_points, compact_triangles, compact_values, qa


def mesh_boundary_topology(
    triangles: np.ndarray,
) -> dict[str, object]:
    """
    Inspect the exterior-edge graph used by SWAN.
    """
    edges = exterior_edges_from_triangles(triangles)
    adjacency: dict[int, set[int]] = defaultdict(set)

    for node_a, node_b in edges:
        adjacency[int(node_a)].add(int(node_b))
        adjacency[int(node_b)].add(int(node_a))

    invalid_degrees = {
        node: len(neighbours)
        for node, neighbours in adjacency.items()
        if len(neighbours) != 2
    }

    unseen = set(adjacency)
    component_sizes: list[int] = []

    while unseen:
        start = unseen.pop()
        queue = deque([start])
        size = 1

        while queue:
            current = queue.popleft()

            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
                    size += 1

        component_sizes.append(size)

    component_sizes.sort(reverse=True)

    return {
        "boundary_edge_count": int(edges.shape[0]),
        "boundary_vertex_count": len(adjacency),
        "boundary_component_count": len(component_sizes),
        "boundary_component_sizes": component_sizes,
        "invalid_boundary_degrees": invalid_degrees,
        "is_swan_traversable": not invalid_degrees,
    }


def require_swan_mesh_topology(
    triangles: np.ndarray,
) -> dict[str, object]:
    qa = mesh_boundary_topology(triangles)
    invalid = qa["invalid_boundary_degrees"]

    if invalid:
        examples = list(sorted(invalid.items()))[:20]

        raise ValueError(
            "Mesh exterior boundary is branched and cannot be traversed by "
            "SWAN. Every exterior vertex must have degree 2. "
            f"Problem vertices (zero-based): {examples}"
        )

    return qa


def exterior_edges_from_triangles(
    triangles: np.ndarray,
) -> np.ndarray:
    """Returns edges used by exactly one triangle."""
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
            "No exterior edges remained after wet-domain clipping."
        )

    return np.asarray(edges, dtype=int)


def classify_exterior_edges(
    points: np.ndarray,
    triangles: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    """
    Rebuilds the real mesh boundary after landward triangles are removed.
    """
    edges = exterior_edges_from_triangles(triangles)

    dx = abs(float(np.median(np.diff(longitude))))
    dy = abs(float(np.median(np.diff(latitude))))
    lon_max = float(longitude.max())
    lat_min = float(latitude.min())

    result: dict[str, list[tuple[int, int]]] = {
        "coast": [],
        "east": [],
        "south": [],
    }

    for node_a, node_b in edges:
        point_a = points[int(node_a)]
        point_b = points[int(node_b)]

        if (
            abs(point_a[0] - lon_max) <= dx * 0.75
            and abs(point_b[0] - lon_max) <= dx * 0.75
        ):
            boundary_class = "east"
        elif (
            abs(point_a[1] - lat_min) <= dy * 0.75
            and abs(point_b[1] - lat_min) <= dy * 0.75
        ):
            boundary_class = "south"
        else:
            boundary_class = "coast"

        result[boundary_class].append(
            (int(node_a), int(node_b))
        )

    return result


def clip_mesh_to_wet_domain(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    coast_level: float,
    repair_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    """
    Removes triangles that entered clearly terrestrial regions after polygon
    simplification or straight-line meshing.

    A triangle is retained only when all its vertices and its centroid are no
    farther landward than coast_level - repair_tolerance. The mesh is then
    compacted, so SWAN receives no unused nodes.
    """
    interpolate = depth_interpolator(
        longitude,
        latitude,
        depth,
        fill_value=-999.0,
    )

    node_depth = np.asarray(
        interpolate(
            np.column_stack((points[:, 1], points[:, 0]))
        ),
        dtype=float,
    )

    centroids = points[triangles].mean(axis=1)
    centroid_depth = np.asarray(
        interpolate(
            np.column_stack((centroids[:, 1], centroids[:, 0]))
        ),
        dtype=float,
    )

    minimum_allowed = coast_level - repair_tolerance

    valid_nodes = (
        np.isfinite(node_depth)
        & (node_depth >= minimum_allowed)
    )
    valid_centroids = (
        np.isfinite(centroid_depth)
        & (centroid_depth >= minimum_allowed)
    )

    keep = (
        np.all(valid_nodes[triangles], axis=1)
        & valid_centroids
    )

    kept_triangles = triangles[keep]

    if kept_triangles.size == 0:
        raise RuntimeError(
            "Wet-domain clipping removed every triangle. "
            "Decrease --coast-level or --depth-repair-tolerance."
        )

    used_nodes = np.unique(kept_triangles.reshape(-1))
    mapping = np.full(points.shape[0], -1, dtype=int)
    mapping[used_nodes] = np.arange(used_nodes.size)

    compact_points = points[used_nodes]
    compact_depth = node_depth[used_nodes]
    compact_triangles = mapping[kept_triangles]

    qa = {
        "original_node_count": int(points.shape[0]),
        "original_triangle_count": int(triangles.shape[0]),
        "removed_triangle_count": int(np.count_nonzero(~keep)),
        "removed_triangle_fraction": float(np.count_nonzero(~keep) / keep.size),
        "final_node_count": int(compact_points.shape[0]),
        "final_triangle_count": int(compact_triangles.shape[0]),
        "minimum_allowed_depth_m": float(minimum_allowed),
    }

    return (
        compact_points,
        compact_triangles,
        compact_depth,
        qa,
    )


def interpolate_depth_to_nodes(
    longitude: np.ndarray,
    latitude: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
    coast_level: float,
    minimum_water_depth: float,
    repair_tolerance: float,
    precomputed_values: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """
    Interpolates depth to unstructured nodes and repairs only near-coast
    numerical values.

    Nodes on a 0 m contour naturally interpolate to zero, while SWAN requires
    positive water depth. Values no farther landward than
    coast_level - repair_tolerance are assigned minimum_water_depth. More
    negative values are treated as a geometry error and stop the pipeline.
    """
    if minimum_water_depth <= 0:
        raise ValueError(
            "--minimum-water-depth must be positive."
        )

    if repair_tolerance < 0:
        raise ValueError(
            "--depth-repair-tolerance cannot be negative."
        )

    if precomputed_values is None:
        interpolate = depth_interpolator(
            longitude,
            latitude,
            depth,
            fill_value=-999.0,
        )

        values = np.asarray(
            interpolate(
                np.column_stack(
                    (points[:, 1], points[:, 0])
                )
            ),
            dtype=float,
        )
    else:
        values = np.asarray(
            precomputed_values,
            dtype=float,
        ).copy()

        if values.size != points.shape[0]:
            raise ValueError(
                "Precomputed depth count does not match mesh node count."
            )

    invalid = ~np.isfinite(values)
    severe_landward = (
        invalid
        | (values < coast_level - repair_tolerance)
    )

    if np.any(severe_landward):
        severe_values = values[severe_landward]
        finite_severe = severe_values[
            np.isfinite(severe_values)
        ]

        minimum_found = (
            float(np.min(finite_severe))
            if finite_severe.size
            else float("nan")
        )

        raise ValueError(
            f"{int(np.count_nonzero(severe_landward))} mesh nodes are too "
            "far landward for a safe depth repair. "
            f"Minimum interpolated value: {minimum_found:.3f} m. "
            "Reduce --simplify/--smooth, increase --coast-level, or inspect "
            "coastline_and_domain.png."
        )

    repair_mask = values < minimum_water_depth
    original_repair_values = values[repair_mask].copy()
    values[repair_mask] = minimum_water_depth

    qa = {
        "repaired_node_count": int(
            np.count_nonzero(repair_mask)
        ),
        "repaired_fraction": float(
            np.count_nonzero(repair_mask) / values.size
        ),
        "minimum_before_repair_m": (
            float(np.min(original_repair_values))
            if original_repair_values.size
            else float(np.min(values))
        ),
        "minimum_after_repair_m": float(
            np.min(values)
        ),
        "maximum_depth_m": float(
            np.max(values)
        ),
    }

    return values, qa

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
        f"None of {candidates} was found."
    )


def format_swan_time(value: np.datetime64) -> str:
    text = np.datetime_as_string(
        value,
        unit="s",
    )
    date, clock = text.split("T")

    return (
        date.replace("-", "")
        + "."
        + clock.replace(":", "")
    )


def constant_time_step_hours(
    times: np.ndarray,
) -> float:
    seconds = np.diff(
        times.astype("datetime64[s]").astype(np.int64)
    )

    if seconds.size == 0:
        raise ValueError(
            "At least two wind timestamps are required."
        )

    if np.any(seconds <= 0) or not np.all(
        seconds == seconds[0]
    ):
        raise ValueError(
            "Wind times must have a constant interval."
        )

    return float(seconds[0]) / 3600.0


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
            dtype=float,
        )
        v_values = np.asarray(
            v.values,
            dtype=float,
        )

    if (
        not np.isfinite(u_values).all()
        or not np.isfinite(v_values).all()
    ):
        raise ValueError(
            "Wind interpolation produced NaN values."
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


def node_markers(
    node_count: int,
    marked_edges: list[tuple[int, int, int]],
) -> np.ndarray:
    markers = np.zeros(
        node_count,
        dtype=int,
    )

    priority = {
        0: 0,
        COAST_MARKER: 1,
    }

    for node_a, node_b, marker in marked_edges:
        for node in (node_a, node_b):
            current = int(markers[node])

            current_priority = priority.get(
                current,
                2,
            )
            new_priority = priority.get(
                marker,
                2,
            )

            if new_priority >= current_priority:
                markers[node] = marker

    return markers


def write_triangle_files(
    points: np.ndarray,
    triangles: np.ndarray,
    marked_edges: list[tuple[int, int, int]],
) -> None:
    markers = node_markers(
        points.shape[0],
        marked_edges,
    )

    open_edge_markers = sorted(
        {
            int(marker)
            for _, _, marker in marked_edges
            if marker != COAST_MARKER
        }
    )

    missing_markers = [
        marker
        for marker in open_edge_markers
        if not np.any(markers == marker)
    ]

    if missing_markers:
        raise ValueError(
            "Open-boundary markers were not assigned to mesh.node: "
            + ", ".join(map(str, missing_markers))
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
            zip(points, markers),
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

        for triangle_id, triangle in enumerate(
            triangles,
            start=1,
        ):
            n1, n2, n3 = triangle + 1
            stream.write(
                f"{triangle_id} {n1} {n2} {n3}\n"
            )

    with EDGE_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        stream.write(
            f"{len(marked_edges)} 1\n"
        )

        for edge_id, (node_a, node_b, marker) in enumerate(
            marked_edges,
            start=1,
        ):
            stream.write(
                f"{edge_id} {node_a + 1} {node_b + 1} {marker}\n"
            )

    with POLY_FILE.open(
        "w",
        encoding="ascii",
        newline="\n",
    ) as stream:
        stream.write("0 2 0 1\n")
        stream.write(
            f"{len(marked_edges)} 1\n"
        )

        for segment_id, (node_a, node_b, marker) in enumerate(
            marked_edges,
            start=1,
        ):
            stream.write(
                f"{segment_id} {node_a + 1} {node_b + 1} {marker}\n"
            )

        stream.write("0\n")
        stream.write("0\n")




def format_unstructured_segment_command(
    nodes: np.ndarray,
    filename: str,
    maximum_line_length: int = 170,
) -> str:
    """
    Format BOUNDSPEC SEGMENT for an unstructured Triangle mesh.

    For unstructured meshes SWAN expects the one-based boundary vertex
    indices directly after SEGMENT. The IJ keyword must not be used because
    it is reserved for structured computational-grid indices.

    The vertices must be supplied counterclockwise. Lines are wrapped with
    SWAN's continuation marker (&) so they remain below the 180-character
    input-line limit.
    """
    if nodes.size < 2:
        raise ValueError(
            "An open-boundary segment must contain at least two vertices."
        )

    vertex_tokens = [
        str(int(node) + 1)
        for node in nodes
    ]

    suffix = f"CONSTANT FILE '{filename}' 1"
    lines: list[str] = []
    current = "BOUNDSPEC SEGMENT"

    for token in vertex_tokens:
        candidate = f"{current} {token}"

        if len(candidate) + 2 > maximum_line_length:
            lines.append(current + " &")
            current = "  " + token
        else:
            current = candidate

    candidate = f"{current} {suffix}"

    if len(candidate) <= maximum_line_length:
        lines.append(candidate)
    else:
        lines.append(current + " &")
        lines.append("  " + suffix)

    return "\n".join(lines)

def generate_input(
    times: np.ndarray,
    wind_step_hours: float,
    output_step_hours: float,
    compute_step_minutes: int,
    open_segments: dict[int, np.ndarray],
) -> None:
    start = format_swan_time(
        times[0]
    )
    end = format_swan_time(
        times[-1]
    )

    commands = []

    for marker, nodes in sorted(
        open_segments.items()
    ):
        filename = (
            "boundary_east.txt"
            if marker < SOUTH_MARKER_BASE
            else "boundary_south.txt"
        )

        commands.append(
            format_unstructured_segment_command(
                nodes=nodes,
                filename=filename,
            )
        )

    boundary_block = "\n\n".join(commands)

    content = f"""PROJECT 'RES_UNSTRUCT' '01'

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


def triangle_quality(
    points: np.ndarray,
    triangles: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for triangle_index, triangle in enumerate(
        triangles,
        start=1,
    ):
        coordinates = points[triangle]

        side_lengths = np.asarray(
            [
                np.linalg.norm(
                    coordinates[1] - coordinates[0]
                ),
                np.linalg.norm(
                    coordinates[2] - coordinates[1]
                ),
                np.linalg.norm(
                    coordinates[0] - coordinates[2]
                ),
            ]
        )

        a, b, c = side_lengths

        angles = np.degrees(
            np.arccos(
                np.clip(
                    [
                        (b*b + c*c - a*a) / (2*b*c),
                        (c*c + a*a - b*b) / (2*c*a),
                        (a*a + b*b - c*c) / (2*a*b),
                    ],
                    -1.0,
                    1.0,
                )
            )
        )

        area = abs(
            triangle_signed_area(
                points,
                triangle,
            )
        )

        rows.append(
            {
                "triangle_id": triangle_index,
                "area_degree2": area,
                "minimum_angle_degree": float(
                    np.min(angles)
                ),
                "maximum_angle_degree": float(
                    np.max(angles)
                ),
                "minimum_edge_degree": float(
                    np.min(side_lengths)
                ),
                "maximum_edge_degree": float(
                    np.max(side_lengths)
                ),
                "aspect_ratio": float(
                    np.max(side_lengths)
                    / np.min(side_lengths)
                ),
            }
        )

    return pd.DataFrame(rows)


def write_domain_geojson(
    domain: Polygon,
) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "SWAN research domain",
                },
                "geometry": mapping(domain),
            }
        ],
    }

    DOMAIN_GEOJSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def plot_domain_preview(
    domain: Polygon,
    raw_coastlines: list[LineString],
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    for coastline in raw_coastlines:
        x, y = coastline.xy
        axis.plot(
            x,
            y,
            linewidth=0.6,
            alpha=0.45,
        )

    x, y = domain.exterior.xy
    axis.plot(
        x,
        y,
        linewidth=1.6,
        label="Processed model boundary",
    )

    for interior in domain.interiors:
        x, y = interior.xy
        axis.plot(
            x,
            y,
            linewidth=1.2,
        )

    axis.set_title(
        "Extracted bathymetric coastline and model domain"
    )
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(
        True,
        linewidth=0.35,
        linestyle="--",
        alpha=0.3,
    )
    axis.legend()

    figure.savefig(
        DOMAIN_PREVIEW,
        dpi=250,
        facecolor="white",
    )
    plt.close(figure)


def plot_mesh_preview(
    points: np.ndarray,
    triangles: np.ndarray,
    depth: np.ndarray,
    open_segments: dict[int, np.ndarray],
) -> None:
    triangulation = mtri.Triangulation(
        points[:, 0],
        points[:, 1],
        triangles,
    )

    figure, axis = plt.subplots(
        figsize=(11, 8.5),
        constrained_layout=True,
    )

    image = axis.tripcolor(
        triangulation,
        depth,
        shading="gouraud",
    )
    axis.triplot(
        triangulation,
        linewidth=0.12,
        alpha=0.28,
    )

    for marker, nodes in sorted(
        open_segments.items()
    ):
        side = (
            "east"
            if marker < SOUTH_MARKER_BASE
            else "south"
        )

        axis.plot(
            points[nodes, 0],
            points[nodes, 1],
            linewidth=2.2,
            label=f"{side} open segment {marker}",
        )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        pad=0.08,
    )
    colorbar.set_label("Water depth (m)")

    axis.set_title(
        "Research SWAN unstructured mesh"
    )
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")

    figure.savefig(
        MESH_PREVIEW,
        dpi=250,
        facecolor="white",
    )
    plt.close(figure)


def copy_boundary_files() -> None:
    shutil.copy2(
        BOUNDARY_EAST_FILE,
        OUTPUT_DIR / BOUNDARY_EAST_FILE.name,
    )
    shutil.copy2(
        BOUNDARY_SOUTH_FILE,
        OUTPUT_DIR / BOUNDARY_SOUTH_FILE.name,
    )


def write_metadata(
    args: argparse.Namespace,
    domain: Polygon,
    points: np.ndarray,
    triangles: np.ndarray,
    marked_edges: list[tuple[int, int, int]],
    open_segments: dict[int, np.ndarray],
    quality: pd.DataFrame,
    depth_qa: dict[str, float | int],
    mesh_clip_qa: dict[str, float | int],
) -> None:
    metadata = {
        "generator": "Gmsh coastline-following research mesh",
        "coast_level_m": args.coast_level,
        "minimum_water_depth_m": args.minimum_water_depth,
        "depth_repair_tolerance_m": args.depth_repair_tolerance,
        "depth_interpolation_qa": depth_qa,
        "wet_domain_mesh_clipping_qa": mesh_clip_qa,
        "simplify_tolerance_degree": args.simplify,
        "smoothing_distance_degree": args.smooth,
        "coast_target_size_degree": args.coast_size,
        "offshore_target_size_degree": args.offshore_size,
        "refine_distance_min_degree": args.refine_distance_min,
        "refine_distance_max_degree": args.refine_distance_max,
        "domain_area_degree2": float(domain.area),
        "node_count": int(points.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "boundary_edge_count": len(marked_edges),
        "open_segments": {
            str(marker): {
                "node_count": int(nodes.size),
                "node_ids": [
                    int(node) + 1
                    for node in nodes
                ],
            }
            for marker, nodes in open_segments.items()
        },
        "quality": {
            "minimum_angle_degree": float(
                quality["minimum_angle_degree"].min()
            ),
            "median_minimum_angle_degree": float(
                quality["minimum_angle_degree"].median()
            ),
            "maximum_aspect_ratio": float(
                quality["aspect_ratio"].max()
            ),
            "median_aspect_ratio": float(
                quality["aspect_ratio"].median()
            ),
        },
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


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

    print("\nExecuting:")
    print(" ".join(command))

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
        require_files(
            (
                GRID_FILE,
                DEPTH_FILE,
                WIND_FILE,
                BOUNDARY_EAST_FILE,
                BOUNDARY_SOUTH_FILE,
            )
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        longitude, latitude, regular_depth = (
            load_regular_domain()
        )

        raw_domain, raw_coastlines = build_water_polygon(
            longitude,
            latitude,
            regular_depth,
            args.coast_level,
        )

        if args.topology_cleanup < 0:
            raise ValueError("--topology-cleanup cannot be negative.")

        if args.topology_cleanup_attempts < 1:
            raise ValueError("--topology-cleanup-attempts must be at least 1.")

        if args.minimum_hole_area < 0:
            raise ValueError("--minimum-hole-area cannot be negative.")

        last_topology_error: Exception | None = None
        component_cleanup_qa: dict[str, object] = {}
        topology_qa: dict[str, object] = {}

        for topology_attempt in range(args.topology_cleanup_attempts):
            cleanup_distance = (
                args.topology_cleanup * (2 ** topology_attempt)
            )

            print(
                "\nMesh topology attempt "
                f"{topology_attempt + 1}/{args.topology_cleanup_attempts}: "
                f"cleanup={cleanup_distance:.8f}°"
            )

            domain = process_domain_polygon(
                raw_domain,
                args.simplify,
                args.smooth,
                topology_cleanup=cleanup_distance,
                minimum_hole_area=args.minimum_hole_area,
            )

            points, triangles, _ = gmsh_mesh(
                domain,
                longitude,
                latitude,
                args,
            )

            (
                points,
                triangles,
                raw_node_depth,
                mesh_clip_qa,
            ) = clip_mesh_to_wet_domain(
                longitude=longitude,
                latitude=latitude,
                depth=regular_depth,
                points=points,
                triangles=triangles,
                coast_level=args.coast_level,
                repair_tolerance=args.depth_repair_tolerance,
            )

            (
                points,
                triangles,
                raw_node_depth,
                component_cleanup_qa,
            ) = keep_largest_triangle_component(
                points=points,
                triangles=triangles,
                node_values=raw_node_depth,
            )

            try:
                topology_qa = require_swan_mesh_topology(
                    triangles
                )
                mesh_clip_qa.update(component_cleanup_qa)
                mesh_clip_qa.update(topology_qa)
                mesh_clip_qa["topology_cleanup_distance_degree"] = float(
                    cleanup_distance
                )
                mesh_clip_qa["topology_attempt"] = topology_attempt + 1
                break
            except ValueError as exc:
                last_topology_error = exc
                print(f"Topology retry required: {exc}")
        else:
            raise RuntimeError(
                "Could not generate a SWAN-traversable mesh after "
                f"{args.topology_cleanup_attempts} attempts. "
                f"Last error: {last_topology_error}"
            )

        write_domain_geojson(domain)
        plot_domain_preview(
            domain,
            raw_coastlines,
        )

        rebuilt_boundary_edges = classify_exterior_edges(
            points=points,
            triangles=triangles,
            longitude=longitude,
            latitude=latitude,
        )

        marked_edges, open_segments = build_boundary_segments(
            points,
            rebuilt_boundary_edges,
        )

        node_depth, depth_qa = interpolate_depth_to_nodes(
            longitude=longitude,
            latitude=latitude,
            depth=regular_depth,
            points=points,
            coast_level=args.coast_level,
            minimum_water_depth=args.minimum_water_depth,
            repair_tolerance=args.depth_repair_tolerance,
            precomputed_values=raw_node_depth,
        )

        write_triangle_files(
            points,
            triangles,
            marked_edges,
        )
        np.savetxt(
            BOTTOM_FILE,
            node_depth,
            fmt="%.6f",
        )

        times, wind_step_hours = (
            generate_unstructured_wind(points)
        )

        copy_boundary_files()

        generate_input(
            times=times,
            wind_step_hours=wind_step_hours,
            output_step_hours=args.output_step_hours,
            compute_step_minutes=args.compute_step_minutes,
            open_segments=open_segments,
        )

        quality = triangle_quality(
            points,
            triangles,
        )
        quality.to_csv(
            QUALITY_REPORT,
            index=False,
        )

        plot_mesh_preview(
            points,
            triangles,
            node_depth,
            open_segments,
        )

        write_metadata(
            args,
            domain,
            points,
            triangles,
            marked_edges,
            open_segments,
            quality,
            depth_qa,
            mesh_clip_qa,
        )

        print("\nResearch unstructured case generated.")
        print(f"Nodes:       {points.shape[0]}")
        print(f"Triangles:   {triangles.shape[0]}")
        print(
            "Min angle:   "
            f"{quality['minimum_angle_degree'].min():.2f}°"
        )
        print(
            "Median angle:"
            f" {quality['minimum_angle_degree'].median():.2f}°"
        )
        print(
            "Mesh clipping:"
            f" {mesh_clip_qa['removed_triangle_count']} triangles removed "
            f"({100.0 * mesh_clip_qa['removed_triangle_fraction']:.2f}%)"
        )
        print(
            "Depth repair:"
            f" {depth_qa['repaired_node_count']} nodes "
            f"({100.0 * depth_qa['repaired_fraction']:.2f}%)"
        )
        print(
            "Depth range: "
            f"{depth_qa['minimum_after_repair_m']:.2f}–"
            f"{depth_qa['maximum_depth_m']:.2f} m"
        )
        print(
            "Segments:    "
            + ", ".join(
                f"{marker} ({nodes.size} nodes)"
                for marker, nodes in sorted(
                    open_segments.items()
                )
            )
        )
        print(
            "Components:  "
            f"{mesh_clip_qa.get('component_count_before_cleanup', 1)} before, "
            f"{mesh_clip_qa.get('removed_disconnected_triangles', 0)} "
            "disconnected triangles removed"
        )
        print(
            "Boundary QA: "
            f"{mesh_clip_qa.get('boundary_component_count', 0)} components, "
            f"{len(mesh_clip_qa.get('invalid_boundary_degrees', {}))} "
            "invalid-degree vertices"
        )
        print(
            "Topology cleanup: "
            f"{mesh_clip_qa.get('topology_cleanup_distance_degree', 0.0):.8f}° "
            f"(attempt {mesh_clip_qa.get('topology_attempt', 1)})"
        )
        print(f"Output:      {OUTPUT_DIR}")

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
