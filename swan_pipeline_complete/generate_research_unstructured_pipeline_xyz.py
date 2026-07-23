"""
Research-grade SWAN unstructured mesh and input generator.

The script creates a coastline-following unstructured mesh instead of
triangulating the pixels of the regular raster. It is intended as a solid
starting point for research workflows and publication-quality domain figures.

Workflow
--------
1. Read brasil-coast.xyz and crop it to the wind/model domain.
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
data/processed/brasil-coast.xyz
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
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
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

XYZ_BATHYMETRY_FILE = PROCESSED_DIR / "brasil-coast.xyz"
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
        "--bathymetry-xyz",
        type=Path,
        default=XYZ_BATHYMETRY_FILE,
        help="XYZ bathymetry: longitude latitude elevation.",
    )
    parser.add_argument(
        "--xyz-value-mode",
        choices=("positive-depth", "negative-elevation", "auto"),
        default="positive-depth",
        help=(
            "Meaning of the third XYZ column. Use 'positive-depth' for files "
            "such as brasil-coast.xyz, where 3394 means 3394 m water depth; "
            "'negative-elevation' when underwater values are negative; or "
            "'auto' to infer from the cropped sample. Default: positive-depth."
        ),
    )
    parser.add_argument(
        "--xyz-resolution",
        type=float,
        default=0.01,
        help="Temporary local bathymetry grid resolution in degrees.",
    )
    parser.add_argument(
        "--xyz-margin",
        type=float,
        default=0.20,
        help="Crop margin around the model domain in degrees.",
    )
    parser.add_argument(
        "--xyz-chunk-size",
        type=int,
        default=500000,
        help="Rows read per chunk from the national XYZ file.",
    )
    parser.add_argument(
        "--xyz-max-points",
        type=int,
        default=600000,
        help="Maximum cropped points used by scattered interpolation.",
    )
    parser.add_argument(
        "--domain-bounds",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional explicit bounds; otherwise inferred from wind.nc.",
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
        default=0.0025,
        help=(
            "Topology-preserving coastline simplification tolerance in degrees. "
            "Default: 0.006."
        ),
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.004,
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
        "--minimum-boundary-edge",
        type=float,
        default=0.0025,
        help=(
            "Minimum retained boundary-edge length in degrees before Gmsh. "
            "Shorter consecutive segments are merged. Default: 0.0025."
        ),
    )
    parser.add_argument(
        "--boundary-collinearity-tolerance",
        type=float,
        default=0.00025,
        help=(
            "Maximum perpendicular deviation in degrees for removing a nearly "
            "collinear boundary vertex. Default: 0.00025."
        ),
    )
    parser.add_argument(
        "--boundary-cleanup-passes",
        type=int,
        default=12,
        help=(
            "Maximum iterative passes used to remove duplicate, very short "
            "and nearly collinear boundary segments. Default: 12."
        ),
    )
    parser.add_argument(
        "--coast-size",
        type=float,
        default=0.010,
        help="Target Gmsh element size near the coastline in degrees.",
    )
    parser.add_argument(
        "--offshore-size",
        type=float,
        default=0.080,
        help="Target Gmsh element size offshore in degrees.",
    )
    parser.add_argument(
        "--refine-distance-min",
        type=float,
        default=0.035,
        help=(
            "Distance from coastline over which coast-size is retained, "
            "in degrees."
        ),
    )
    parser.add_argument(
        "--refine-distance-max",
        type=float,
        default=0.55,
        help=(
            "Distance from coastline at which offshore-size is reached, "
            "in degrees."
        ),
    )
    parser.add_argument(
        "--minimum-angle",
        type=float,
        default=4.0,
        help=(
            "Minimum triangle angle used for mesh QA, in degrees. "
            "Without --strict-minimum-angle, angles from 2 degrees up to "
            "this value are reported but do not alone reject the mesh. "
            "Default: 4 degrees."
        ),
    )
    parser.add_argument(
        "--strict-minimum-angle",
        action="store_true",
        help=(
            "Reject every mesh containing a triangle below "
            "--minimum-angle. The default mode only rejects angles below "
            "2 degrees when the other SWAN quality limits are satisfied."
        ),
    )
    parser.add_argument(
        "--max-faces-per-vertex",
        type=int,
        default=10,
        help=(
            "Maximum number of incident triangles at one vertex. "
            "SWAN 41.51A supports at most 10. Default: 10."
        ),
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=50.0,
        help=(
            "Maximum triangle quality ratio R/(2r), where R is the "
            "circumradius and r is the inradius. Default: 50."
        ),
    )
    parser.add_argument(
        "--mesh-attempts",
        type=int,
        default=8,
        help=(
            "Maximum automatic Gmsh attempts for each topology cleanup "
            "stage. Default: 8."
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


def infer_model_bounds_from_wind(
    explicit_bounds: list[float] | None,
) -> tuple[float, float, float, float]:
    if explicit_bounds is not None:
        west, south, east, north = map(float, explicit_bounds)
    else:
        with xr.open_dataset(WIND_FILE) as dataset:
            lon_name = find_name(dataset, ("longitude", "lon", "x"))
            lat_name = find_name(dataset, ("latitude", "lat", "y"))
            lon = np.asarray(dataset[lon_name].values, dtype=float).reshape(-1)
            lat = np.asarray(dataset[lat_name].values, dtype=float).reshape(-1)

        lon = np.where(lon > 180.0, lon - 360.0, lon)
        west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
        south, north = float(np.nanmin(lat)), float(np.nanmax(lat))

    if not (west < east and south < north):
        raise ValueError("Invalid model bounds.")

    return west, south, east, north


def read_cropped_xyz(
    path: Path,
    bounds: tuple[float, float, float, float],
    margin: float,
    chunk_size: int,
) -> np.ndarray:
    west, south, east, north = bounds
    pieces: list[np.ndarray] = []

    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=("longitude", "latitude", "elevation"),
        usecols=(0, 1, 2),
        dtype=np.float64,
        chunksize=chunk_size,
    ):
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        values = values[np.isfinite(values).all(axis=1)]
        if values.size == 0:
            continue

        values[:, 0] = np.where(
            values[:, 0] > 180.0,
            values[:, 0] - 360.0,
            values[:, 0],
        )

        keep = (
            (values[:, 0] >= west - margin)
            & (values[:, 0] <= east + margin)
            & (values[:, 1] >= south - margin)
            & (values[:, 1] <= north + margin)
        )
        if np.any(keep):
            pieces.append(values[keep].copy())

    if not pieces:
        raise ValueError(
            "No XYZ points intersect the model domain. "
            f"Requested bounds with margin: "
            f"longitude [{west - margin:.6f}, {east + margin:.6f}], "
            f"latitude [{south - margin:.6f}, {north + margin:.6f}]. "
            "Check the coordinate order and use "
            "--domain-bounds WEST SOUTH EAST NORTH."
        )

    frame = pd.DataFrame(
        np.vstack(pieces),
        columns=("longitude", "latitude", "elevation"),
    )
    frame = (
        frame.groupby(["longitude", "latitude"], as_index=False, sort=False)
        ["elevation"].median()
    )
    return frame.to_numpy(dtype=np.float64)


def spatially_thin_xyz(
    xyz: np.ndarray,
    maximum_points: int,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    if xyz.shape[0] <= maximum_points:
        return xyz

    west, south, east, north = bounds
    cells = max(1, int(math.sqrt(maximum_points)))
    dx = max((east - west) / cells, np.finfo(float).eps)
    dy = max((north - south) / cells, np.finfo(float).eps)

    frame = pd.DataFrame(
        xyz,
        columns=("longitude", "latitude", "elevation"),
    )
    frame["ix"] = np.floor((frame["longitude"] - west) / dx).astype(np.int64)
    frame["iy"] = np.floor((frame["latitude"] - south) / dy).astype(np.int64)

    reduced = (
        frame.groupby(["ix", "iy"], as_index=False, sort=False)
        .agg(
            longitude=("longitude", "mean"),
            latitude=("latitude", "mean"),
            elevation=("elevation", "median"),
        )
    )
    values = reduced[["longitude", "latitude", "elevation"]].to_numpy(dtype=np.float64)
    if values.shape[0] > maximum_points:
        values = values[np.linspace(0, values.shape[0] - 1, maximum_points, dtype=int)]
    return values


def xyz_values_to_water_depth(
    values: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The cropped XYZ file has no finite vertical values.")

    selected_mode = mode
    if mode == "auto":
        positive_fraction = float(np.mean(finite >= 0.0))
        negative_fraction = float(np.mean(finite <= 0.0))

        if positive_fraction >= 0.90:
            selected_mode = "positive-depth"
        elif negative_fraction >= 0.90:
            selected_mode = "negative-elevation"
        else:
            raise ValueError(
                "Could not infer the XYZ vertical convention because the "
                "cropped data contains a substantial mixture of positive and "
                "negative values. Select --xyz-value-mode explicitly."
            )

    if selected_mode == "positive-depth":
        water_depth = finite.copy()
    elif selected_mode == "negative-elevation":
        water_depth = -finite
    else:
        raise ValueError(f"Unsupported XYZ value mode: {selected_mode}")

    return water_depth, selected_mode



def load_xyz_domain(
    path: Path,
    resolution: float,
    margin: float,
    chunk_size: int,
    maximum_points: int,
    explicit_bounds: list[float] | None,
    value_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    Read `longitude latitude elevation`, crop locally, and convert elevation
    to SWAN water depth using the selected third-column convention.
    """
    if resolution <= 0:
        raise ValueError("--xyz-resolution must be positive.")
    if margin < 0:
        raise ValueError("--xyz-margin cannot be negative.")
    if chunk_size < 1000:
        raise ValueError("--xyz-chunk-size must be at least 1000.")
    if maximum_points < 10000:
        raise ValueError("--xyz-max-points must be at least 10000.")

    bounds = infer_model_bounds_from_wind(explicit_bounds)
    xyz = read_cropped_xyz(path, bounds, margin, chunk_size)
    cropped_count = int(xyz.shape[0])

    interpolation_bounds = (
        bounds[0] - margin,
        bounds[1] - margin,
        bounds[2] + margin,
        bounds[3] + margin,
    )
    xyz = spatially_thin_xyz(xyz, maximum_points, interpolation_bounds)

    west, south, east, north = bounds
    longitude = np.arange(west, east + resolution * 0.5, resolution)
    latitude = np.arange(south, north + resolution * 0.5, resolution)
    xx, yy = np.meshgrid(longitude, latitude)

    samples = xyz[:, :2]
    water_depth, selected_value_mode = xyz_values_to_water_depth(
        xyz[:, 2],
        value_mode,
    )

    linear = LinearNDInterpolator(
        samples,
        water_depth,
        fill_value=np.nan,
        rescale=True,
    )
    depth = np.asarray(linear(xx, yy), dtype=np.float64)

    missing = ~np.isfinite(depth)
    nearest_fill_count = int(np.count_nonzero(missing))
    if nearest_fill_count:
        nearest = NearestNDInterpolator(samples, water_depth, rescale=True)
        depth[missing] = nearest(xx[missing], yy[missing])

    metadata = {
        "source": str(path),
        "format": "longitude latitude elevation",
        "xyz_value_mode_requested": value_mode,
        "xyz_value_mode_selected": selected_value_mode,
        "conversion": (
            "water_depth = value"
            if selected_value_mode == "positive-depth"
            else "water_depth = -value"
        ),
        "bounds": {"west": west, "south": south, "east": east, "north": north},
        "crop_margin_degree": float(margin),
        "grid_resolution_degree": float(resolution),
        "cropped_point_count": cropped_count,
        "interpolation_point_count": int(xyz.shape[0]),
        "nearest_fill_cell_count": nearest_fill_count,
        "grid_shape": [int(latitude.size), int(longitude.size)],
    }
    return longitude, latitude, depth, metadata


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


def _point_line_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))

    if length_squared <= np.finfo(float).eps:
        return float(np.linalg.norm(point - start))

    parameter = float(
        np.clip(
            np.dot(point - start, segment) / length_squared,
            0.0,
            1.0,
        )
    )
    projection = start + parameter * segment
    return float(np.linalg.norm(point - projection))


def clean_ring_coordinates(
    coordinates: list[tuple[float, float]],
    minimum_edge: float,
    collinearity_tolerance: float,
    maximum_passes: int,
) -> list[tuple[float, float]]:
    """
    Remove boundary defects that force Gmsh to create sliver triangles.

    The mesh attempts cannot repair a triangle whose tiny edge is constrained
    by the input polygon. This function therefore removes duplicate points,
    merges extremely short consecutive edges and eliminates nearly collinear
    intermediate vertices before the Gmsh geometry is created.
    """
    if minimum_edge <= 0:
        raise ValueError("--minimum-boundary-edge must be positive.")

    if collinearity_tolerance < 0:
        raise ValueError(
            "--boundary-collinearity-tolerance cannot be negative."
        )

    if maximum_passes < 1:
        raise ValueError("--boundary-cleanup-passes must be at least 1.")

    values = np.asarray(coordinates, dtype=float)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Ring coordinates must have shape (N, 2).")

    if values.shape[0] >= 2 and np.allclose(
        values[0],
        values[-1],
        rtol=0.0,
        atol=np.finfo(float).eps * 100,
    ):
        values = values[:-1]

    # Exact/near duplicates first.
    deduplicated = [values[0]]
    for point in values[1:]:
        if np.linalg.norm(point - deduplicated[-1]) > 1.0e-12:
            deduplicated.append(point)

    values = np.asarray(deduplicated, dtype=float)

    for _ in range(maximum_passes):
        count = values.shape[0]
        if count < 4:
            break

        remove = np.zeros(count, dtype=bool)

        # Prefer removing the middle point of the shortest constrained pair.
        for index in range(count):
            previous = values[(index - 1) % count]
            current = values[index]
            following = values[(index + 1) % count]

            previous_length = float(
                np.linalg.norm(current - previous)
            )
            next_length = float(
                np.linalg.norm(following - current)
            )

            if min(previous_length, next_length) < minimum_edge:
                # Do not remove two adjacent vertices in the same pass.
                if not remove[(index - 1) % count]:
                    remove[index] = True
                    continue

            if collinearity_tolerance > 0:
                deviation = _point_line_distance(
                    current,
                    previous,
                    following,
                )
                chord = float(np.linalg.norm(following - previous))

                if (
                    deviation <= collinearity_tolerance
                    and chord >= max(previous_length, next_length)
                ):
                    if not remove[(index - 1) % count]:
                        remove[index] = True

        if not np.any(remove):
            break

        candidate = values[~remove]
        if candidate.shape[0] < 3:
            break

        values = candidate

    if values.shape[0] < 3:
        raise ValueError(
            "Boundary cleanup left fewer than three unique vertices."
        )

    closed = np.vstack((values, values[0]))
    result = [
        (float(x), float(y))
        for x, y in closed
    ]
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
    algorithm: int,
    smoothing_steps: int,
    minimum_boundary_edge: float,
    boundary_collinearity_tolerance: float,
    boundary_cleanup_passes: int,
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
        original_count = len(coordinates) - 1
        coordinates = clean_ring_coordinates(
            coordinates=coordinates,
            minimum_edge=minimum_boundary_edge,
            collinearity_tolerance=(
                boundary_collinearity_tolerance
            ),
            maximum_passes=boundary_cleanup_passes,
        )
        cleaned_count = len(coordinates) - 1

        if cleaned_count != original_count:
            print(
                "Boundary cleanup: "
                f"{original_count} -> {cleaned_count} vertices"
            )

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
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "Sigmoid",
            1,
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "StopAtDistMax",
            1,
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
        0,
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeExtendFromBoundary",
        0,
    )
    gmsh.option.setNumber(
        "Mesh.Algorithm",
        algorithm,
    )
    gmsh.option.setNumber(
        "Mesh.Optimize",
        1,
    )
    gmsh.option.setNumber(
        "Mesh.OptimizeNetgen",
        1,
    )
    gmsh.option.setNumber(
        "Mesh.Smoothing",
        smoothing_steps,
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeMin",
        coast_size * 0.85,
    )
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax",
        offshore_size * 1.05,
    )

    return surface, curve_tags


def _gmsh_mesh_once(
    domain: Polygon,
    longitude: np.ndarray,
    latitude: np.ndarray,
    args: argparse.Namespace,
    algorithm: int,
    smoothing_steps: int,
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
            algorithm=algorithm,
            smoothing_steps=smoothing_steps,
            minimum_boundary_edge=args.minimum_boundary_edge,
            boundary_collinearity_tolerance=(
                args.boundary_collinearity_tolerance
            ),
            boundary_cleanup_passes=args.boundary_cleanup_passes,
        )

        gmsh.model.mesh.generate(2)

        try:
            gmsh.model.mesh.optimize("Laplace2D")
        except Exception as exc:
            print(f"Laplace2D optimization warning: {exc}")

        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception as exc:
            print(f"Netgen optimization warning: {exc}")

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



def mesh_quality_metrics(
    points: np.ndarray,
    triangles: np.ndarray,
    maximum_faces_per_vertex: int,
    minimum_angle_degree: float,
    maximum_aspect_ratio: float,
    strict_minimum_angle: bool,
) -> dict[str, object]:
    """
    Calculate the mesh limits that are most relevant to SWAN.

    Aspect ratio is R/(2r), using circumradius R and inradius r. An
    equilateral triangle therefore has aspect ratio 1.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Mesh points must have shape (N, 2).")

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Mesh triangles must have shape (M, 3).")

    if triangles.size == 0:
        raise ValueError("The mesh contains no triangles.")

    coordinates = points[triangles]
    side_a = np.linalg.norm(coordinates[:, 1] - coordinates[:, 2], axis=1)
    side_b = np.linalg.norm(coordinates[:, 0] - coordinates[:, 2], axis=1)
    side_c = np.linalg.norm(coordinates[:, 0] - coordinates[:, 1], axis=1)

    signed_double_area = (
        (coordinates[:, 1, 0] - coordinates[:, 0, 0])
        * (coordinates[:, 2, 1] - coordinates[:, 0, 1])
        - (coordinates[:, 1, 1] - coordinates[:, 0, 1])
        * (coordinates[:, 2, 0] - coordinates[:, 0, 0])
    )
    area = 0.5 * np.abs(signed_double_area)
    semiperimeter = 0.5 * (side_a + side_b + side_c)

    with np.errstate(divide="ignore", invalid="ignore"):
        inradius = np.divide(
            area,
            semiperimeter,
            out=np.zeros_like(area),
            where=semiperimeter > 0,
        )
        circumradius = np.divide(
            side_a * side_b * side_c,
            4.0 * area,
            out=np.full_like(area, np.inf),
            where=area > 0,
        )
        aspect_ratio = np.divide(
            circumradius,
            2.0 * inradius,
            out=np.full_like(area, np.inf),
            where=inradius > 0,
        )

    def opposite_angle(
        opposite: np.ndarray,
        adjacent_1: np.ndarray,
        adjacent_2: np.ndarray,
    ) -> np.ndarray:
        denominator = 2.0 * adjacent_1 * adjacent_2
        cosine = np.divide(
            adjacent_1**2 + adjacent_2**2 - opposite**2,
            denominator,
            out=np.full_like(opposite, np.nan),
            where=denominator > 0,
        )
        return np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )

    angles = np.column_stack(
        (
            opposite_angle(side_a, side_b, side_c),
            opposite_angle(side_b, side_a, side_c),
            opposite_angle(side_c, side_a, side_b),
        )
    )
    triangle_minimum_angle = np.nanmin(angles, axis=1)

    incident_faces = np.zeros(points.shape[0], dtype=np.int64)
    np.add.at(incident_faces, triangles.reshape(-1), 1)

    # Edge multiplicity detects non-manifold mesh edges.
    edge_counter: Counter[tuple[int, int]] = Counter()
    for node_a, node_b, node_c in triangles:
        edge_counter.update(
            (
                tuple(sorted((int(node_a), int(node_b)))),
                tuple(sorted((int(node_b), int(node_c)))),
                tuple(sorted((int(node_c), int(node_a)))),
            )
        )

    non_manifold_edge_count = sum(
        1 for count in edge_counter.values() if count > 2
    )

    invalid_area_count = int(
        np.count_nonzero(
            ~np.isfinite(area)
            | (area <= np.finfo(np.float64).eps)
        )
    )
    vertices_over_face_limit = int(
        np.count_nonzero(
            incident_faces > maximum_faces_per_vertex
        )
    )
    aspect_ratio_error_count = int(
        np.count_nonzero(
            ~np.isfinite(aspect_ratio)
            | (aspect_ratio > maximum_aspect_ratio)
        )
    )
    minimum_angle_error_count = int(
        np.count_nonzero(
            ~np.isfinite(triangle_minimum_angle)
            | (triangle_minimum_angle < minimum_angle_degree)
        )
    )
    severe_angle_error_count = int(
        np.count_nonzero(
            ~np.isfinite(triangle_minimum_angle)
            | (triangle_minimum_angle < 2.0)
        )
    )

    angle_passed = (
        minimum_angle_error_count == 0
        if strict_minimum_angle
        else severe_angle_error_count == 0
    )

    passed = (
        vertices_over_face_limit == 0
        and aspect_ratio_error_count == 0
        and invalid_area_count == 0
        and non_manifold_edge_count == 0
        and angle_passed
    )

    return {
        "node_count": int(points.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "maximum_faces_per_vertex": int(np.max(incident_faces)),
        "vertices_over_face_limit": vertices_over_face_limit,
        "minimum_angle_degrees": float(
            np.nanmin(triangle_minimum_angle)
        ),
        "minimum_angle_error_count": minimum_angle_error_count,
        "severe_angle_error_count": severe_angle_error_count,
        "maximum_aspect_ratio": float(np.nanmax(aspect_ratio)),
        "aspect_ratio_error_count": aspect_ratio_error_count,
        "invalid_area_count": invalid_area_count,
        "non_manifold_edge_count": int(non_manifold_edge_count),
        "passed": bool(passed),
    }


def _mesh_candidate_score(
    quality: dict[str, object],
) -> tuple[float, ...]:
    """
    Rank failed candidates by SWAN-critical defects first.
    """
    return (
        float(quality["vertices_over_face_limit"]),
        float(quality["non_manifold_edge_count"]),
        float(quality["invalid_area_count"]),
        float(quality["severe_angle_error_count"]),
        float(quality["aspect_ratio_error_count"]),
        float(quality["minimum_angle_error_count"]),
        float(quality["maximum_aspect_ratio"]),
        -float(quality["minimum_angle_degrees"]),
    )


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
    """
    Generate and validate several adaptive Gmsh candidates.

    The function returns only a mesh that satisfies the configured SWAN
    preflight limits. The original args values are restored before returning.
    """
    if args.mesh_attempts < 1:
        raise ValueError("--mesh-attempts must be at least 1.")

    if args.max_faces_per_vertex < 3:
        raise ValueError("--max-faces-per-vertex must be at least 3.")

    if args.max_aspect_ratio <= 1:
        raise ValueError("--max-aspect-ratio must be greater than 1.")

    if args.minimum_angle <= 0 or args.minimum_angle >= 60:
        raise ValueError("--minimum-angle must be between 0 and 60 degrees.")

    algorithms = (6, 5, 6, 5, 8, 6, 5, 8, 6, 5, 8, 6)
    attempts: list[dict[str, object]] = []
    best_quality: dict[str, object] | None = None
    best_score: tuple[float, ...] | None = None

    original = {
        "coast_size": float(args.coast_size),
        "offshore_size": float(args.offshore_size),
        "distance_min": float(args.refine_distance_min),
        "distance_max": float(args.refine_distance_max),
    }

    try:
        for attempt_index in range(args.mesh_attempts):
            attempt = attempt_index + 1
            algorithm = algorithms[
                attempt_index % len(algorithms)
            ]

            # Progressive coarsening removes clusters of tiny triangles and
            # reduces vertex valence. Offshore/coastal grading remains smooth.
            coast_factor = 1.0 + 0.10 * attempt_index
            offshore_factor = 1.0 + 0.04 * attempt_index
            distance_factor = 1.0 + 0.06 * attempt_index

            args.coast_size = original["coast_size"] * coast_factor
            args.offshore_size = max(
                args.coast_size * 2.5,
                original["offshore_size"] * offshore_factor,
            )
            args.refine_distance_min = (
                original["distance_min"] * distance_factor
            )
            args.refine_distance_max = (
                original["distance_max"] * distance_factor
            )
            smoothing_steps = min(80, 20 + 5 * attempt_index)

            print(
                "\nMesh quality attempt "
                f"{attempt}/{args.mesh_attempts}: "
                f"algorithm={algorithm}, "
                f"coast_size={args.coast_size:.6f}, "
                f"offshore_size={args.offshore_size:.6f}, "
                f"smoothing={smoothing_steps}"
            )

            points, triangles, boundary_edges = _gmsh_mesh_once(
                domain=domain,
                longitude=longitude,
                latitude=latitude,
                args=args,
                algorithm=algorithm,
                smoothing_steps=smoothing_steps,
            )

            quality = mesh_quality_metrics(
                points=points,
                triangles=triangles,
                maximum_faces_per_vertex=args.max_faces_per_vertex,
                minimum_angle_degree=args.minimum_angle,
                maximum_aspect_ratio=args.max_aspect_ratio,
                strict_minimum_angle=args.strict_minimum_angle,
            )
            quality.update(
                {
                    "attempt": attempt,
                    "algorithm": algorithm,
                    "coast_size": float(args.coast_size),
                    "offshore_size": float(args.offshore_size),
                    "refine_distance_min": float(
                        args.refine_distance_min
                    ),
                    "refine_distance_max": float(
                        args.refine_distance_max
                    ),
                    "smoothing_steps": smoothing_steps,
                }
            )
            attempts.append(quality)
            print(json.dumps(quality, indent=2))

            score = _mesh_candidate_score(quality)
            if best_score is None or score < best_score:
                best_score = score
                best_quality = quality

            if bool(quality["passed"]):
                report_file = (
                    OUTPUT_DIR / "mesh_generation_attempts.json"
                )
                report_file.write_text(
                    json.dumps(attempts, indent=2),
                    encoding="utf-8",
                )
                print(
                    "Mesh accepted by the SWAN preflight quality checks."
                )
                return points, triangles, boundary_edges

        report_file = OUTPUT_DIR / "mesh_generation_attempts.json"
        report_file.write_text(
            json.dumps(attempts, indent=2),
            encoding="utf-8",
        )

        if best_quality is None:
            raise RuntimeError("Gmsh produced no mesh candidates.")

        raise RuntimeError(
            "No Gmsh candidate satisfied the SWAN quality limits. "
            f"Best candidate: max faces="
            f"{best_quality['maximum_faces_per_vertex']}, "
            f"minimum angle="
            f"{best_quality['minimum_angle_degrees']:.6f}°, "
            f"maximum aspect ratio="
            f"{best_quality['maximum_aspect_ratio']:.6f}. "
            f"Inspect {report_file}."
        )
    finally:
        args.coast_size = original["coast_size"]
        args.offshore_size = original["offshore_size"]
        args.refine_distance_min = original["distance_min"]
        args.refine_distance_max = original["distance_max"]



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
    """
    Return every vertex of one open, non-branched boundary component exactly
    once and in edge-contiguous order.

    Node identifiers need not be numerically consecutive. Continuity is
    established exclusively from mesh-edge connectivity.
    """
    if not edges:
        raise ValueError("Cannot order an empty boundary component.")

    normalized_edges = [
        tuple(sorted((int(node_a), int(node_b))))
        for node_a, node_b in edges
    ]

    if len(set(normalized_edges)) != len(normalized_edges):
        raise ValueError(
            "Boundary component contains duplicate mesh edges."
        )

    adjacency: dict[int, set[int]] = defaultdict(set)

    for node_a, node_b in normalized_edges:
        if node_a == node_b:
            raise ValueError(
                "Boundary component contains a zero-length topological edge."
            )

        adjacency[node_a].add(node_b)
        adjacency[node_b].add(node_a)

    invalid_degrees = {
        node: len(neighbours)
        for node, neighbours in adjacency.items()
        if len(neighbours) not in (1, 2)
    }

    if invalid_degrees:
        examples = list(sorted(invalid_degrees.items()))[:20]
        raise ValueError(
            "Boundary component is branched. Open chains may contain only "
            f"degree-1 endpoints and degree-2 interior vertices: {examples}"
        )

    endpoints = sorted(
        node
        for node, neighbours in adjacency.items()
        if len(neighbours) == 1
    )

    if len(endpoints) != 2:
        raise ValueError(
            "An open SWAN boundary component must contain exactly two "
            f"endpoints; found {len(endpoints)}."
        )

    start_node = endpoints[0]
    ordered = [start_node]
    used_edges: set[tuple[int, int]] = set()
    current = start_node

    while True:
        candidates = [
            neighbour
            for neighbour in adjacency[current]
            if tuple(sorted((current, neighbour))) not in used_edges
        ]

        if not candidates:
            break

        if len(candidates) != 1:
            raise ValueError(
                "Boundary traversal became ambiguous at vertex "
                f"{current + 1}: candidates="
                f"{[item + 1 for item in sorted(candidates)]}"
            )

        next_node = candidates[0]
        edge = tuple(sorted((current, next_node)))
        used_edges.add(edge)
        ordered.append(next_node)
        current = next_node

        if len(ordered) > len(adjacency):
            raise RuntimeError(
                "Boundary traversal repeated a vertex."
            )

    if len(used_edges) != len(normalized_edges):
        raise ValueError(
            "Boundary traversal did not consume every component edge: "
            f"{len(used_edges)}/{len(normalized_edges)}."
        )

    if len(ordered) != len(adjacency):
        raise ValueError(
            "Boundary traversal did not visit every component vertex: "
            f"{len(ordered)}/{len(adjacency)}."
        )

    if len(set(ordered)) != len(ordered):
        raise ValueError(
            "Boundary traversal contains repeated vertices."
        )

    if ordered[-1] not in endpoints or ordered[-1] == start_node:
        raise ValueError(
            "Boundary traversal did not terminate at the opposite endpoint."
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

    traversed_edges = {
        tuple(sorted((int(node_a), int(node_b))))
        for node_a, node_b in zip(nodes[:-1], nodes[1:])
    }

    if traversed_edges != edge_set:
        raise ValueError(
            f"Boundary segment '{side}' does not cover its component exactly. "
            f"Missing={len(edge_set - traversed_edges)}, "
            f"extra={len(traversed_edges - edge_set)}."
        )

    if len(set(map(int, nodes))) != int(nodes.size):
        raise ValueError(
            f"Boundary segment '{side}' contains repeated vertices."
        )

    segment_lengths = np.linalg.norm(
        points[nodes[1:]] - points[nodes[:-1]],
        axis=1,
    )

    if (
        not np.isfinite(segment_lengths).all()
        or np.any(segment_lengths <= 0)
    ):
        raise ValueError(
            f"Boundary segment '{side}' contains invalid geometric edges."
        )

    median_length = float(np.median(segment_lengths))
    maximum_length = float(np.max(segment_lengths))

    if median_length > 0 and maximum_length > 8.0 * median_length:
        longest_index = int(np.argmax(segment_lengths))

        raise ValueError(
            f"Boundary segment '{side}' contains an anomalously long edge "
            f"between vertices {int(nodes[longest_index]) + 1} and "
            f"{int(nodes[longest_index + 1]) + 1}: "
            f"{maximum_length:.8f}°, median={median_length:.8f}°."
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




def format_unstructured_side_command(
    marker: int,
    filename: str,
) -> str:
    """
    Format a marker-based boundary command for an unstructured SWAN grid.

    Triangle boundary markers are stored in the final column of mesh.node.
    SWAN can then recover the complete contiguous boundary from the marker,
    avoiding a long explicit list of boundary-vertex indices.
    """
    if marker <= 0:
        raise ValueError(
            "An unstructured SWAN boundary marker must be positive."
        )

    return (
        f"BOUNDSPEC SIDE {int(marker)} "
        f"CONSTANT FILE '{filename}' 1"
    )


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

        if nodes.size < 2:
            raise ValueError(
                f"Open-boundary marker {marker} has fewer than two nodes."
            )

        commands.append(
            format_unstructured_side_command(
                marker=marker,
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
    bathymetry_metadata: dict[str, object],
) -> None:
    metadata = {
        "generator": "Gmsh coastline-following research mesh",
        "bathymetry": bathymetry_metadata,
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
        "minimum_boundary_edge_degree": args.minimum_boundary_edge,
        "boundary_collinearity_tolerance_degree": (
            args.boundary_collinearity_tolerance
        ),
        "boundary_cleanup_passes": args.boundary_cleanup_passes,
        "domain_area_degree2": float(domain.area),
        "node_count": int(points.shape[0]),
        "triangle_count": int(triangles.shape[0]),
        "boundary_edge_count": len(marked_edges),
        "swan_boundary_command_mode": "marker_based_SIDE",
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
        args.bathymetry_xyz = args.bathymetry_xyz.resolve()
        require_files(
            (
                args.bathymetry_xyz,
                WIND_FILE,
                BOUNDARY_EAST_FILE,
                BOUNDARY_SOUTH_FILE,
            )
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            longitude,
            latitude,
            regular_depth,
            bathymetry_metadata,
        ) = load_xyz_domain(
            path=args.bathymetry_xyz,
            resolution=args.xyz_resolution,
            margin=args.xyz_margin,
            chunk_size=args.xyz_chunk_size,
            maximum_points=args.xyz_max_points,
            explicit_bounds=args.domain_bounds,
            value_mode=args.xyz_value_mode,
        )

        print(
            "XYZ bathymetry: "
            f"{bathymetry_metadata['cropped_point_count']} cropped points; "
            f"{bathymetry_metadata['interpolation_point_count']} used; "
            f"grid={bathymetry_metadata['grid_shape']}; "
            f"value mode={bathymetry_metadata['xyz_value_mode_selected']}"
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

        if args.minimum_boundary_edge <= 0:
            raise ValueError("--minimum-boundary-edge must be positive.")

        if args.boundary_collinearity_tolerance < 0:
            raise ValueError(
                "--boundary-collinearity-tolerance cannot be negative."
            )

        if args.boundary_cleanup_passes < 1:
            raise ValueError(
                "--boundary-cleanup-passes must be at least 1."
            )

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

            final_quality_qa = mesh_quality_metrics(
                points=points,
                triangles=triangles,
                maximum_faces_per_vertex=args.max_faces_per_vertex,
                minimum_angle_degree=args.minimum_angle,
                maximum_aspect_ratio=args.max_aspect_ratio,
                strict_minimum_angle=args.strict_minimum_angle,
            )

            if not final_quality_qa["passed"]:
                print(
                    "Quality retry required after wet-domain clipping: "
                    + json.dumps(final_quality_qa)
                )
                last_topology_error = RuntimeError(
                    "The clipped mesh failed SWAN quality QA."
                )
                continue

            try:
                topology_qa = require_swan_mesh_topology(
                    triangles
                )
                mesh_clip_qa.update(component_cleanup_qa)
                mesh_clip_qa.update(topology_qa)
                mesh_clip_qa["final_mesh_quality"] = final_quality_qa
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
            bathymetry_metadata,
        )

        print("\nResearch unstructured XYZ-bathymetry case generated.")
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
            "Boundary input mode: marker-based BOUNDSPEC SIDE"
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
        final_quality = mesh_clip_qa.get(
            "final_mesh_quality",
            {},
        )
        print(
            "SWAN mesh QA: "
            f"max faces="
            f"{final_quality.get('maximum_faces_per_vertex', 'n/a')}, "
            f"min angle="
            f"{final_quality.get('minimum_angle_degrees', float('nan')):.4f}°, "
            f"max aspect="
            f"{final_quality.get('maximum_aspect_ratio', float('nan')):.4f}"
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