from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


@dataclass
class Mesh:
    node_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    triangles: np.ndarray
    triangle_ids: np.ndarray


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    summary: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Triangle/Gmsh-compatible unstructured mesh "
            "before running SWAN."
        )
    )
    parser.add_argument(
        "--node",
        type=Path,
        default=Path("data/unstructured_research/mesh.node"),
        help="Path to mesh.node.",
    )
    parser.add_argument(
        "--ele",
        type=Path,
        default=Path("data/unstructured_research/mesh.ele"),
        help="Path to mesh.ele.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/unstructured_research/mesh_quality"),
        help="Directory for reports and figures.",
    )
    parser.add_argument(
        "--max-faces-per-vertex",
        type=int,
        default=10,
        help="Maximum number of incident triangles allowed by SWAN.",
    )
    parser.add_argument(
        "--min-angle-error",
        type=float,
        default=5.0,
        help="Reject triangles below this minimum angle in degrees.",
    )
    parser.add_argument(
        "--min-angle-warning",
        type=float,
        default=15.0,
        help="Warn about triangles below this angle in degrees.",
    )
    parser.add_argument(
        "--max-aspect-ratio-error",
        type=float,
        default=50.0,
        help="Reject triangles above this aspect-ratio threshold.",
    )
    parser.add_argument(
        "--max-aspect-ratio-warning",
        type=float,
        default=20.0,
        help="Warn about triangles above this aspect-ratio threshold.",
    )
    parser.add_argument(
        "--duplicate-node-tolerance",
        type=float,
        default=1.0e-10,
        help="Coordinate tolerance for duplicate-node detection.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return a nonzero exit code when warnings exist.",
    )
    return parser.parse_args()


def _read_data_lines(path: Path) -> list[list[str]]:
    lines: list[list[str]] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()

        if not stripped or stripped.startswith("#"):
            continue

        content = stripped.split("#", 1)[0].strip()

        if content:
            lines.append(content.split())

    return lines


def read_node_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _read_data_lines(path)

    if not rows:
        raise ValueError(f"Empty node file: {path}")

    header = rows[0]
    node_count = int(header[0])
    dimension = int(header[1])

    if dimension < 2:
        raise ValueError(
            f"Node file dimension must be at least 2, got {dimension}."
        )

    data = rows[1:1 + node_count]

    if len(data) != node_count:
        raise ValueError(
            f"Node count mismatch: header={node_count}, read={len(data)}."
        )

    node_ids = np.asarray([int(row[0]) for row in data], dtype=int)
    x = np.asarray([float(row[1]) for row in data], dtype=float)
    y = np.asarray([float(row[2]) for row in data], dtype=float)

    return node_ids, x, y


def read_ele_file(
    path: Path,
    node_id_to_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_data_lines(path)

    if not rows:
        raise ValueError(f"Empty element file: {path}")

    header = rows[0]
    triangle_count = int(header[0])
    nodes_per_triangle = int(header[1])

    if nodes_per_triangle != 3:
        raise ValueError(
            f"Only triangular elements are supported, got "
            f"{nodes_per_triangle} nodes per element."
        )

    data = rows[1:1 + triangle_count]

    if len(data) != triangle_count:
        raise ValueError(
            f"Triangle count mismatch: header={triangle_count}, "
            f"read={len(data)}."
        )

    triangle_ids = np.asarray([int(row[0]) for row in data], dtype=int)
    triangles = np.empty((triangle_count, 3), dtype=int)

    for index, row in enumerate(data):
        node_ids = [int(row[1]), int(row[2]), int(row[3])]

        try:
            triangles[index] = [
                node_id_to_index[node_id]
                for node_id in node_ids
            ]
        except KeyError as exc:
            raise ValueError(
                f"Triangle {triangle_ids[index]} references missing "
                f"node ID {exc.args[0]}."
            ) from exc

    return triangle_ids, triangles


def load_mesh(node_path: Path, ele_path: Path) -> Mesh:
    node_ids, x, y = read_node_file(node_path)

    if len(np.unique(node_ids)) != node_ids.size:
        raise ValueError("Duplicate node IDs detected in mesh.node.")

    node_id_to_index = {
        int(node_id): index
        for index, node_id in enumerate(node_ids)
    }

    triangle_ids, triangles = read_ele_file(
        ele_path,
        node_id_to_index,
    )

    if len(np.unique(triangle_ids)) != triangle_ids.size:
        raise ValueError("Duplicate triangle IDs detected in mesh.ele.")

    return Mesh(
        node_ids=node_ids,
        x=x,
        y=y,
        triangles=triangles,
        triangle_ids=triangle_ids,
    )


def triangle_geometry(mesh: Mesh) -> dict[str, np.ndarray]:
    points = np.column_stack((mesh.x, mesh.y))
    triangle_points = points[mesh.triangles]

    a = np.linalg.norm(
        triangle_points[:, 1] - triangle_points[:, 2],
        axis=1,
    )
    b = np.linalg.norm(
        triangle_points[:, 0] - triangle_points[:, 2],
        axis=1,
    )
    c = np.linalg.norm(
        triangle_points[:, 0] - triangle_points[:, 1],
        axis=1,
    )

    double_area_signed = (
        (
            triangle_points[:, 1, 0]
            - triangle_points[:, 0, 0]
        )
        * (
            triangle_points[:, 2, 1]
            - triangle_points[:, 0, 1]
        )
        - (
            triangle_points[:, 1, 1]
            - triangle_points[:, 0, 1]
        )
        * (
            triangle_points[:, 2, 0]
            - triangle_points[:, 0, 0]
        )
    )

    area = np.abs(double_area_signed) * 0.5
    semiperimeter = (a + b + c) * 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        inradius = np.divide(
            area,
            semiperimeter,
            out=np.zeros_like(area),
            where=semiperimeter > 0,
        )
        circumradius = np.divide(
            a * b * c,
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

    def angle(opposite: np.ndarray, side1: np.ndarray, side2: np.ndarray) -> np.ndarray:
        denominator = 2.0 * side1 * side2
        cosine = np.divide(
            side1**2 + side2**2 - opposite**2,
            denominator,
            out=np.full_like(opposite, np.nan),
            where=denominator > 0,
        )
        cosine = np.clip(cosine, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    angle_a = angle(a, b, c)
    angle_b = angle(b, a, c)
    angle_c = angle(c, a, b)
    angles = np.column_stack((angle_a, angle_b, angle_c))

    return {
        "area": area,
        "signed_double_area": double_area_signed,
        "edge_a": a,
        "edge_b": b,
        "edge_c": c,
        "angles": angles,
        "min_angle": np.nanmin(angles, axis=1),
        "max_angle": np.nanmax(angles, axis=1),
        "aspect_ratio": aspect_ratio,
    }


def count_faces_per_vertex(mesh: Mesh) -> np.ndarray:
    counts = np.zeros(mesh.node_ids.size, dtype=int)

    for vertex_index in mesh.triangles.ravel():
        counts[vertex_index] += 1

    return counts


def triangle_components(mesh: Mesh) -> list[np.ndarray]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)

    for triangle_index, triangle in enumerate(mesh.triangles):
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(start), int(end))))
            edge_to_triangles[edge].append(triangle_index)

    adjacency: list[set[int]] = [
        set()
        for _ in range(mesh.triangles.shape[0])
    ]

    for attached in edge_to_triangles.values():
        if len(attached) == 2:
            first, second = attached
            adjacency[first].add(second)
            adjacency[second].add(first)

    unseen = set(range(mesh.triangles.shape[0]))
    components: list[np.ndarray] = []

    while unseen:
        start = next(iter(unseen))
        queue = deque([start])
        unseen.remove(start)
        component: list[int] = []

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)

        components.append(np.asarray(component, dtype=int))

    components.sort(key=len, reverse=True)
    return components


def edge_statistics(mesh: Mesh) -> dict:
    edge_counter: Counter[tuple[int, int]] = Counter()

    for triangle in mesh.triangles:
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge_counter[tuple(sorted((int(start), int(end))))] += 1

    boundary_edges = np.asarray(
        [
            edge
            for edge, count in edge_counter.items()
            if count == 1
        ],
        dtype=int,
    )

    non_manifold_edges = np.asarray(
        [
            edge
            for edge, count in edge_counter.items()
            if count > 2
        ],
        dtype=int,
    )

    boundary_degree = np.zeros(mesh.node_ids.size, dtype=int)

    for start, end in boundary_edges:
        boundary_degree[start] += 1
        boundary_degree[end] += 1

    invalid_boundary_vertices = np.flatnonzero(
        (boundary_degree != 0)
        & (boundary_degree != 2)
    )

    return {
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "boundary_degree": boundary_degree,
        "invalid_boundary_vertices": invalid_boundary_vertices,
    }


def duplicate_coordinate_groups(
    mesh: Mesh,
    tolerance: float,
) -> list[list[int]]:
    if tolerance <= 0:
        return []

    scale = 1.0 / tolerance
    keys = np.column_stack(
        (
            np.round(mesh.x * scale).astype(np.int64),
            np.round(mesh.y * scale).astype(np.int64),
        )
    )

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)

    for index, key in enumerate(keys):
        groups[(int(key[0]), int(key[1]))].append(index)

    return [
        indices
        for indices in groups.values()
        if len(indices) > 1
    ]


def validate_mesh(
    mesh: Mesh,
    args: argparse.Namespace,
) -> tuple[ValidationResult, dict]:
    errors: list[str] = []
    warnings: list[str] = []

    geometry = triangle_geometry(mesh)
    faces_per_vertex = count_faces_per_vertex(mesh)
    components = triangle_components(mesh)
    edges = edge_statistics(mesh)
    duplicates = duplicate_coordinate_groups(
        mesh,
        args.duplicate_node_tolerance,
    )

    repeated_vertices = np.flatnonzero(
        np.apply_along_axis(
            lambda triangle: len(set(map(int, triangle))) < 3,
            1,
            mesh.triangles,
        )
    )
    zero_area = np.flatnonzero(
        ~np.isfinite(geometry["area"])
        | (geometry["area"] <= np.finfo(float).eps)
    )
    inverted = np.flatnonzero(
        geometry["signed_double_area"] < 0
    )
    excessive_faces = np.flatnonzero(
        faces_per_vertex > args.max_faces_per_vertex
    )
    min_angle_errors = np.flatnonzero(
        geometry["min_angle"] < args.min_angle_error
    )
    min_angle_warnings = np.flatnonzero(
        (
            geometry["min_angle"] >= args.min_angle_error
        )
        & (
            geometry["min_angle"] < args.min_angle_warning
        )
    )
    aspect_errors = np.flatnonzero(
        geometry["aspect_ratio"] > args.max_aspect_ratio_error
    )
    aspect_warnings = np.flatnonzero(
        (
            geometry["aspect_ratio"] > args.max_aspect_ratio_warning
        )
        & (
            geometry["aspect_ratio"] <= args.max_aspect_ratio_error
        )
    )

    if repeated_vertices.size:
        errors.append(
            f"{repeated_vertices.size} triangles repeat a node."
        )

    if zero_area.size:
        errors.append(
            f"{zero_area.size} zero-area or invalid triangles."
        )

    if excessive_faces.size:
        worst = int(faces_per_vertex[excessive_faces].max())
        errors.append(
            f"{excessive_faces.size} vertices have more than "
            f"{args.max_faces_per_vertex} incident faces; maximum={worst}."
        )

    if min_angle_errors.size:
        errors.append(
            f"{min_angle_errors.size} triangles have minimum angle below "
            f"{args.min_angle_error:.2f}°; global minimum="
            f"{np.nanmin(geometry['min_angle']):.6f}°."
        )

    if aspect_errors.size:
        errors.append(
            f"{aspect_errors.size} triangles exceed aspect ratio "
            f"{args.max_aspect_ratio_error:.2f}; maximum="
            f"{np.nanmax(geometry['aspect_ratio']):.3f}."
        )

    if len(components) != 1:
        errors.append(
            f"Mesh has {len(components)} connected triangle components."
        )

    if edges["non_manifold_edges"].size:
        errors.append(
            f"{edges['non_manifold_edges'].shape[0]} non-manifold edges "
            "belong to more than two triangles."
        )

    if edges["invalid_boundary_vertices"].size:
        errors.append(
            f"{edges['invalid_boundary_vertices'].size} boundary vertices "
            "have degree different from 2."
        )

    if duplicates:
        errors.append(
            f"{len(duplicates)} duplicate-coordinate node groups detected."
        )

    if min_angle_warnings.size:
        warnings.append(
            f"{min_angle_warnings.size} triangles have minimum angle "
            f"between {args.min_angle_error:.2f}° and "
            f"{args.min_angle_warning:.2f}°."
        )

    if aspect_warnings.size:
        warnings.append(
            f"{aspect_warnings.size} triangles have aspect ratio between "
            f"{args.max_aspect_ratio_warning:.2f} and "
            f"{args.max_aspect_ratio_error:.2f}."
        )

    if inverted.size:
        warnings.append(
            f"{inverted.size} triangles are clockwise oriented. "
            "This may be acceptable, but consistent orientation is preferred."
        )

    summary = {
        "node_count": int(mesh.node_ids.size),
        "triangle_count": int(mesh.triangles.shape[0]),
        "triangle_components": int(len(components)),
        "largest_component_triangles": (
            int(components[0].size)
            if components
            else 0
        ),
        "boundary_edge_count": int(
            edges["boundary_edges"].shape[0]
        ),
        "invalid_boundary_vertex_count": int(
            edges["invalid_boundary_vertices"].size
        ),
        "non_manifold_edge_count": int(
            edges["non_manifold_edges"].shape[0]
        ),
        "duplicate_coordinate_group_count": int(len(duplicates)),
        "maximum_faces_per_vertex": int(faces_per_vertex.max()),
        "vertices_over_face_limit": int(excessive_faces.size),
        "minimum_angle_degrees": float(
            np.nanmin(geometry["min_angle"])
        ),
        "median_minimum_angle_degrees": float(
            np.nanmedian(geometry["min_angle"])
        ),
        "maximum_angle_degrees": float(
            np.nanmax(geometry["max_angle"])
        ),
        "minimum_triangle_area": float(
            np.nanmin(geometry["area"])
        ),
        "median_triangle_area": float(
            np.nanmedian(geometry["area"])
        ),
        "maximum_triangle_area": float(
            np.nanmax(geometry["area"])
        ),
        "maximum_aspect_ratio": float(
            np.nanmax(geometry["aspect_ratio"])
        ),
        "median_aspect_ratio": float(
            np.nanmedian(geometry["aspect_ratio"])
        ),
        "zero_area_triangle_count": int(zero_area.size),
        "repeated_vertex_triangle_count": int(
            repeated_vertices.size
        ),
        "clockwise_triangle_count": int(inverted.size),
        "minimum_angle_error_count": int(
            min_angle_errors.size
        ),
        "minimum_angle_warning_count": int(
            min_angle_warnings.size
        ),
        "aspect_ratio_error_count": int(aspect_errors.size),
        "aspect_ratio_warning_count": int(
            aspect_warnings.size
        ),
        "error_count": len(errors),
        "warning_count": len(warnings),
    }

    diagnostics = {
        "geometry": geometry,
        "faces_per_vertex": faces_per_vertex,
        "components": components,
        "edges": edges,
        "duplicates": duplicates,
        "repeated_vertices": repeated_vertices,
        "zero_area": zero_area,
        "excessive_faces": excessive_faces,
        "min_angle_errors": min_angle_errors,
        "min_angle_warnings": min_angle_warnings,
        "aspect_errors": aspect_errors,
        "aspect_warnings": aspect_warnings,
    }

    return (
        ValidationResult(
            errors=errors,
            warnings=warnings,
            summary=summary,
        ),
        diagnostics,
    )


def write_triangle_quality_csv(
    output_path: Path,
    mesh: Mesh,
    diagnostics: dict,
) -> None:
    geometry = diagnostics["geometry"]
    excessive_faces = set(
        map(int, diagnostics["excessive_faces"])
    )

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "triangle_id",
                "node_id_1",
                "node_id_2",
                "node_id_3",
                "area",
                "min_angle_deg",
                "max_angle_deg",
                "aspect_ratio",
                "has_vertex_over_face_limit",
            ]
        )

        for index, triangle in enumerate(mesh.triangles):
            writer.writerow(
                [
                    int(mesh.triangle_ids[index]),
                    int(mesh.node_ids[triangle[0]]),
                    int(mesh.node_ids[triangle[1]]),
                    int(mesh.node_ids[triangle[2]]),
                    float(geometry["area"][index]),
                    float(geometry["min_angle"][index]),
                    float(geometry["max_angle"][index]),
                    float(geometry["aspect_ratio"][index]),
                    any(
                        int(vertex) in excessive_faces
                        for vertex in triangle
                    ),
                ]
            )


def write_vertex_quality_csv(
    output_path: Path,
    mesh: Mesh,
    diagnostics: dict,
    args: argparse.Namespace,
) -> None:
    faces = diagnostics["faces_per_vertex"]
    boundary_degree = diagnostics["edges"]["boundary_degree"]

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "node_id",
                "x",
                "y",
                "incident_faces",
                "over_swan_limit",
                "boundary_degree",
            ]
        )

        for index, node_id in enumerate(mesh.node_ids):
            writer.writerow(
                [
                    int(node_id),
                    float(mesh.x[index]),
                    float(mesh.y[index]),
                    int(faces[index]),
                    bool(
                        faces[index]
                        > args.max_faces_per_vertex
                    ),
                    int(boundary_degree[index]),
                ]
            )


def plot_quality_map(
    path: Path,
    mesh: Mesh,
    diagnostics: dict,
) -> None:
    geometry = diagnostics["geometry"]
    triangulation = mtri.Triangulation(
        mesh.x,
        mesh.y,
        mesh.triangles,
    )

    figure, axis = plt.subplots(
        figsize=(10, 10),
        constrained_layout=True,
    )
    image = axis.tripcolor(
        triangulation,
        facecolors=geometry["min_angle"],
        shading="flat",
    )
    axis.triplot(
        triangulation,
        linewidth=0.10,
        alpha=0.35,
    )

    bad_vertices = diagnostics["excessive_faces"]

    if bad_vertices.size:
        axis.scatter(
            mesh.x[bad_vertices],
            mesh.y[bad_vertices],
            marker="x",
            s=32,
            linewidths=1.0,
            label="> face limit",
            zorder=5,
        )
        axis.legend(loc="best")

    axis.set_title(
        "Minimum angle per triangle and SWAN-problem vertices"
    )
    axis.set_xlabel("Longitude / X")
    axis.set_ylabel("Latitude / Y")
    axis.set_aspect("equal", adjustable="box")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Minimum angle (degrees)")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def plot_face_count_map(
    path: Path,
    mesh: Mesh,
    diagnostics: dict,
    args: argparse.Namespace,
) -> None:
    counts = diagnostics["faces_per_vertex"]

    figure, axis = plt.subplots(
        figsize=(10, 10),
        constrained_layout=True,
    )
    image = axis.scatter(
        mesh.x,
        mesh.y,
        c=counts,
        s=7,
    )

    excessive = diagnostics["excessive_faces"]

    if excessive.size:
        axis.scatter(
            mesh.x[excessive],
            mesh.y[excessive],
            marker="x",
            s=42,
            linewidths=1.3,
            label=(
                f"> {args.max_faces_per_vertex} faces"
            ),
        )
        axis.legend(loc="best")

    axis.set_title("Incident triangles per vertex")
    axis.set_xlabel("Longitude / X")
    axis.set_ylabel("Latitude / Y")
    axis.set_aspect("equal", adjustable="box")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Incident triangle count")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def write_text_report(
    path: Path,
    result: ValidationResult,
    args: argparse.Namespace,
) -> None:
    lines = [
        "SWAN UNSTRUCTURED MESH QUALITY REPORT",
        "=" * 44,
        "",
        f"Node file: {args.node.resolve()}",
        f"Element file: {args.ele.resolve()}",
        "",
        "Thresholds",
        "-" * 10,
        (
            "Maximum incident faces per vertex: "
            f"{args.max_faces_per_vertex}"
        ),
        (
            "Minimum-angle error threshold: "
            f"{args.min_angle_error:.3f}°"
        ),
        (
            "Minimum-angle warning threshold: "
            f"{args.min_angle_warning:.3f}°"
        ),
        (
            "Aspect-ratio error threshold: "
            f"{args.max_aspect_ratio_error:.3f}"
        ),
        (
            "Aspect-ratio warning threshold: "
            f"{args.max_aspect_ratio_warning:.3f}"
        ),
        "",
        "Summary",
        "-" * 7,
    ]

    for key, value in result.summary.items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "Errors", "-" * 6])

    if result.errors:
        lines.extend(f"- {message}" for message in result.errors)
    else:
        lines.append("- None")

    lines.extend(["", "Warnings", "-" * 8])

    if result.warnings:
        lines.extend(f"- {message}" for message in result.warnings)
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "Final status",
            "-" * 12,
            (
                "FAILED"
                if result.errors
                else "PASSED"
            ),
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    args.node = args.node.resolve()
    args.ele = args.ele.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.node.is_file():
        raise FileNotFoundError(args.node)

    if not args.ele.is_file():
        raise FileNotFoundError(args.ele)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh = load_mesh(args.node, args.ele)
    result, diagnostics = validate_mesh(mesh, args)

    report_path = (
        args.output_dir
        / "mesh_quality_report.txt"
    )
    json_path = (
        args.output_dir
        / "mesh_quality_summary.json"
    )
    triangle_csv = (
        args.output_dir
        / "triangle_quality.csv"
    )
    vertex_csv = (
        args.output_dir
        / "vertex_quality.csv"
    )
    quality_figure = (
        args.output_dir
        / "minimum_angle_quality.png"
    )
    face_figure = (
        args.output_dir
        / "faces_per_vertex.png"
    )

    write_text_report(
        report_path,
        result,
        args,
    )
    json_path.write_text(
        json.dumps(
            result.summary,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_triangle_quality_csv(
        triangle_csv,
        mesh,
        diagnostics,
    )
    write_vertex_quality_csv(
        vertex_csv,
        mesh,
        diagnostics,
        args,
    )
    plot_quality_map(
        quality_figure,
        mesh,
        diagnostics,
    )
    plot_face_count_map(
        face_figure,
        mesh,
        diagnostics,
        args,
    )

    print("SWAN mesh quality validation")
    print(f"Nodes: {result.summary['node_count']}")
    print(f"Triangles: {result.summary['triangle_count']}")
    print(
        "Maximum incident faces: "
        f"{result.summary['maximum_faces_per_vertex']}"
    )
    print(
        "Minimum angle: "
        f"{result.summary['minimum_angle_degrees']:.6f}°"
    )
    print(
        "Maximum aspect ratio: "
        f"{result.summary['maximum_aspect_ratio']:.3f}"
    )
    print(
        "Triangle components: "
        f"{result.summary['triangle_components']}"
    )
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Report: {report_path}")
    print(f"Triangle CSV: {triangle_csv}")
    print(f"Vertex CSV: {vertex_csv}")
    print(f"Quality figure: {quality_figure}")
    print(f"Face-count figure: {face_figure}")

    if result.errors:
        print()
        print("Validation FAILED. Do not run SWAN with this mesh.")

        for message in result.errors:
            print(f"  ERROR: {message}")

        return 2

    if result.warnings:
        print()

        for message in result.warnings:
            print(f"  WARNING: {message}")

        if args.fail_on_warning:
            print(
                "Validation failed because --fail-on-warning was used."
            )
            return 3

    print()
    print("Validation PASSED. The mesh is eligible for SWAN execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
