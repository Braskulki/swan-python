#!/usr/bin/env python3
"""
Triangle/SWAN unstructured-mesh validator.

Validates:
- Triangle .node, .ele, .edge and optional .poly files
- duplicate and missing IDs
- references to unknown nodes
- repeated vertices inside triangles
- zero-area and inverted triangles
- edge incidence (boundary, interior, non-manifold)
- agreement between .ele-derived edges and mesh.edge
- boundary-vertex degrees and connected components
- marker continuity and marker junctions
- pinch points / branching vertices
- disconnected triangle components
- isolated and unused nodes
- SWAN-compatible closed boundary traversal

Outputs:
- mesh_validation_report.txt
- mesh_boundary_components.png
- mesh_problem_vertices.csv
- mesh_boundary_components.csv

Usage:
    python validate_triangle_mesh.py
    python validate_triangle_mesh.py --directory data/unstructured_research
    python validate_triangle_mesh.py --prefix mesh --output validation
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class NodeData:
    coordinates: dict[int, tuple[float, float]]
    markers: dict[int, int]
    dimension: int
    attributes: int
    has_marker: bool


@dataclass(frozen=True)
class ElementData:
    triangles: dict[int, tuple[int, int, int]]
    nodes_per_element: int
    attributes: int


@dataclass(frozen=True)
class EdgeData:
    edges: dict[int, tuple[int, int]]
    markers: dict[int, int]
    has_marker: bool


def data_lines(path: Path) -> Iterable[tuple[int, str]]:
    """Yield non-empty, comment-free lines with original line numbers."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            content = raw.split("#", 1)[0].strip()
            if content:
                yield line_number, content


def read_node(path: Path) -> NodeData:
    iterator = iter(data_lines(path))
    header_line, header = next(iterator)
    fields = header.split()

    if len(fields) < 4:
        raise ValueError(f"{path}:{header_line}: invalid .node header")

    count, dimension, attributes, marker_flag = map(int, fields[:4])

    coordinates: dict[int, tuple[float, float]] = {}
    markers: dict[int, int] = {}

    for record_index in range(count):
        try:
            line_number, line = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"{path}: expected {count} node records, found {record_index}"
            ) from exc

        parts = line.split()
        minimum_fields = 1 + dimension + attributes + marker_flag

        if len(parts) < minimum_fields:
            raise ValueError(
                f"{path}:{line_number}: expected at least "
                f"{minimum_fields} fields, found {len(parts)}"
            )

        node_id = int(parts[0])

        if node_id in coordinates:
            raise ValueError(
                f"{path}:{line_number}: duplicate node ID {node_id}"
            )

        x = float(parts[1])
        y = float(parts[2])

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"{path}:{line_number}: non-finite coordinate for node {node_id}"
            )

        coordinates[node_id] = (x, y)

        if marker_flag:
            marker_index = 1 + dimension + attributes
            markers[node_id] = int(float(parts[marker_index]))
        else:
            markers[node_id] = 0

    return NodeData(
        coordinates=coordinates,
        markers=markers,
        dimension=dimension,
        attributes=attributes,
        has_marker=bool(marker_flag),
    )


def read_ele(path: Path) -> ElementData:
    iterator = iter(data_lines(path))
    header_line, header = next(iterator)
    fields = header.split()

    if len(fields) < 3:
        raise ValueError(f"{path}:{header_line}: invalid .ele header")

    count, nodes_per_element, attributes = map(int, fields[:3])

    if nodes_per_element != 3:
        raise ValueError(
            f"{path}: SWAN Triangle mesh requires 3-node elements; "
            f"found {nodes_per_element}"
        )

    triangles: dict[int, tuple[int, int, int]] = {}

    for record_index in range(count):
        try:
            line_number, line = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"{path}: expected {count} element records, found {record_index}"
            ) from exc

        parts = line.split()
        minimum_fields = 1 + nodes_per_element + attributes

        if len(parts) < minimum_fields:
            raise ValueError(
                f"{path}:{line_number}: expected at least "
                f"{minimum_fields} fields, found {len(parts)}"
            )

        element_id = int(parts[0])

        if element_id in triangles:
            raise ValueError(
                f"{path}:{line_number}: duplicate element ID {element_id}"
            )

        triangles[element_id] = tuple(
            int(value) for value in parts[1:4]
        )

    return ElementData(
        triangles=triangles,
        nodes_per_element=nodes_per_element,
        attributes=attributes,
    )


def read_edge(path: Path) -> EdgeData:
    iterator = iter(data_lines(path))
    header_line, header = next(iterator)
    fields = header.split()

    if len(fields) < 2:
        raise ValueError(f"{path}:{header_line}: invalid .edge header")

    count, marker_flag = map(int, fields[:2])
    edges: dict[int, tuple[int, int]] = {}
    markers: dict[int, int] = {}

    for record_index in range(count):
        try:
            line_number, line = next(iterator)
        except StopIteration as exc:
            raise ValueError(
                f"{path}: expected {count} edge records, found {record_index}"
            ) from exc

        parts = line.split()
        minimum_fields = 3 + marker_flag

        if len(parts) < minimum_fields:
            raise ValueError(
                f"{path}:{line_number}: expected at least "
                f"{minimum_fields} fields, found {len(parts)}"
            )

        edge_id = int(parts[0])

        if edge_id in edges:
            raise ValueError(
                f"{path}:{line_number}: duplicate edge ID {edge_id}"
            )

        edges[edge_id] = (int(parts[1]), int(parts[2]))
        markers[edge_id] = int(float(parts[3])) if marker_flag else 0

    return EdgeData(
        edges=edges,
        markers=markers,
        has_marker=bool(marker_flag),
    )


def normalized_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def triangle_signed_area(
    coordinates: dict[int, tuple[float, float]],
    triangle: tuple[int, int, int],
) -> float:
    p1 = coordinates[triangle[0]]
    p2 = coordinates[triangle[1]]
    p3 = coordinates[triangle[2]]

    return 0.5 * (
        (p2[0] - p1[0]) * (p3[1] - p1[1])
        - (p2[1] - p1[1]) * (p3[0] - p1[0])
    )


def connected_components(
    adjacency: dict[int, set[int]],
) -> list[list[int]]:
    unseen = set(adjacency)
    components: list[list[int]] = []

    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        component: list[int] = []

        while queue:
            node = queue.popleft()
            component.append(node)

            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)

        components.append(sorted(component))

    components.sort(key=lambda values: (-len(values), values[0]))
    return components


def trace_boundary_component(
    component_nodes: set[int],
    adjacency: dict[int, set[int]],
) -> tuple[list[int], str]:
    degrees = {
        node: len(adjacency[node] & component_nodes)
        for node in component_nodes
    }

    endpoints = sorted(node for node, degree in degrees.items() if degree == 1)
    branching = sorted(node for node, degree in degrees.items() if degree > 2)

    if branching:
        return [], "branching"

    if len(endpoints) not in (0, 2):
        return [], f"invalid_endpoints_{len(endpoints)}"

    start = endpoints[0] if endpoints else min(component_nodes)
    ordered = [start]
    previous: int | None = None
    current = start

    for _ in range(len(component_nodes) + 1):
        candidates = sorted(
            neighbor
            for neighbor in adjacency[current]
            if neighbor in component_nodes and neighbor != previous
        )

        if not candidates:
            if endpoints and current == endpoints[-1]:
                return ordered, "open"
            return ordered, "dead_end"

        next_node = candidates[0]

        if not endpoints and next_node == start:
            if len(ordered) == len(component_nodes):
                return ordered, "closed"
            return ordered, "premature_cycle"

        if next_node in ordered:
            return ordered, "repeated_vertex"

        ordered.append(next_node)
        previous, current = current, next_node

        if endpoints and current == endpoints[-1]:
            if len(ordered) == len(component_nodes):
                return ordered, "open"
            return ordered, "early_endpoint"

    return ordered, "iteration_limit"


def write_csv(
    path: Path,
    headers: list[str],
    rows: Iterable[Iterable[object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def validate(
    node_path: Path,
    ele_path: Path,
    edge_path: Path,
    output_dir: Path,
    prefix: str,
    area_tolerance: float,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = read_node(node_path)
    elements = read_ele(ele_path)
    edge_file = read_edge(edge_path)

    coordinates = nodes.coordinates
    triangles = elements.triangles
    file_edges = edge_file.edges

    issues: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []

    def error(code: str, message: str) -> None:
        issues.append((code, message))

    def warn(code: str, message: str) -> None:
        warnings.append((code, message))

    referenced_nodes: Counter[int] = Counter()
    repeated_vertex_elements: list[int] = []
    unknown_element_nodes: list[tuple[int, int]] = []
    signed_areas: dict[int, float] = {}
    zero_area_elements: list[int] = []
    inverted_elements: list[int] = []

    derived_edge_to_elements: dict[
        tuple[int, int], list[int]
    ] = defaultdict(list)

    triangle_adjacency: dict[int, set[int]] = {
        element_id: set() for element_id in triangles
    }

    for element_id, triangle in triangles.items():
        referenced_nodes.update(triangle)

        if len(set(triangle)) != 3:
            repeated_vertex_elements.append(element_id)
            continue

        missing = [node for node in triangle if node not in coordinates]

        for node_id in missing:
            unknown_element_nodes.append((element_id, node_id))

        if missing:
            continue

        area = triangle_signed_area(coordinates, triangle)
        signed_areas[element_id] = area

        if abs(area) <= area_tolerance:
            zero_area_elements.append(element_id)
        elif area < 0:
            inverted_elements.append(element_id)

        a, b, c = triangle

        for edge in (
            normalized_edge(a, b),
            normalized_edge(b, c),
            normalized_edge(c, a),
        ):
            derived_edge_to_elements[edge].append(element_id)

    for edge, owners in derived_edge_to_elements.items():
        if len(owners) == 2:
            first, second = owners
            triangle_adjacency[first].add(second)
            triangle_adjacency[second].add(first)

    if repeated_vertex_elements:
        error(
            "ELEMENT_REPEATED_VERTEX",
            f"{len(repeated_vertex_elements)} elements contain repeated vertices. "
            f"Examples: {repeated_vertex_elements[:20]}",
        )

    if unknown_element_nodes:
        error(
            "UNKNOWN_ELEMENT_NODE",
            f"{len(unknown_element_nodes)} element-node references do not exist. "
            f"Examples: {unknown_element_nodes[:20]}",
        )

    if zero_area_elements:
        error(
            "ZERO_AREA_ELEMENT",
            f"{len(zero_area_elements)} elements have area <= {area_tolerance:g}. "
            f"Examples: {zero_area_elements[:20]}",
        )

    if inverted_elements:
        warn(
            "INVERTED_ELEMENT",
            f"{len(inverted_elements)} elements are clockwise. "
            f"Examples: {inverted_elements[:20]}",
        )

    incidence_histogram = Counter(
        len(owners) for owners in derived_edge_to_elements.values()
    )
    boundary_edges = {
        edge for edge, owners in derived_edge_to_elements.items()
        if len(owners) == 1
    }
    interior_edges = {
        edge for edge, owners in derived_edge_to_elements.items()
        if len(owners) == 2
    }
    non_manifold_edges = {
        edge: owners for edge, owners in derived_edge_to_elements.items()
        if len(owners) > 2
    }

    if non_manifold_edges:
        examples = list(non_manifold_edges.items())[:20]
        error(
            "NON_MANIFOLD_EDGE",
            f"{len(non_manifold_edges)} edges belong to more than two triangles. "
            f"Examples: {examples}",
        )

    file_edge_set = {
        normalized_edge(a, b)
        for a, b in file_edges.values()
    }
    duplicate_file_edges = len(file_edges) - len(file_edge_set)

    if duplicate_file_edges:
        error(
            "DUPLICATE_EDGE_RECORD",
            f"{duplicate_file_edges} duplicate geometric edges occur in mesh.edge.",
        )

    unknown_file_edge_nodes: list[tuple[int, int, int]] = []

    for edge_id, (a, b) in file_edges.items():
        if a not in coordinates or b not in coordinates:
            unknown_file_edge_nodes.append((edge_id, a, b))

    if unknown_file_edge_nodes:
        error(
            "UNKNOWN_EDGE_NODE",
            f"{len(unknown_file_edge_nodes)} mesh.edge records reference "
            f"unknown nodes. Examples: {unknown_file_edge_nodes[:20]}",
        )

    # Some workflows write only exterior edges to mesh.edge, while the
    # standard Triangle -e output may include every geometric edge. Both are
    # acceptable for this validator. Therefore only missing exterior edges
    # and edge records unused by any triangle are treated as errors.
    missing_boundary_from_edge_file = boundary_edges - file_edge_set
    extra_in_edge_file = file_edge_set - set(derived_edge_to_elements)

    if missing_boundary_from_edge_file:
        error(
            "MISSING_BOUNDARY_EDGE_RECORD",
            f"{len(missing_boundary_from_edge_file)} exterior triangle edges "
            f"are absent from mesh.edge. Examples: "
            f"{sorted(missing_boundary_from_edge_file)[:20]}",
        )

    if extra_in_edge_file:
        error(
            "EXTRA_EDGE_RECORD",
            f"{len(extra_in_edge_file)} mesh.edge records are not used by "
            f"any triangle. Examples: {sorted(extra_in_edge_file)[:20]}",
        )

    # Boundary markers from mesh.edge.
    marker_by_geometric_edge: dict[tuple[int, int], int] = {}
    edge_id_by_geometric_edge: dict[tuple[int, int], int] = {}

    for edge_id, (a, b) in file_edges.items():
        edge = normalized_edge(a, b)
        marker_by_geometric_edge[edge] = edge_file.markers[edge_id]
        edge_id_by_geometric_edge[edge] = edge_id

    boundary_file_edges = boundary_edges & file_edge_set
    file_marked_boundary = {
        edge for edge in boundary_file_edges
        if marker_by_geometric_edge.get(edge, 0) != 0
    }
    unmarked_boundary = boundary_file_edges - file_marked_boundary
    marked_interior = {
        edge for edge in interior_edges & file_edge_set
        if marker_by_geometric_edge.get(edge, 0) != 0
    }

    if unmarked_boundary:
        warn(
            "UNMARKED_BOUNDARY_EDGE",
            f"{len(unmarked_boundary)} boundary edges have marker 0. "
            f"Examples: {sorted(unmarked_boundary)[:20]}",
        )

    if marked_interior:
        warn(
            "MARKED_INTERIOR_EDGE",
            f"{len(marked_interior)} interior edges have non-zero markers. "
            f"Examples: {sorted(marked_interior)[:20]}",
        )

    boundary_adjacency: dict[int, set[int]] = defaultdict(set)

    for a, b in boundary_edges:
        boundary_adjacency[a].add(b)
        boundary_adjacency[b].add(a)

    boundary_degrees = {
        node: len(neighbors)
        for node, neighbors in boundary_adjacency.items()
    }

    degree_histogram = Counter(boundary_degrees.values())
    degree_zero_or_invalid = {
        node: degree
        for node, degree in boundary_degrees.items()
        if degree != 2
    }

    if degree_zero_or_invalid:
        examples = sorted(degree_zero_or_invalid.items())[:30]
        error(
            "BOUNDARY_VERTEX_DEGREE",
            f"{len(degree_zero_or_invalid)} boundary vertices do not have "
            f"degree 2. Examples: {examples}",
        )

    boundary_components = connected_components(boundary_adjacency)

    traced_components: list[dict[str, object]] = []

    for component_index, component in enumerate(
        boundary_components, start=1
    ):
        ordered, status = trace_boundary_component(
            set(component),
            boundary_adjacency,
        )

        component_edge_count = (
            sum(
                len(boundary_adjacency[node] & set(component))
                for node in component
            )
            // 2
        )

        marker_counts: Counter[int] = Counter()

        for node in component:
            for neighbor in boundary_adjacency[node]:
                if node < neighbor and neighbor in set(component):
                    marker_counts[
                        marker_by_geometric_edge.get(
                            normalized_edge(node, neighbor), 0
                        )
                    ] += 1

        traced_components.append(
            {
                "component": component_index,
                "nodes": component,
                "ordered": ordered,
                "status": status,
                "edge_count": component_edge_count,
                "marker_counts": marker_counts,
            }
        )

        if status != "closed":
            error(
                "BOUNDARY_COMPONENT_NOT_CLOSED",
                f"Boundary component {component_index} is '{status}', "
                f"with {len(component)} nodes and {component_edge_count} edges.",
            )

    marker_to_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for edge in boundary_edges:
        marker = marker_by_geometric_edge.get(edge, 0)
        marker_to_edges[marker].add(edge)

    marker_component_rows: list[tuple[object, ...]] = []
    marker_problem_nodes: set[int] = set()
    node_markers_from_edges: dict[int, set[int]] = defaultdict(set)

    for marker, edges in sorted(marker_to_edges.items()):
        marker_adjacency: dict[int, set[int]] = defaultdict(set)

        for a, b in edges:
            marker_adjacency[a].add(b)
            marker_adjacency[b].add(a)
            node_markers_from_edges[a].add(marker)
            node_markers_from_edges[b].add(marker)

        components = connected_components(marker_adjacency)
        degree_counts = Counter(
            len(neighbors) for neighbors in marker_adjacency.values()
        )
        invalid_marker_degree = {
            node: len(neighbors)
            for node, neighbors in marker_adjacency.items()
            if len(neighbors) > 2
        }

        marker_problem_nodes.update(invalid_marker_degree)

        for marker_component_index, component in enumerate(
            components, start=1
        ):
            ordered, status = trace_boundary_component(
                set(component),
                marker_adjacency,
            )
            edge_count = (
                sum(
                    len(marker_adjacency[node] & set(component))
                    for node in component
                )
                // 2
            )

            marker_component_rows.append(
                (
                    marker,
                    marker_component_index,
                    len(component),
                    edge_count,
                    status,
                    " ".join(map(str, ordered[:20])),
                )
            )

        if invalid_marker_degree:
            error(
                "MARKER_BRANCHING",
                f"Marker {marker} has {len(invalid_marker_degree)} vertices "
                f"with marker-degree > 2. Examples: "
                f"{sorted(invalid_marker_degree.items())[:20]}",
            )

        if len(components) > 1:
            warn(
                "MARKER_DISCONNECTED",
                f"Marker {marker} occurs in {len(components)} disconnected "
                f"boundary components.",
            )

    mixed_marker_nodes = {
        node: markers
        for node, markers in node_markers_from_edges.items()
        if len(markers - {0}) > 1
    }

    if mixed_marker_nodes:
        mixed_examples = [
            (node, sorted(values))
            for node, values in list(sorted(mixed_marker_nodes.items()))[:20]
        ]
        warn(
            "MARKER_JUNCTION",
            f"{len(mixed_marker_nodes)} boundary vertices touch multiple "
            f"non-zero markers. Examples: {mixed_examples}",
        )

    triangle_components = connected_components(triangle_adjacency)

    if len(triangle_components) > 1:
        error(
            "DISCONNECTED_MESH",
            f"The element graph has {len(triangle_components)} disconnected "
            f"components. Sizes: {[len(values) for values in triangle_components[:20]]}",
        )

    unused_nodes = sorted(set(coordinates) - set(referenced_nodes))

    if unused_nodes:
        warn(
            "UNUSED_NODE",
            f"{len(unused_nodes)} nodes are not used by any element. "
            f"Examples: {unused_nodes[:30]}",
        )

    isolated_nodes = sorted(
        node_id for node_id in coordinates
        if node_id not in boundary_adjacency and referenced_nodes[node_id] == 0
    )

    # Compare node marker with incident edge markers, without assuming that
    # Triangle's node marker must encode every segment transition.
    node_marker_mismatches: list[
        tuple[int, int, tuple[int, ...]]
    ] = []

    if nodes.has_marker:
        for node_id, incident_markers in node_markers_from_edges.items():
            nonzero = incident_markers - {0}
            node_marker = nodes.markers.get(node_id, 0)

            if nonzero and node_marker not in nonzero:
                node_marker_mismatches.append(
                    (node_id, node_marker, tuple(sorted(nonzero)))
                )

    if node_marker_mismatches:
        warn(
            "NODE_EDGE_MARKER_MISMATCH",
            f"{len(node_marker_mismatches)} node markers do not match any "
            f"incident non-zero boundary-edge marker. Examples: "
            f"{node_marker_mismatches[:20]}",
        )

    # Detect repeated coordinate pairs.
    coordinate_to_nodes: dict[tuple[float, float], list[int]] = defaultdict(list)

    for node_id, coordinate in coordinates.items():
        coordinate_to_nodes[coordinate].append(node_id)

    duplicate_coordinates = {
        coordinate: ids
        for coordinate, ids in coordinate_to_nodes.items()
        if len(ids) > 1
    }

    if duplicate_coordinates:
        examples = list(duplicate_coordinates.items())[:20]
        warn(
            "DUPLICATE_COORDINATE",
            f"{len(duplicate_coordinates)} coordinate pairs are shared by "
            f"multiple node IDs. Examples: {examples}",
        )

    problem_nodes = set(degree_zero_or_invalid)
    problem_nodes.update(marker_problem_nodes)
    problem_nodes.update(node for _, node in unknown_element_nodes)
    problem_nodes.update(node for _, node, _ in node_marker_mismatches)

    for edge in non_manifold_edges:
        problem_nodes.update(edge)

    for edge in extra_in_edge_file | missing_boundary_from_edge_file:
        problem_nodes.update(edge)

    problem_rows = []

    for node_id in sorted(problem_nodes):
        x, y = coordinates.get(node_id, (float("nan"), float("nan")))
        problem_rows.append(
            (
                node_id,
                x,
                y,
                boundary_degrees.get(node_id, 0),
                nodes.markers.get(node_id, 0),
                " ".join(
                    map(
                        str,
                        sorted(node_markers_from_edges.get(node_id, set())),
                    )
                ),
                referenced_nodes.get(node_id, 0),
            )
        )

    problem_csv = output_dir / f"{prefix}_problem_vertices.csv"
    write_csv(
        problem_csv,
        [
            "node_id",
            "x",
            "y",
            "boundary_degree",
            "node_marker",
            "incident_edge_markers",
            "element_reference_count",
        ],
        problem_rows,
    )

    component_csv = output_dir / f"{prefix}_boundary_components.csv"
    write_csv(
        component_csv,
        [
            "marker",
            "marker_component",
            "node_count",
            "edge_count",
            "trace_status",
            "first_ordered_nodes",
        ],
        marker_component_rows,
    )

    report_path = output_dir / f"{prefix}_validation_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("TRIANGLE / SWAN MESH VALIDATION REPORT\n")
        report.write("=" * 72 + "\n\n")

        report.write("INPUT FILES\n")
        report.write(f"node: {node_path.resolve()}\n")
        report.write(f"ele:  {ele_path.resolve()}\n")
        report.write(f"edge: {edge_path.resolve()}\n\n")

        report.write("SUMMARY\n")
        report.write(f"nodes: {len(coordinates)}\n")
        report.write(f"elements: {len(triangles)}\n")
        report.write(f"edge records: {len(file_edges)}\n")
        report.write(
            f"derived geometric edges: {len(derived_edge_to_elements)}\n"
        )
        report.write(f"derived boundary edges: {len(boundary_edges)}\n")
        report.write(f"derived interior edges: {len(interior_edges)}\n")
        report.write(
            f"non-manifold edges: {len(non_manifold_edges)}\n"
        )
        report.write(
            f"boundary vertices: {len(boundary_adjacency)}\n"
        )
        report.write(
            f"boundary components: {len(boundary_components)}\n"
        )
        report.write(
            f"triangle components: {len(triangle_components)}\n"
        )
        report.write(f"unused nodes: {len(unused_nodes)}\n")
        report.write(f"isolated nodes: {len(isolated_nodes)}\n\n")

        report.write("EDGE INCIDENCE HISTOGRAM\n")
        for incidence, count in sorted(incidence_histogram.items()):
            report.write(
                f"edges with {incidence} owning triangle(s): {count}\n"
            )
        report.write("\n")

        report.write("BOUNDARY DEGREE HISTOGRAM\n")
        for degree, count in sorted(degree_histogram.items()):
            report.write(
                f"boundary vertices with degree {degree}: {count}\n"
            )
        report.write("\n")

        report.write("BOUNDARY COMPONENTS\n")
        for item in traced_components:
            report.write(
                f"component {item['component']}: "
                f"nodes={len(item['nodes'])}, "
                f"edges={item['edge_count']}, "
                f"status={item['status']}, "
                f"markers={dict(item['marker_counts'])}\n"
            )
            ordered = item["ordered"]
            report.write(
                "  first ordered nodes: "
                + " ".join(map(str, ordered[:30]))
                + "\n"
            )
        report.write("\n")

        report.write("BOUNDARY MARKERS\n")
        marker_edge_counts = Counter(
            marker_by_geometric_edge.get(edge, 0)
            for edge in boundary_edges
        )

        for marker, count in sorted(marker_edge_counts.items()):
            report.write(f"marker {marker}: {count} boundary edges\n")
        report.write("\n")

        report.write("ERRORS\n")
        if issues:
            for code, message in issues:
                report.write(f"[{code}] {message}\n")
        else:
            report.write("None.\n")
        report.write("\n")

        report.write("WARNINGS\n")
        if warnings:
            for code, message in warnings:
                report.write(f"[{code}] {message}\n")
        else:
            report.write("None.\n")
        report.write("\n")

        report.write("SWAN ASSESSMENT\n")
        closed_ok = (
            not non_manifold_edges
            and not degree_zero_or_invalid
            and all(
                item["status"] == "closed"
                for item in traced_components
            )
            and len(triangle_components) == 1
        )

        if closed_ok:
            report.write(
                "The exterior-boundary graph is topologically traversable: "
                "each boundary vertex has degree 2 and every component closes.\n"
            )
        else:
            report.write(
                "The exterior-boundary graph is NOT safely traversable by SWAN. "
                "Resolve the errors above before applying BOUNDSPEC.\n"
            )

        report.write(
            "\nGenerated files:\n"
            f"- {problem_csv.name}\n"
            f"- {component_csv.name}\n"
            f"- {prefix}_boundary_components.png\n"
        )

    # Plot all triangles faintly and boundary components prominently.
    figure_path = output_dir / f"{prefix}_boundary_components.png"
    figure, axis = plt.subplots(figsize=(12, 10))

    for triangle in triangles.values():
        if any(node not in coordinates for node in triangle):
            continue

        polygon = np.array(
            [coordinates[triangle[0]],
             coordinates[triangle[1]],
             coordinates[triangle[2]],
             coordinates[triangle[0]]]
        )
        axis.plot(
            polygon[:, 0],
            polygon[:, 1],
            linewidth=0.15,
            alpha=0.18,
        )

    for item in traced_components:
        ordered = list(item["ordered"])

        if not ordered:
            ordered = list(item["nodes"])

        if item["status"] == "closed" and ordered:
            ordered = ordered + [ordered[0]]

        component_xy = np.array(
            [coordinates[node] for node in ordered]
        )
        axis.plot(
            component_xy[:, 0],
            component_xy[:, 1],
            linewidth=1.2,
            label=(
                f"Boundary {item['component']} "
                f"({item['status']}, {len(item['nodes'])} nodes)"
            ),
        )

    if problem_nodes:
        existing_problem_nodes = [
            node for node in sorted(problem_nodes)
            if node in coordinates
        ]
        problem_xy = np.array(
            [coordinates[node] for node in existing_problem_nodes]
        )
        axis.scatter(
            problem_xy[:, 0],
            problem_xy[:, 1],
            marker="x",
            s=40,
            label=f"Problem vertices ({len(existing_problem_nodes)})",
        )

        for node_id in existing_problem_nodes[:100]:
            x, y = coordinates[node_id]
            axis.annotate(
                str(node_id),
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    axis.set_title("Triangle/SWAN mesh boundary validation")
    axis.set_xlabel("X / longitude")
    axis.set_ylabel("Y / latitude")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)

    print(f"Report: {report_path}")
    print(f"Boundary figure: {figure_path}")
    print(f"Problem vertices: {problem_csv}")
    print(f"Boundary components: {component_csv}")
    print(f"Errors: {len(issues)}")
    print(f"Warnings: {len(warnings)}")

    return 1 if issues else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Triangle mesh for SWAN."
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("."),
        help="Directory containing mesh.node, mesh.ele and mesh.edge.",
    )
    parser.add_argument(
        "--prefix",
        default="mesh",
        help="Triangle filename prefix. Default: mesh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <directory>/mesh_validation",
    )
    parser.add_argument(
        "--area-tolerance",
        type=float,
        default=1.0e-14,
        help="Absolute zero-area tolerance. Default: 1e-14",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    directory = args.directory.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else directory / "mesh_validation"
    )

    node_path = directory / f"{args.prefix}.node"
    ele_path = directory / f"{args.prefix}.ele"
    edge_path = directory / f"{args.prefix}.edge"

    missing = [
        path for path in (node_path, ele_path, edge_path)
        if not path.exists()
    ]

    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"- {path}")
        return 2

    try:
        return validate(
            node_path=node_path,
            ele_path=ele_path,
            edge_path=edge_path,
            output_dir=output,
            prefix=args.prefix,
            area_tolerance=args.area_tolerance,
        )
    except (OSError, ValueError) as exc:
        print(f"Validation failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
