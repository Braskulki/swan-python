"""
Run the generated research-grade SWAN unstructured case through Docker.

Expected files produced by generate_research_unstructured_pipeline.py:

    data/unstructured_research/
    ├── INPUT
    ├── mesh.node
    ├── mesh.ele
    ├── bottom_unstructured.txt
    ├── wind_unstructured.txt
    ├── boundary_east.txt
    └── boundary_south.txt

This runner executes the research case directly in its generated directory:

    data/unstructured_research/
    ├── INPUT
    ├── mesh.node
    ├── mesh.ele
    ├── mesh.edge
    ├── mesh.poly
    ├── bottom_unstructured.txt
    ├── wind_unstructured.txt
    ├── boundary_east.txt
    ├── boundary_south.txt
    ├── PRINT
    └── output_unstructured.mat

Docker defaults:

    image:      openeuler/swan:latest
    executable: /opt/swan/swan.exe

Usage:

    py run_research_unstructured_swan.py
    py run_research_unstructured_swan.py --clean
    py run_research_unstructured_swan.py --show-command
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VALIDATOR_FILE = BASE_DIR / "validate_triangle_mesh.py"
RESEARCH_DIR = BASE_DIR / "data" / "unstructured_research"
CASE_DIR = RESEARCH_DIR

DEFAULT_DOCKER_IMAGE = os.getenv(
    "SWAN_DOCKER_IMAGE",
    "openeuler/swan:latest",
)

DEFAULT_SWAN_EXECUTABLE = os.getenv(
    "SWAN_EXECUTABLE",
    "/opt/swan/swan.exe",
)

REQUIRED_FILES = (
    "INPUT",
    "mesh.node",
    "mesh.ele",
    "bottom_unstructured.txt",
    "wind_unstructured.txt",
    "boundary_east.txt",
    "boundary_south.txt",
)

OPTIONAL_FILES = (
    "mesh.edge",
    "mesh.poly",
    "mesh.msh",
    "mesh_metadata.json",
    "mesh_preview.png",
    "quality_report.csv",
    "coastline_and_domain.png",
    "domain_polygon.geojson",
)

GENERATED_OUTPUTS = (
    "PRINT",
    "output_unstructured.mat",
    "output.mat",
    "norm_end",
    "Errfile",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the research-grade unstructured SWAN case directly through Docker."
        )
    )

    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help=(
            "Docker image containing SWAN. "
            f"Default: {DEFAULT_DOCKER_IMAGE}"
        ),
    )

    parser.add_argument(
        "--swan-executable",
        default=DEFAULT_SWAN_EXECUTABLE,
        help=(
            "SWAN executable inside the container. "
            f"Default: {DEFAULT_SWAN_EXECUTABLE}"
        ),
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove previous SWAN output files from the isolated case "
            "directory before running."
        ),
    )

    parser.add_argument(
        "--show-command",
        action="store_true",
        help="Print the Docker command without executing it.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help=(
            "Maximum execution time in seconds. Default: 7200."
        ),
    )

    return parser.parse_args()


def validate_source_files() -> None:
    if not RESEARCH_DIR.exists():
        raise FileNotFoundError(
            f"Unstructured directory does not exist: {RESEARCH_DIR}"
        )

    missing = [
        RESEARCH_DIR / filename
        for filename in REQUIRED_FILES
        if not (RESEARCH_DIR / filename).exists()
    ]

    if missing:
        listing = "\n".join(f"- {path}" for path in missing)

        raise FileNotFoundError(
            "Required research SWAN files are missing:\n"
            f"{listing}\n\n"
            "Run generate_research_unstructured_pipeline.py first."
        )


def clean_previous_outputs() -> None:
    if not CASE_DIR.exists():
        return

    removed = []

    for filename in GENERATED_OUTPUTS:
        path = CASE_DIR / filename

        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

            removed.append(path.name)

    # SWAN may generate numbered or temporary result files.
    for pattern in (
        "*.mat",
        "*.tab",
        "*.tbl",
        "*.spc",
        "*.prt",
        "*.log",
    ):
        for path in CASE_DIR.glob(pattern):
            if path.name in REQUIRED_FILES:
                continue

            path.unlink()
            removed.append(path.name)

    if removed:
        print(
            "Removed previous outputs: "
            + ", ".join(sorted(set(removed)))
        )


def prepare_case_directory(clean: bool) -> None:
    """
    Validate the generated research case in place.

    Unlike the older unstructured runner, this script does not copy files into
    a nested case directory. SWAN writes PRINT and output_unstructured.mat
    directly into data/unstructured_research, which is also the directory used
    by publication_research_unstructured_results.py.
    """
    validate_source_files()

    if clean:
        clean_previous_outputs()

    print(f"Research case directory: {CASE_DIR}")
    print("Required files:")

    for filename in REQUIRED_FILES:
        path = CASE_DIR / filename
        print(f"  {filename}: {path.stat().st_size:,} bytes")

    available_optional = [
        filename
        for filename in OPTIONAL_FILES
        if (CASE_DIR / filename).exists()
    ]

    if available_optional:
        print("Optional research files:")

        for filename in available_optional:
            path = CASE_DIR / filename
            print(f"  {filename}: {path.stat().st_size:,} bytes")

def docker_mount_path(path: Path) -> str:
    """
    Returns an absolute path suitable for Docker Desktop volume mounting.

    subprocess.run receives arguments directly, so spaces do not need manual
    quoting here.
    """
    return str(path.resolve())


def build_docker_command(
    docker_image: str,
    swan_executable: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{docker_mount_path(CASE_DIR)}:/work",
        "-w",
        "/work",
        docker_image,
        swan_executable,
    ]


def print_command(command: list[str]) -> None:
    print("\nDocker command:")

    # This rendering is for display only. subprocess.run uses the list above.
    display = []

    for argument in command:
        if " " in argument:
            display.append(f'"{argument}"')
        else:
            display.append(argument)

    print(" ".join(display))


def validate_docker_image(image: str) -> None:
    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Docker was not found in PATH. Start Docker Desktop and "
            "confirm that the 'docker' command works."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Docker image was not found locally: {image}\n"
            "Run 'docker images' to verify the image name."
        )


def run_swan(
    command: list[str],
    timeout: int,
) -> None:
    validate_docker_image(command[-2])

    print("\nStarting SWAN...")

    try:
        result = subprocess.run(
            command,
            cwd=CASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"SWAN exceeded the timeout of {timeout} seconds."
        ) from exc

    if result.stdout:
        print("\n--- SWAN STDOUT ---")
        print(result.stdout)

    if result.stderr:
        print("\n--- SWAN STDERR ---")
        print(result.stderr)

    print(f"\nContainer return code: {result.returncode}")

    if result.returncode != 0:
        print_print_tail()

        raise RuntimeError(
            "SWAN execution failed. Check the PRINT file and the messages "
            "shown above."
        )

    validate_outputs()


def print_print_tail(lines: int = 80) -> None:
    print_file = CASE_DIR / "PRINT"

    if not print_file.exists():
        print("\nPRINT file was not generated.")
        return

    try:
        content = print_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError as exc:
        print(f"\nCould not read PRINT: {exc}")
        return

    print(f"\n--- Last {min(lines, len(content))} PRINT lines ---")

    for line in content[-lines:]:
        print(line)


def validate_outputs() -> None:
    print_file = CASE_DIR / "PRINT"
    expected_mat = CASE_DIR / "output_unstructured.mat"
    mat_candidates = sorted(CASE_DIR.glob("*.mat"))

    print("\nExecution completed.")

    if print_file.exists():
        print(
            f"PRINT: {print_file} "
            f"({print_file.stat().st_size:,} bytes)"
        )
    else:
        print("WARNING: PRINT was not found.")

    if expected_mat.exists():
        print(
            f"Expected MAT output: {expected_mat.name} "
            f"({expected_mat.stat().st_size:,} bytes)"
        )
    elif mat_candidates:
        print(
            "WARNING: output_unstructured.mat was not found, "
            "but other MAT files exist:"
        )

        for path in mat_candidates:
            print(
                f"  {path.name}: {path.stat().st_size:,} bytes"
            )
    else:
        print(
            "WARNING: no MATLAB output file was found. "
            "Inspect PRINT for SWAN errors."
        )

    if print_file.exists():
        content = print_file.read_text(
            encoding="utf-8",
            errors="replace",
        )

        severe_count = content.count("Severe error")
        error_count = content.count("** Error")

        print(f"Severe errors in PRINT: {severe_count}")
        print(f"Errors in PRINT: {error_count}")

        if severe_count or error_count:
            print_print_tail()



def validate_mesh_before_swan() -> None:
    """Validate mesh topology before starting SWAN."""
    if not VALIDATOR_FILE.exists():
        raise FileNotFoundError(
            f"Mesh validator was not found: {VALIDATOR_FILE}"
        )

    command = [
        sys.executable,
        str(VALIDATOR_FILE),
        "--directory",
        str(RESEARCH_DIR),
        "--output",
        str(RESEARCH_DIR / "mesh_validation"),
    ]

    print("Validating mesh topology before SWAN...")
    completed = subprocess.run(command, check=False)

    if completed.returncode != 0:
        raise RuntimeError(
            "Mesh validation failed. Inspect "
            f"{RESEARCH_DIR / 'mesh_validation' / 'mesh_validation_report.txt'}"
        )



def main() -> int:
    args = parse_args()

    validate_mesh_before_swan()
    try:
        prepare_case_directory(
            clean=args.clean,
        )

        command = build_docker_command(
            docker_image=args.docker_image,
            swan_executable=args.swan_executable,
        )

        print_command(command)

        if args.show_command:
            print(
                "\nCommand displayed only. Docker was not started."
            )
            return 0

        run_swan(
            command=command,
            timeout=args.timeout,
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())