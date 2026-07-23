# SWAN Unstructured Mesh Pipeline — Command-Line Parameters

# 11. Parameter summary

| Parameter | Type | Runtime default |
|---|---|---:|
| `--coast-level` | float | `0.0` |
| `--bathymetry-xyz` | path | configured script path |
| `--xyz-value-mode` | choice | `positive-depth` |
| `--xyz-resolution` | float | `0.01` |
| `--xyz-margin` | float | `0.20` |
| `--xyz-chunk-size` | integer | `500000` |
| `--xyz-max-points` | integer | `600000` |
| `--domain-bounds` | four floats | inferred from `wind.nc` |
| `--minimum-water-depth` | float | `0.5` |
| `--depth-repair-tolerance` | float | `2.0` |
| `--simplify` | float | `0.0025` |
| `--smooth` | float | `0.004` |
| `--topology-cleanup` | float | `0.0005` |
| `--topology-cleanup-attempts` | integer | `5` |
| `--minimum-hole-area` | float | `1e-6` |
| `--minimum-boundary-edge` | float | `0.0025` |
| `--boundary-collinearity-tolerance` | float | `0.00025` |
| `--boundary-cleanup-passes` | integer | `12` |
| `--coast-size` | float | `0.010` |
| `--offshore-size` | float | `0.080` |
| `--refine-distance-min` | float | `0.035` |
| `--refine-distance-max` | float | `0.55` |
| `--minimum-angle` | float | `4.0` |
| `--strict-minimum-angle` | flag | disabled |
| `--max-faces-per-vertex` | integer | `10` |
| `--max-aspect-ratio` | float | `50.0` |
| `--mesh-attempts` | integer | `8` |
| `--compute-step-minutes` | integer | `15` |
| `--output-step-hours` | float | `6.0` |
| `--run` | flag | disabled |
| `--docker-image` | string | configured default |
| `--swan-executable` | string | `/opt/swan/swan.exe` |

This document describes all command-line parameters supported by:

```text
generate_research_unstructured_pipeline_xyz.py
```

The script reads scattered XYZ bathymetry, creates a temporary interpolated grid, extracts and cleans the wet-domain boundary, generates a Gmsh triangular mesh, performs SWAN-oriented quality assurance, interpolates depths onto mesh nodes, writes the SWAN case files, and can optionally run SWAN through Docker.

---

## Basic usage

```powershell
py .\generate_research_unstructured_pipeline_xyz_v2.py `
  --bathymetry-xyz ".\data\processed\CENTRO.xyz" `
  --xyz-value-mode positive-depth `
  --domain-bounds -37.0 -11.0 -34.5 -8.0
```

To display the parameters directly from the script:

```powershell
py .\generate_research_unstructured_pipeline_xyz_v2.py --help
```

---

# 1. Bathymetry input and interpolation

## `--bathymetry-xyz`

Path to the scattered bathymetric XYZ file.

Expected format:

```text
longitude latitude value
```

Example:

```text
-36.125000 -9.450000 3250.0
```

The columns must be separated by spaces or other whitespace.

Example:

```powershell
--bathymetry-xyz ".\data\processed\CENTRO.xyz"
```

If omitted, the script uses its internally configured default XYZ path.

---

## `--xyz-value-mode`

Defines the meaning of the third XYZ column.

Accepted values:

### `positive-depth`

Positive values represent water depth below mean sea level.

Example:

```text
-36.125 -9.450 3250
```

means a water depth of approximately `3250 m`.

Values at or below the coastline level may represent land, coastline, or shallow cells, depending on the source dataset.

This is the recommended mode for the supplied `CENTRO.xyz`, `SUL.xyz`, and `NORTE.xyz` datasets.

```powershell
--xyz-value-mode positive-depth
```

### `negative-elevation`

Negative values represent seabed elevation relative to mean sea level.

Example:

```text
-36.125 -9.450 -3250
```

is converted internally to a positive SWAN water depth of `3250 m`.

```powershell
--xyz-value-mode negative-elevation
```

### `auto`

The script attempts to infer the vertical convention from the cropped data.

This should only be used when the cropped values are overwhelmingly positive or overwhelmingly negative. It may fail when the dataset contains a substantial mixture of positive, negative, and zero values.

```powershell
--xyz-value-mode auto
```

Default:

```text
positive-depth
```

---

## `--xyz-resolution`

Resolution, in decimal degrees, of the temporary regular bathymetry grid created from the scattered XYZ points.

This grid is used for:

- scattered interpolation;
- coastline contour extraction;
- wet-domain polygon construction;
- subsequent depth interpolation support.

Smaller values produce a finer temporary grid but require more memory and processing time.

Typical values:

| Value | Approximate north-south spacing |
|---:|---:|
| `0.005` | about 0.56 km |
| `0.010` | about 1.11 km |
| `0.015` | about 1.67 km |
| `0.020` | about 2.22 km |

Longitude spacing in kilometres varies with latitude.

Example:

```powershell
--xyz-resolution 0.01
```

Default:

```text
0.01 degrees
```

This parameter controls the intermediate raster-like grid, not the final unstructured mesh size.

---

## `--xyz-margin`

Additional margin, in decimal degrees, used when cropping XYZ points around the model domain.

The interpolation dataset is selected from:

```text
WEST  - xyz-margin
SOUTH - xyz-margin
EAST  + xyz-margin
NORTH + xyz-margin
```

The margin helps:

- reduce interpolation edge effects;
- provide points outside the computational boundary;
- improve coastline continuity near domain limits.

Example:

```powershell
--xyz-margin 0.20
```

Default:

```text
0.20 degrees
```

A larger margin increases the number of points loaded and the interpolation cost.

---

## `--xyz-chunk-size`

Number of rows read at a time from the XYZ file.

The script streams the file in chunks so that large national bathymetry files do not need to be loaded entirely into memory.

Example:

```powershell
--xyz-chunk-size 500000
```

Default:

```text
500000 rows
```

Guidance:

- decrease this value if memory is limited;
- increase it slightly when sufficient memory is available and file reading is a bottleneck;
- this does not directly change mesh quality.

---

## `--xyz-max-points`

Maximum number of cropped XYZ points retained for scattered interpolation.

When the cropped area contains more points than this limit, the script spatially reduces the interpolation sample.

Example:

```powershell
--xyz-max-points 600000
```

Default:

```text
600000 points
```

Higher values may improve preservation of small bathymetric features but increase:

- triangulation time;
- interpolation time;
- memory consumption.

Lower values make preprocessing faster but may smooth or omit local bathymetric detail.

---

# 2. Computational domain

## `--domain-bounds`

Explicit geographic bounds of the model domain.

Required order:

```text
WEST SOUTH EAST NORTH
```

Example:

```powershell
--domain-bounds -37.0 -11.0 -34.5 -8.0
```

This means:

| Boundary | Value |
|---|---:|
| West | `-37.0` |
| South | `-11.0` |
| East | `-34.5` |
| North | `-8.0` |

If this parameter is omitted, the script attempts to infer the domain from `wind.nc`.

Important rules:

```text
WEST < EAST
SOUTH < NORTH
```

The XYZ dataset must overlap the domain after the XYZ crop margin is applied.

---

# 3. Coastline definition and node-depth handling

## `--coast-level`

Bathymetric contour, in metres, used to define the coastline and wet-domain boundary.

Example:

```powershell
--coast-level 0.0
```

Default:

```text
0.0 m
```

With `positive-depth` data:

- values greater than the coast level are normally interpreted as water;
- values near or below the coast level form land or the coastline.

Changing this value shifts the extracted wet/dry boundary.

Examples:

```powershell
--coast-level 0
```

uses the zero-depth contour.

```powershell
--coast-level 2
```

may exclude water shallower than approximately `2 m`.

Use a non-zero value only when the vertical datum and scientific purpose justify it.

---

## `--minimum-water-depth`

Minimum positive water depth assigned to mesh nodes located on, or extremely close to, the selected coastline contour.

Example:

```powershell
--minimum-water-depth 0.5
```

Default:

```text
0.5 m
```

This prevents SWAN from receiving zero or non-positive depths at valid wet mesh nodes caused by interpolation or contour-rounding effects.

It does not generally deepen the whole model. It is intended only for nodes sufficiently close to the accepted coastal contour.

---

## `--depth-repair-tolerance`

Maximum bathymetric mismatch, in metres, that may be repaired using `--minimum-water-depth`.

Example:

```powershell
--depth-repair-tolerance 2.0
```

Default:

```text
2.0 m
```

For a coastline level of `0 m`, a node with an interpolated value only slightly below zero may be repaired. A node substantially farther landward than the allowed tolerance causes an error rather than being silently converted into water.

Conceptually, the repair is accepted only near:

```text
coast_level ± depth_repair_tolerance
```

A larger value makes the repair more permissive. A smaller value makes the land/water validation stricter.

---

# 4. Coastline geometry processing

## `--simplify`

Topology-preserving coastline simplification tolerance, in decimal degrees.

The simplification removes small geometric variations while attempting to preserve the polygon topology.

Example:

```powershell
--simplify 0.005
```

Actual runtime default:

```text
0.0025 degrees
```

Effects of increasing the value:

- fewer coastline vertices;
- fewer short boundary segments;
- faster Gmsh processing;
- lower mesh density near geometrically complex coastlines;
- reduced representation of small bays, islands, and shoreline details.

Effects of decreasing the value:

- greater coastline detail;
- more boundary vertices;
- potentially more elements;
- greater risk of short edges and poor triangles.

Note: the script's argument help text mentions `0.006`, but the actual configured default in the code is `0.0025`.

---

## `--smooth`

Polygon smoothing distance, in decimal degrees.

The script applies a morphological smoothing operation equivalent to:

```text
buffer(+distance).buffer(-distance)
```

Example:

```powershell
--smooth 0.004
```

Actual runtime default:

```text
0.004 degrees
```

Effects:

- rounds sharp coastline corners;
- removes very small indentations;
- may improve triangle quality;
- may slightly shift the extracted coastline.

Set to zero to disable this smoothing stage:

```powershell
--smooth 0
```

Note: the argument help text says `Default: 0`, but the actual configured default in the code is `0.004`.

---

## `--topology-cleanup`

Initial erosion/dilation distance, in decimal degrees, used to remove problematic wet-domain topology.

This stage targets:

- point contacts;
- extremely narrow wet passages;
- narrow polygon necks;
- branched boundary configurations that SWAN cannot traverse safely.

Example:

```powershell
--topology-cleanup 0.0005
```

Default:

```text
0.0005 degrees
```

The cleanup distance is increased automatically between topology attempts.

Set to zero to disable the initial topology-cleanup distance:

```powershell
--topology-cleanup 0
```

However, disabling it may allow boundary structures that are valid geometrically but unsuitable for SWAN.

---

## `--topology-cleanup-attempts`

Maximum number of automatic topology-cleanup and remeshing attempts.

Example:

```powershell
--topology-cleanup-attempts 5
```

Default:

```text
5 attempts
```

For each subsequent attempt, the script increases the cleanup distance. This allows it to progressively eliminate branched or non-traversable boundary structures.

Higher values provide more opportunities for automatic recovery but can cause stronger coastline modification in later attempts.

Minimum valid value:

```text
1
```

---

## `--minimum-hole-area`

Minimum retained polygon-hole area, in square degrees.

Holes smaller than this threshold are removed before meshing.

Example:

```powershell
--minimum-hole-area 1e-6
```

Default:

```text
1e-6 square degrees
```

This is primarily used to remove tiny land holes or geometric artifacts that would otherwise generate unnecessary internal boundaries.

Increasing it removes larger holes or islands.

Setting it to zero preserves every detected hole:

```powershell
--minimum-hole-area 0
```

Use caution because many tiny holes can substantially increase boundary and mesh complexity.

---

## `--minimum-boundary-edge`

Minimum boundary-segment length retained before Gmsh, in decimal degrees.

Consecutive boundary segments shorter than this threshold are merged during iterative cleanup.

Example:

```powershell
--minimum-boundary-edge 0.005
```

Default:

```text
0.0025 degrees
```

Increasing this value:

- removes short coastline edges;
- reduces boundary-node count;
- reduces the risk of sliver triangles;
- makes the boundary coarser.

Decreasing it:

- preserves more coastline detail;
- may produce very short mesh edges;
- may make SWAN-oriented mesh QA harder to satisfy.

This value must be greater than zero.

---

## `--boundary-collinearity-tolerance`

Maximum perpendicular deviation, in decimal degrees, used to classify a boundary vertex as nearly collinear with its neighbours.

Nearly collinear vertices are removed.

Example:

```powershell
--boundary-collinearity-tolerance 0.0005
```

Default:

```text
0.00025 degrees
```

Increasing this value removes more nearly straight-line vertices.

Setting it to zero disables tolerance-based collinear removal, except for exact geometric degeneracies handled elsewhere.

---

## `--boundary-cleanup-passes`

Maximum number of iterative boundary-cleanup passes.

Each pass may remove:

- duplicate vertices;
- repeated consecutive coordinates;
- very short edges;
- nearly collinear vertices.

Example:

```powershell
--boundary-cleanup-passes 12
```

Default:

```text
12 passes
```

The process may stop earlier when no additional changes are needed.

Minimum valid value:

```text
1
```

---

# 5. Mesh-size controls

All mesh-size and refinement-distance values in this section are expressed in decimal degrees.

## `--coast-size`

Target Gmsh element size near the coastline.

Example:

```powershell
--coast-size 0.03
```

Default:

```text
0.010 degrees
```

Approximate north-south scales:

| Value | Approximate size |
|---:|---:|
| `0.010` | 1.1 km |
| `0.020` | 2.2 km |
| `0.030` | 3.3 km |
| `0.050` | 5.6 km |

Increasing this value reduces coastal mesh density.

Decreasing it increases coastal resolution and computational cost.

For offshore-oriented studies, a value such as `0.03` or `0.05` may be more appropriate than the default.

---

## `--offshore-size`

Target Gmsh element size far from the coastline.

Example:

```powershell
--offshore-size 0.08
```

Default:

```text
0.080 degrees
```

This generally controls the largest target triangles in the offshore portion of the domain.

The offshore size should normally be greater than the coast size.

Typical configurations:

```text
coast-size    = 0.03
offshore-size = 0.08
```

or for a coarser regional model:

```text
coast-size    = 0.05
offshore-size = 0.10
```

---

## `--refine-distance-min`

Distance from the coastline over which `--coast-size` is retained.

Example:

```powershell
--refine-distance-min 0.06
```

Default:

```text
0.035 degrees
```

Inside this distance, the mesh target remains close to the coastal element size.

A smaller value confines fine resolution to a narrower coastal strip.

A larger value carries fine resolution farther offshore.

---

## `--refine-distance-max`

Distance from the coastline at which the target mesh size reaches `--offshore-size`.

Example:

```powershell
--refine-distance-max 0.35
```

Default:

```text
0.55 degrees
```

Between `refine-distance-min` and `refine-distance-max`, Gmsh transitions from the coastal size to the offshore size.

Expected relationship:

```text
refine-distance-min < refine-distance-max
```

For an offshore-focused study, reducing `refine-distance-max` limits the width of the refined coastal band.

---

# 6. Mesh-quality acceptance criteria

## `--minimum-angle`

Minimum triangle angle used by the mesh quality-assurance stage, in degrees.

Example:

```powershell
--minimum-angle 4.0
```

Default:

```text
4.0 degrees
```

Valid values must be greater than `0` and less than `60`.

Without `--strict-minimum-angle`:

- triangles below approximately `2°` are treated as severe;
- triangles between approximately `2°` and the configured minimum are reported;
- those reported triangles do not necessarily reject the mesh by themselves if the other SWAN quality limits pass.

With `--strict-minimum-angle`, every triangle below this value rejects the mesh.

Higher values demand better-shaped triangles but may make mesh generation more difficult.

---

## `--strict-minimum-angle`

Boolean flag that enables strict enforcement of `--minimum-angle`.

Usage:

```powershell
--strict-minimum-angle
```

This option does not take a numeric value.

When omitted, the script uses its less restrictive SWAN-oriented minimum-angle acceptance logic.

When included, any triangle with an angle below `--minimum-angle` causes that mesh attempt to fail.

---

## `--max-faces-per-vertex`

Maximum number of triangles allowed to meet at a single mesh vertex.

Example:

```powershell
--max-faces-per-vertex 10
```

Default:

```text
10
```

SWAN 41.51A supports at most 10 incident faces around a vertex in this workflow.

A mesh exceeding this limit is rejected.

Minimum accepted command-line value:

```text
3
```

For SWAN 41.51A, keep this value at `10` or lower.

---

## `--max-aspect-ratio`

Maximum accepted triangle quality ratio:

```text
R / (2r)
```

where:

- `R` is the triangle circumradius;
- `r` is the triangle inradius.

Example:

```powershell
--max-aspect-ratio 50
```

Default:

```text
50
```

An equilateral triangle has a value of `1`.

Larger values indicate increasingly elongated or distorted triangles.

The value must be greater than `1`.

Lower limits demand better triangles but can require more Gmsh retries or more aggressive boundary cleanup.

---

## `--mesh-attempts`

Maximum number of automatic Gmsh attempts for each topology-cleanup stage.

Example:

```powershell
--mesh-attempts 8
```

Default:

```text
8 attempts
```

After a failed mesh QA attempt, the script automatically adjusts mesh-generation settings, including target sizes and refinement distances, and tries again.

The topology and mesh retries are nested conceptually:

```text
topology-cleanup attempts
    └── mesh attempts for each topology stage
```

Therefore, large values for both parameters can significantly increase total execution time.

Minimum valid value:

```text
1
```

---

# 7. SWAN temporal controls

## `--compute-step-minutes`

SWAN computational time step, in minutes.

Example:

```powershell
--compute-step-minutes 15
```

Default:

```text
15 minutes
```

This value is written into the generated SWAN input configuration.

Smaller time steps:

- improve temporal resolution;
- may improve numerical stability in some cases;
- increase simulation cost.

Larger time steps:

- reduce runtime;
- may reduce temporal accuracy;
- should remain compatible with forcing resolution and wave-propagation scales.

---

## `--output-step-hours`

Interval between requested SWAN outputs, in hours.

Example:

```powershell
--output-step-hours 6
```

Default:

```text
6.0 hours
```

This controls output frequency rather than the internal computational time step.

Examples:

```powershell
--output-step-hours 1
```

requests hourly output.

```powershell
--output-step-hours 6
```

requests output every six hours.

More frequent output increases file size and post-processing volume.

---

# 8. Optional SWAN execution through Docker

## `--run`

Boolean flag that runs SWAN through Docker after all files have been generated and validated.

Usage:

```powershell
--run
```

When omitted, the script only generates the mesh, QA outputs, metadata, bathymetry, boundaries, and SWAN input files.

This option does not take a value.

---

## `--docker-image`

Docker image used when `--run` is enabled.

Example:

```powershell
--docker-image your-swan-image:latest
```

If omitted, the script uses its configured default, which may also be influenced by the environment in which the script is executed.

This option has no practical effect unless `--run` is provided.

---

## `--swan-executable`

Path to the SWAN executable inside the Docker container.

Example:

```powershell
--swan-executable /opt/swan/swan.exe
```

Default:

```text
/opt/swan/swan.exe
```

The default may be overridden through the `SWAN_EXECUTABLE` environment variable.

This parameter is only used when `--run` is enabled.

---

# 9. Recommended offshore-oriented configuration

For a study focused mainly on offshore wave propagation rather than detailed nearshore processes:

```powershell
py .\generate_research_unstructured_pipeline_xyz_v2.py `
  --bathymetry-xyz ".\data\processed\CENTRO.xyz" `
  --xyz-value-mode positive-depth `
  --domain-bounds -37.0 -11.0 -34.5 -8.0 `
  --coast-level 0.0 `
  --xyz-resolution 0.015 `
  --xyz-margin 0.20 `
  --xyz-chunk-size 500000 `
  --xyz-max-points 600000 `
  --minimum-water-depth 0.5 `
  --depth-repair-tolerance 2.0 `
  --simplify 0.005 `
  --smooth 0.006 `
  --topology-cleanup 0.0005 `
  --topology-cleanup-attempts 5 `
  --minimum-hole-area 1e-6 `
  --minimum-boundary-edge 0.005 `
  --boundary-collinearity-tolerance 0.0005 `
  --boundary-cleanup-passes 12 `
  --coast-size 0.030 `
  --offshore-size 0.080 `
  --refine-distance-min 0.060 `
  --refine-distance-max 0.35 `
  --minimum-angle 4.0 `
  --max-faces-per-vertex 10 `
  --max-aspect-ratio 50 `
  --mesh-attempts 8 `
  --compute-step-minutes 15 `
  --output-step-hours 6
```

This configuration:

- reduces mesh density close to the coast;
- retains a gradual coastal-to-offshore transition;
- preserves the SWAN vertex-connectivity restriction;
- keeps the temporary interpolation grid moderately detailed;
- avoids the cost of an unnecessarily fine shoreline mesh.

---

# 10. High-resolution coastal configuration

For studies where coastal geometry and shallow-water transformation are important:

```powershell
py .\generate_research_unstructured_pipeline_xyz_v2.py `
  --bathymetry-xyz ".\data\processed\CENTRO.xyz" `
  --xyz-value-mode positive-depth `
  --domain-bounds -37.0 -11.0 -34.5 -8.0 `
  --xyz-resolution 0.005 `
  --xyz-margin 0.20 `
  --simplify 0.0025 `
  --smooth 0.004 `
  --minimum-boundary-edge 0.0025 `
  --boundary-collinearity-tolerance 0.00025 `
  --coast-size 0.010 `
  --offshore-size 0.080 `
  --refine-distance-min 0.035 `
  --refine-distance-max 0.55 `
  --minimum-angle 4.0 `
  --max-faces-per-vertex 10 `
  --max-aspect-ratio 50 `
  --mesh-attempts 8
```

This configuration generates more coastline vertices and smaller coastal triangles and therefore usually requires more processing time.

---

# 11. Parameter summary

| Parameter | Type | Runtime default |
|---|---|---:|
| `--coast-level` | float | `0.0` |
| `--bathymetry-xyz` | path | configured script path |
| `--xyz-value-mode` | choice | `positive-depth` |
| `--xyz-resolution` | float | `0.01` |
| `--xyz-margin` | float | `0.20` |
| `--xyz-chunk-size` | integer | `500000` |
| `--xyz-max-points` | integer | `600000` |
| `--domain-bounds` | four floats | inferred from `wind.nc` |
| `--minimum-water-depth` | float | `0.5` |
| `--depth-repair-tolerance` | float | `2.0` |
| `--simplify` | float | `0.0025` |
| `--smooth` | float | `0.004` |
| `--topology-cleanup` | float | `0.0005` |
| `--topology-cleanup-attempts` | integer | `5` |
| `--minimum-hole-area` | float | `1e-6` |
| `--minimum-boundary-edge` | float | `0.0025` |
| `--boundary-collinearity-tolerance` | float | `0.00025` |
| `--boundary-cleanup-passes` | integer | `12` |
| `--coast-size` | float | `0.010` |
| `--offshore-size` | float | `0.080` |
| `--refine-distance-min` | float | `0.035` |
| `--refine-distance-max` | float | `0.55` |
| `--minimum-angle` | float | `4.0` |
| `--strict-minimum-angle` | flag | disabled |
| `--max-faces-per-vertex` | integer | `10` |
| `--max-aspect-ratio` | float | `50.0` |
| `--mesh-attempts` | integer | `8` |
| `--compute-step-minutes` | integer | `15` |
| `--output-step-hours` | float | `6.0` |
| `--run` | flag | disabled |
| `--docker-image` | string | configured default |
| `--swan-executable` | string | `/opt/swan/swan.exe` |

---

# 12. Important relationships

For normal configurations:

```text
coast-size < offshore-size
```

```text
refine-distance-min < refine-distance-max
```

```text
WEST < EAST
```

```text
SOUTH < NORTH
```

For SWAN 41.51A:

```text
max-faces-per-vertex <= 10
```

The following parameters affect different processing stages and should not be confused:

```text
xyz-resolution
```

controls the temporary regular interpolation grid.

```text
coast-size
offshore-size
```

control the final unstructured Gmsh mesh.

```text
simplify
smooth
minimum-boundary-edge
boundary-collinearity-tolerance
```

control coastline and boundary geometry.

```text
minimum-angle
max-faces-per-vertex
max-aspect-ratio
```

control mesh acceptance during QA.
