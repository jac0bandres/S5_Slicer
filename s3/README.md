# S³ in Python

Paper-based S³ deformation with SF/SR/SQ constraints, scale-controlled geometry,
and scalar isosurfaces. This is being developed in validated stages;
[PORT_STATUS.md](PORT_STATUS.md) defines what is implemented and what remains.
The visual pipeline includes fixed-count or adaptive layers, boundary-contour
toolpaths, and stress-directed hybrid toolpaths. Full paper-result replication,
coverage validation, and printer motion remain unfinished.

Uses NumPy and SciPy; the existing `S5.py` is not imported or modified.

```bash
python3 -m venv .venv-s3
.venv-s3/bin/python -m pip install -r s3/requirements.txt
.venv-s3/bin/python -m pytest s3/tests -q

.venv-s3/bin/python -m s3 s3/tests/fixtures/cantilever.tet \
    -o output/s3_cantilever --layers 12
```

## Visual workbench

```bash
.venv-s3/bin/python -m pip install -r s3/requirements-ui.txt
.venv-s3/bin/python -m streamlit run s3/streamlit_app.py
```

1. Choose a tetrahedral mesh and run **Run paper pipeline**.
   **Fix build-plate contact surface** is enabled by default: place the intended
   flat base at the input's minimum Z (or choose Y-up before conversion).
2. Inspect **Geometry**, including the original-to-deformed interpolation slider.
3. Choose **Fixed count** or **Adaptive thickness** in **Curved layers**.
4. In **Toolpaths**, set spacing and waypoint distance, then click
   **Generate layers and toolpaths**. Inspect one layer, accumulated paths, or
   all layers, with optional surface-normal arrows.
5. **Prepare ZIP** in **Download** exports an offline `preview.html`, layer
   meshes, toolpath OBJ polylines, and JSON waypoints with unit normals.

Orange paths are boundary-distance contours; pink paths are stress-directed
interiors. Curves remain separate deposition strokes, with no implicit travel
connections. The HTML preview includes deformation and cumulative-layer sliders
and embeds Plotly, so it works without an internet connection.

**Stress hybrid** requires the per-tetrahedron `stress` and `stress_mask` arrays
attached to the deformation run. Layers crossing a critical region use boundary
contours plus interior paths from the projected stress field. Other layers use
contours throughout. Editing input controls does not change a saved run; the UI
marks stale deformation and path settings explicitly.

Equivalent CLI example:

```bash
.venv-s3/bin/python -m s3 s3/tests/fixtures/cantilever.tet \
    -o output/s3_visual --layers 12 --toolpaths --path-spacing .8 --preview-html
```

Use `--layer-mode adaptive --min-thickness .16 --max-thickness .4` for adaptive
layers instead of a fixed count. Use `--path-mode hybrid --objectives fields.npz`
for stress-directed interiors. `--waypoint-distance` sets the maximum segment
length; triangle crossings are retained to keep chords on the curved surface.
HTML export requires the UI dependencies; numerical path generation does not.

The distance field uses surface FEM heat diffusion and a Poisson reconstruction.
Layers are refined in their existing facets to resolve the requested path spacing.
This is a paper-guided implementation, not a bit-for-bit port of the upstream
remeshing/path-linking routines. Equal field increments approximate physical path
spacing; diagnostics expose stress fit error, empty layers, and adaptive spacing
bounds. No claim of complete deposition coverage is made.

The default deformation now preserves the base contact vertices and the rotations
of base-contact tetrahedra. Equal-angle projection choices are guided by distance
from that base; a feasibility backtrack prevents inversion, below-plate material,
and off-base scalar minima. These are explicit build-boundary additions to the
paper equations. Set `fix_build_plate: false` in a JSON config for the earlier
unconstrained numerical baseline. That baseline can mostly rotate a part and is
not a valid build sequence from the original foot. Saved runs need to be rerun
after changing this setting. Models with no flat contact face need an explicit
base setup; disabling the constraint only enables an unconstrained experiment.

## Numerical pipeline

An environment has already been created locally as `.venv-s3`. On minimal
Debian installations, creating another environment may require `python3-venv`.

Official `.tet` datasets are Y-up; request the coordinate conversion explicitly:

```bash
OPENBLAS_NUM_THREADS=1 .venv-s3/bin/python -m s3 \
    ../S3_DeformFDM/DataSet/TET_MODEL/ring.tet --up-axis y \
    -o output/s3_ring --layers 40
```

Output directories must be empty. Outputs include:

- `result.npz`: original/deformed vertices, tetrahedra, rotations, scales and scalar field.
- `deformed.tet`: deformed geometry for inspection.
- `report.json`: input hash, parameters, versions, per-iteration metrics and stopping reason.
- `layers/*.obj`: optional fixed-count isosurfaces in the original shape.
- `toolpaths/*.obj` and `toolpaths/*.json`: optional deposition strokes, with
  point positions and normals in the JSON files.
- `visualization.json`: layer-spacing and toolpath diagnostics.
- `preview.html`: optional self-contained interactive deformation/path viewer.

The paper mode uses raw deformed Z as scalar values. Layer count does not imply
bounded layer thickness, valid toolpaths, or reachable machine poses.

Configuration overrides use `--config config.json`; fields are defined in
`pipeline.PaperConfig`. For example:

```json
{
  "weight_sf": 0.7,
  "weight_sr": 0.3,
  "weight_sq": 0.0,
  "alpha": 30.0,
  "beta": 10.0,
  "rigidity": 1.0,
  "scale_compatibility": 6.0,
  "inner_iterations": 7,
  "max_outer_iterations": 20
}
```

SR needs `--objectives fields.npz` with `stress` (tet count × 3 principal
directions) and `stress_mask` (tet count booleans selecting critical cells).
Optional `sf_faces` and `sq_faces` are arrays of global face indices in
`TetMesh.faces`. Vectors use the input coordinate frame; the CLI transforms
stress vectors with `--up-axis`. The Python API accepts the same data:

```python
from s3 import read_tet, PaperConfig, run_paper
mesh = read_tet("model.tet")  # Z-up for the paper API
result = run_paper(mesh, PaperConfig(), callback=print)
```

For a source-behavior comparison (not the paper model):

```bash
.venv-s3/bin/python -m s3 ../S3_DeformFDM/DataSet/TET_MODEL/ring.tet \
    --implementation cpp-reference --up-axis y -o output/s3_cpp_port

python3 s3/reference/build_reference.py ../S3_DeformFDM /tmp/s3-reference-build
/tmp/s3-reference-build/s3-reference model_y_up.tet /tmp/native_result
.venv-s3/bin/python -m s3.reference.compare_reference \
    /tmp/s3-reference-build/s3-reference
```

The reference builder requires `g++` and the upstream checkout's vendored Eigen.
It writes only into its build directory and keeps the upstream files intact.

The final machine target is the existing S5 Core-R-Theta printer. The geometric
adapter is available independently; printer motion planning is deferred:

```python
from s3.machine import S5Machine
machine = S5Machine()
pose = machine.inverse([50, 0, 10], [0, 0, 1])  # X, Z, B, C
```

Unreachable orientations raise `ValueError`; angles are never clipped. This API
does not certify collision clearance or generate printer-ready motion.
Adaptive surfaces are available through `s3.adaptive.adaptive_layers`, the CLI,
and the workbench. Consult their reports and the fidelity ledger before
interpreting them as complete layer coverage.
