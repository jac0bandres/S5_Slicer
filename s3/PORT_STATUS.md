# S³ Python implementation: scope and fidelity

Requested scope: the full S³ pipeline, implemented in validated stages, with
**the paper's equations as the authority** and an eventual adapter for the
existing S5 printer. This is a separate implementation, not a change to S5.

Paper: Zhang et al., *S³-Slicer: A General Slicing Framework for Multi-Axis 3D
Printing*, ACM TOG 41(6), 277, 2022; repository-root `SIGAsia22S3Slicer.pdf`.
Comparison source: `S3_DeformFDM`, commit
`a77ef04115b1383a58f228a7f67ce4434c9d09f3`.

## Stage status

| Stage | Status | Validation required |
|---|---|---|
| Tetrahedral data and differential operators | Implemented | Affine gradients, topology, disconnected gauges, input checks |
| SF/SR/SQ spherical projection, Eq. 7 | Implemented | Analytic cases, intersection constraints, dense sphere cross-check |
| Pairwise quaternion blending, Eq. 8 | Implemented | Independently solved small weighted system |
| Rotation/scale/position deformation, Eq. 12 | Implemented | Explicit energy equality, stationary residual, large rigid rotation |
| Outer loop, Eq. 9 switches and Eq. 13 metric | Implemented | Cantilever integration and distinct stop reasons |
| Concavity weighting, Eqs. 5/10/11 | Implemented, limited validation | Valley/ridge sign test; broader geometry validation remains |
| Scalar transfer and fixed-count curved layers | Implemented | Height identity, planar intersections, welded vertices |
| Adaptive partial layers, §4.2 | Implemented in API/CLI/viewer; limited validation | Planar spacing, insertion/partial removal and hanging-edge repair tested; broad geometry/coverage validation remains |
| Contour/stress-directed surface toolpaths, §5.1 | Implemented for visual inspection | Surface confinement, holes, islands, waypoint spacing and constant/sign-flipped stress tested; general coverage and stress-fit accuracy remain |
| Deformation/layer/toolpath visualization | Implemented | Streamlit workflow/export tests and Chrome rendering of offline HTML preview |
| S5 printer motion/G-code adapter | Geometric pose adapter implemented | Pose round trips and unreachable orientations tested; motion transitions, extrusion and collisions pending |

“Implemented” does not establish replication of the paper's experimental results.
Do not call this the completed S³ pipeline or a print-ready slicer yet.

## Paper versus C++ differences

The default `python -m s3` path calls `pipeline.run_paper` and `paper.py`.
The optional `--implementation cpp-reference` path calls `deform.py` and the
separate C++-convention kernels in `numerics.py` and `heat.py`.

| Detail | Paper path | Active C++ support-free path |
|---|---|---|
| Build direction | Z-up | Y-up |
| Quaternion energy | Pairwise differences, Eq. 8 | Squared normalized-Laplacian residual plus anchors |
| Weight interpretation | Energy coefficients; square roots in residuals | Supplied values directly multiply residuals |
| Scale order | `R @ diag(s) @ P`, Eq. 12 | World-axis scaling of `R @ P` |
| Position solve | Coupled positions and local scales | Three independent world-axis systems |
| Target direction | Nearest point in spherical feasible region | Heat-guided two-candidate projection |
| Stopping | Eq. 13 relative change, with explicit zero/cap handling | Fixed GUI-selected iteration counts |
| Height transfer | Raw deformed height on original connectivity | Additional normalized-vector smoothing and scalar reconstruction |
| Input precision | Float64 | Optional emulation of float32 parsing |

The official GUI's SF entry point is `runASAP_SupportLess_test3`, not the older
`runASAP_SupportLess` function. It calls `_globalQuaternionSmooth1_supportLess`.
The apparent 20-step smoothing loop in `_cal_heatMethod` changes a display field,
not the growing field consumed by deformation. `QMeshTetra::CalVolume` returns
six times geometric volume; this affects the heat mass matrix and stored
gradient coefficients. These conventions are retained only in the comparison
path. Its successful execution is not evidence that the paper equations behave
identically.

## Choices the paper does not completely prescribe

- Quaternion signs are lifted consistently across the dual graph and target
  signs aligned to current rotations before linear blending; output is normalized.
- With `fix_build_plate=false`, translation gauges are pinned per connected vertex component. Unconstrained
  quaternion components preserve a root orientation. These remove nullspaces,
  not manufacturing freedom in constrained components.
- Projections onto the spherical halfspace intersection enumerate zero-, one-,
  and two-active-constraint candidates. SQ includes its isolated pole candidates.
- If multiple boundary faces of a cell are selected, their constraints are
  intersected. This is stricter than selecting one representative face; the
  paper's single-face feasibility argument does not cover arbitrary such cells.
  Infeasible selections raise an error naming the cell. Explicit face selections
  can reproduce a prescribed experimental setup.
- SF excludes faces exactly on the build plate by default. This is an explicit
  boundary-condition choice, configurable with `exclude_build_plate`.
- SR takes externally supplied principal-stress directions and a critical-region
  mask. It does not fabricate an FEA result. SQ defaults to all boundary faces,
  or accepts an explicit face selection.
- Seven inner iterations are the default; a 20-step outer cap prevents endless
  runs. Relative stagnation is **not** reported as objective satisfaction.
- `final_conformal_weight=10` applies to SQ-C in the final inner step (§3.3.1).
- Isosurface triangulation chooses the shorter quad diagonal. Near-level vertex
  values receive an epsilon perturbation; ordering of vertices/faces is not a
  replication target. Fixed-count layers do not enforce physical thickness.
- Degenerate geometric inputs/directions are rejected instead of propagating NaNs.

These choices must accompany any claim of fidelity. “Paper-based” is more
accurate than “exact reproduction” until the remaining algorithmic choices and
the full experimental setup have been validated.

## Validation evidence

`tests/test_paper.py` checks the written equations independently of the sparse
assembly. `tests/test_numerics.py` checks the C++-convention port and shared mesh
operations. The cantilever fixture has 24 vertices and 30 tetrahedra.

`validation/paper_cantilever.json` records an SF run: initial Π = 4.18879,
final Π = 0.690978, minimum determinant = 0.905962, no inverted cells.
It stops after three outer passes by relative stagnation, leaving roughly 5.04°
worst-case angular violation. This is a diagnostic, not a manufacturing success.

`reference/build_reference.py` extracts actual active functions from the local
C++ checkout and builds a headless comparison executable. Its documented
modifications are removal of GUI dependencies, an Eigen sparse-backend swap,
and a translation gauge. It does not modify the upstream checkout.
The comparison on the Y-up cantilever matches heat within 4e-15, growing
directions within 4e-17, scales within 5e-10, and centered positions within 6e-9.
See `validation/cpp_cantilever.json`. This is **not** an unmodified full-GUI
binary comparison, nor validation of every C++ branch.

`validation/paper_ring_smoke.json` records a two-pass run on the official Ring
mesh (8,150 vertices, 40,064 tetrahedra). Π decreases from approximately 432.38
to 49.29; no inverted cells were observed, but worst angular violation remains
40.52°. The run stops at its configured two-pass cap and is not a converged
reproduction of the paper. It took approximately 249 seconds on this environment.
This run used the comparison reader's float32 import convention before the
paper reader was changed to float64; the precision difference is recorded in the
artifact. A full-resolution, converged experiment remains a separate validation
milestone. None of these numbers establish an improvement over S³.

## Printer scope

The requested final target is the existing S5 printer. Its default Core-R-Theta
writer has X/Z/B/C axes and a radial-plane nozzle-orientation restriction; S³'s
general 3D directions need a reachability check. The final adapter must expose
unreachable directions rather than silently treating clipping as a valid pose.
S³'s paper separates geometric slicing from robot motion planning; adapting the
latter to this printer is an explicitly machine-specific extension.

`machine.S5Machine` implements the existing writer's nominal 41.5 mm nozzle
offset and B range −130° to +30°. It maps tip positions and printing directions
to X/Z/B/C poses, unwraps C near the preceding yaw, and rejects unreachable
directions. These are software conventions copied from S5, not measured hardware
calibration. It does not yet emit G-code or check travel limits and collisions.

`adaptive.adaptive_layers` is an experimental Python API. Triangle-to-surface
distance bounds drive initial spacing, midpoint insertion and conservative
subdivision trimming. Reports include unresolved distance bounds, removed mixed
pieces, post-trim spacing checks and the remaining scalar cap. Coverage is
explicitly uncertified: passing inter-layer spacing alone does not establish
complete deposition. API, CLI, and viewer now expose this stage. Trimming retains
source tetrahedron IDs and repairs hanging edges before toolpath generation.
Distance refinement can stop when conservative bounds decide a spacing threshold;
reports distinguish this decision from precisely resolved distance intervals.
Triangle-pair minimum distances (including edge/face intersections) avoid slow
subdivision near the minimum-thickness threshold. Maximum distances retain
adaptive conservative bounds.

## Visual toolpath milestone (2026-09-15)

The paper's §5.1 prescribes contour paths from a curved layer's boundary-distance
field, and stress-projected directional paths in the interior of critical layers.
`toolpaths.py` implements that construction using surface FEM heat diffusion,
Poisson distance reconstruction, and a least-squares scalar fit to the tangent
stress line field. Line-field signs are lifted across face adjacency. Scalar
isocurves are marched and joined through actual shared mesh edges. Boundary
contours start at half a path spacing; hybrid interiors are clipped beyond the
requested boundary-band width. Layers without critical cells use contours.

This is a paper-guided implementation. It is not an exact reproduction of the
upstream normalized heat field, external remeshing, path-count heuristics, or
nearest-point linking. We refine existing facets using conforming edge splits,
retain all triangle crossings when sampling, and export separate deposition
strokes with area-weighted interpolated unit normals. We do not invent travel
segments between disconnected curves. Source tetrahedron IDs transfer stress to
surface triangles exactly, including after adaptive trimming/refinement.

Remaining numerical qualifications: heat distances depend on mesh quality; equal
potential increments only approximate physical spacing; stress-field cycles and
singularities can leave integration/alignment errors. Reports include negative
distance vertices before clipping, stress alignment/fit residuals, components too
narrow for the first contour offset, and empty layers. No complete bead-coverage
or collision guarantee is claimed. Closed layer components with no boundary and
critical regions with no projected tangent stress direction are explicitly rejected
when a stress interior is needed.

The workbench supports original/final deformation views and interpolation,
individual or accumulated toolpaths, surface-normal arrows, adaptive/fixed layer
selection, stale-result notices, and ZIP download. The CLI produces the same
OBJ/JSON paths and a self-contained interactive HTML preview. G-code and machine
motion are outside this milestone at the user's request.

Validation includes a 12-layer cantilever example at 0.8 mm path spacing:
107 closed strokes, 14,023 waypoints, and no empty layers. The underlying
deformation retains the previously documented ~5.04° angular violation. This
establishes an inspectable end-to-end example, not a reproduction of the paper's
manufacturing results.

`validation/visual_cantilever.json` records that fixed-count run.
`validation/visual_adaptive_cantilever.json` records the same deformation scaled
uniformly by 0.1, then sliced at 0.16–0.4 mm thickness and 0.2 mm path spacing:
18 layers, passing inter-layer spacing bounds, and 61 strokes. The last layer
has no contour at that spacing and is explicitly reported. This is also why a
passing layer-spacing report must not be interpreted as full material coverage.

Numerical toolpaths follow the construction
in §5.1 and the surface heat method described by
[Crane et al.](https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/).

## Build-start correction (2026-09-15)

Visual inspection by the user exposed a missing build-boundary condition in the
earlier cantilever example. Its best-fit rigid rotation was about 53.14°, with
only 1.04 mm RMS nonrigid residual. Original base vertices occupied scalar heights
0.053–10.40 mm. The former translation gauge and SF base-face exclusion did not
preserve the contact surface. Those archived runs are unconstrained numerical
examples, not valid builds from the original foot.

`fix_build_plate=true` is now the default. It pins all vertices of flat boundary
faces at the minimum input height, anchors base-contact tetrahedron rotations to
identity, and resolves equal-angle spherical projection choices using a mesh-edge
distance-to-base gradient. The guide breaks ties; it does not replace the nearest
projection objective. Candidate position steps are backtracked if they invert
cells, place material below the starting plane, or introduce off-base minima of
the piecewise-linear scalar field. Equal-value plateaus are handled together.
The original plate is normalized to zero without shifting later failures upward.

These are explicit build-boundary and feasibility additions to the paper-based
solver, not a claimed reproduction of undocumented upstream behavior. Disable
`fix_build_plate` for the historical unconstrained equations/comparisons. Inputs
with no flat contact face or disconnected elevated starting regions are rejected
in fixed-base mode. A missing feasible step is reported as `build_feasibility_limit`.

The corrected cantilever keeps all four base vertices exactly fixed, has no
below-plate vertices or off-base scalar minima, and has minimum determinant
0.6753. It still stops by relative stagnation with approximately 7.87° worst
overhang violation. Passing the build-start tests is necessary but does not
establish deposition support, coverage, or collision freedom. The regression
suite covers fixed-base deformation, translated input bases, floating plateaus,
guided projection ties, and feasibility backtracking.

The full suite passes 54 tests. Corrected fixed-count and adaptive results are
recorded in `validation/visual_cantilever_fixed_base.json` and
`validation/visual_cantilever_fixed_adaptive.json`. The adaptive run uses
0.16–0.4 mm layer thickness with 0.8 mm diagnostic path spacing; its 244 layers
replace the sparse 12-layer view when inspecting the start of the build.

## Attribution

The C++ comparison path is derived from zhangty019's BSD-3-Clause source; its
notice is retained in `LICENSE.upstream`. The surrounding repository retains
its existing license. Python implementation and fidelity notes by GPT Astra,
2026-09-12, at the project owner's request.
