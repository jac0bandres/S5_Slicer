# S5 Optimization & Fix Log

Running record of correctness fixes and quality improvements to the S5 nonplanar
slicer (`S5.py`). Each entry: what, why, evidence, status.

---

## Legend
- ✅ done & verified
- 🚧 in progress
- 📋 planned / proposed
- 🔬 finding (evidence we act on)

---

## 1. TetGen mesh repair for self-intersections ✅

**Problem.** TetGen aborts on raw CAD/STL surfaces that are watertight but
self-intersecting (`The input surface mesh contain self-intersections.`) or have
near-degenerate facets (`recoversubfaces`). `tree.stl` reproduces it.

**Fix.** Two-stage `_repair` in `slice()`:
1. trimesh topological cleanup — weld coincident verts, drop degenerate &
   duplicate faces, prune unreferenced verts, fix winding.
2. pymeshfix `MeshFix.repair(joincomp=True, remove_smallest_components=False)` —
   resolves self-intersections, keeps **all** components (so multi-part models
   like a tree's branches aren't gutted — pymeshfix's default drops all but the
   largest component).

Kept the tiered fallback (repair → retry → plain CDT without quality switches).

**Side-effect cleanup.** TetGen drops `_skipped.face` / `_skipped.node` into the
cwd on *every* self-intersection failure. Added `finally: _cleanup_tetgen_scratch()`,
removed the stray files, and added them to `.gitignore`.

**Evidence.** `tree.stl`: repair → 15938 faces → TetGen **182281 cells**. Clean
models (Squirtle, pi, dino, benchy) skip repair unchanged.

---

## 2. Cura configuration for Debian ✅

**Problem.** Defaults were Windows-only and the bundled defs were version-mismatched:
- `--cura` pointed at `C:/Program Files/UltiMaker Cura 5.11.0/CuraEngine.exe`.
- Bundled `config/fdmprinter.def.json` is a Cura **5.11** export; the Debian
  `cura-engine` package is **5.0.0** → `[ERROR] Trying to retrieve setting with
  no value given: 'wireframe_enabled'`.
- `-j core.def.json` was passed as a *definition* file, but it's a GUI settings
  export (already applied via `-s`), which aborts the 5.0.0 engine.

**Fix.**
- `--cura` default → `/usr/bin/CuraEngine`.
- `--printer_path` / `--extruder_path` default to engine-matched system defs in
  `/usr/share/cura/resources/definitions/`, falling back to bundled copies.
- Removed the broken `-j abs_custom` from the slice command (+ dead `abs_custom`).

**Evidence.** `dino.stl` slices end-to-end: return code 0, 5.8 MB G-code, no errors.

---

## 3. Deformation solver: iterative lsqr → direct sparse solve ✅

**Finding (🔬).** `calculate_deformation` (S5.pdf §4.5) uses `scipy.sparse.linalg.lsqr`
with `iter_lim=1000`. On `tree.stl` (182k cells) it is badly non-converged.
Same operator (the production `N` centering), only the solver changes:

| Solver (N-operator, tree, 182k cells) | p99 \|σ−1\| | max \|σ−1\| | inverted tets |
|---|---|---|---|
| lsqr, iter_lim=1000 (current) | 0.878 | **92.3** | **206** |
| direct sparse solve | 0.051 | **2.87** | **0** |

(`σ` = singular values of the per-tet deformation gradient `F_c`; `|σ−1|` measures
shear+scale away from a pure rotation. Inverted = negative-volume tets.)

The system is SPD once one vertex is pinned (kills the translation null space), so
a direct solve is exact, faster, and eliminates all inversions. This is the likely
source of: trunk twisting (`plots/06b`), volume-scaling blowups (`plots/07`), and
the existence of the `MAX_EXTRUSION_MULTIPLIER = 10` clamp (which papers over the
exploded cells). Note: S5's rotation-smoothing stage (§4.4) already uses `spsolve`
correctly — the deformation stage just never got the same treatment, despite the
paper's stated thesis.

**Done.**
- Replaced the 3× lsqr solves with a direct solve of the normal equations
  (`AᵀA`, SPD after pinning the lowest-z vertex); one `splu` factorisation reused
  across x/y/z. Added a permanent inversion diagnostic (`inverted tets=…,
  vol-ratio min/median/max=…`) printed each run.
- Full `tree.stl` pipeline runs end-to-end, exit 0, **0 lost vertices**,
  deformation solve ~1 s.

**Real-field result (the important caveat).** On the synthetic 35°-max field the
direct solve gave 0 inversions. On the **real** overhang-driven field (per-cell
rotations up to **2.85 rad ≈ 163°**) both solvers still invert cells, but direct
is clearly better:

| Solver (real tree field) | inverted tets | vol-ratio min / max |
|---|---|---|
| lsqr, iter_lim=1000 (old) | 2470 | −226 / +105 |
| direct solve (new) | **620** | −77 / +255 |

So the solver swap is a strict ~4× win (and exact + faster + deterministic), **but
it does not eliminate inversions** — at these rotation magnitudes the objective
itself (large, spatially-incompatible per-cell rotations realized by one linear
solve) is inversion-prone. The solver was masking part of the problem; it is not
the whole problem. → see finding 6.

---

## 4. Cotan / FEM Laplacian for deformation 🔬 (marginal — deprioritized)

**Finding.** Apples-to-apples (both solved **directly**), volume/cotan gradient
weighting barely differs from uniform weighting:

| Operator (direct solve, tree) | vol-wtd \|σ−1\| | p99 \|σ−1\| | max \|σ−1\| |
|---|---|---|---|
| uniform gradient | 0.0160 | 0.032 | 0.239 |
| cotan/FEM (volume-weighted) | 0.0153 | 0.046 | 0.172 |

The operator weighting is **not** the bottleneck — the solver (finding 3) is. Cotan
is the "more correct" FEM operator and marginally helps the worst case, so adopt it
as a cheap refinement *on top of* the direct solve, but it is not the win.

---

## 5. `--neighbor-loss-weight` (W) feels inert 🔬

**Finding.** Not a no-op, but effectively invisible in normal ranges:
- It genuinely changes rotation-field smoothness — std 6.1° at W=30 → 1.1° at W=1000
  (measured on dino's actual dual graph + anchored surface field).
- But smoothing length scales ~√W, so 30→60 is imperceptible; you need
  30→300→3000 to see a difference.
- The deformation **integrates** the rotation field, averaging out exactly the
  high-frequency variation W controls. Gross deformed shape is set by the anchored
  *surface* targets, which W smooths but never removes.

**Action.** Leave the solve as-is (it's already a clean direct `spsolve`). ✅ Fixed
the `--neighbor-loss-weight` help text (was copy-pasted from `--rotation-multiplier`).

---

## 7. TetGen segfaults on lattice/frame meshes → subprocess + fTetWild ✅

**Problem.** `synthara-frame.stl` (352k-face genus-95 lattice) **segfaulted** the
whole pipeline. Root cause: genuine geometric self-intersections at strut joints
that pymeshfix cannot resolve (idempotent), and which crash TetGen's boundary
recovery — an uncatchable hard crash (`SIGSEGV`), not a `RuntimeError`. Decimation
"fixed" it but throws away geometry (rejected). manifold3d only validates
*topology* (reports the mesh as valid) and breaks watertightness when forced to
re-arrange.

**Fix (two parts).**
1. **Subprocess isolation.** Tetrahedralization runs in a forked child
   (`_mesh_worker` + `_run`). A segfault now kills only the child; the parent
   sees `exitcode -11`, raises a catchable `RuntimeError`, and falls through to
   the next tier. A bad mesh can never crash the whole slicer again.
2. **FloatTetWild fallback** (`wildmeshing==0.4.1`, new dep). fTetWild is
   envelope-based and robust to self-intersecting / non-manifold input by design
   (does its own winding-number inside/outside). Added as the robust catch-all
   tier. Native C++ logging silenced in the worker (fd redirect).

**Tier ladder** (fast path first, geometry-preserving throughout — no decimation):
`tetgen as-is → tetgen repaired → fTetWild (robust)`.

**Evidence.**
- `synthara-frame.stl`: tetgen as-is → self-int; tetgen repaired → segfault
  (caught); **fTetWild → success**; full pipeline runs end-to-end, 0 lost
  vertices, G-code produced (~446 s total, ~105 s of it fTetWild meshing).
  Deformation: only 35–41 inverted tets (coarser fTetWild mesh deforms cleanly).
- Regression `pi 3mm.stl`: no fallback tier (fast tetgen path), 0 inverted tets,
  24 s — clean models are unaffected.
- Also: `slice()` now `os.makedirs(output_path, exist_ok=True)` so a missing
  `-o` dir no longer crashes at save time.

**Follow-up: floodfill → winding number.** With `--reorient 270,0,0` the frame
crashed in neighbor-finding (`cell_point_neighbours` shape `(0,2)`). Cause:
`get_tet_mesh(floodfill=True)` collapsed the reoriented genus-95 lattice to **2
tets** — flood-fill inside/outside seeds from outside and leaks through the
lattice holes (and is non-deterministic). Switched to `floodfill=False`
(winding-number extraction, robust to topology): 31082 tets, slices end-to-end.
Added a degenerate-output guard in `_run` (reject `n_cells<4` or `n_points >
3·n_cells`, i.e. a disconnected "soup") so any future bad mesh becomes a clean
tier failure instead of a confusing downstream crash. Also hardened: a missing
`-o` output dir is now created (`os.makedirs(..., exist_ok=True)`).

**Open tuning.** fTetWild `edge_length_r=0.05` is coarse-ish; the deformed STL
that Cura slices is therefore lower-res than the input. Fine for the deformation
*field*, but for surface fidelity on fallback meshes consider a smaller
`edge_length_r` (more tets, slower). Only affects meshes that fall through to
fTetWild; tetgen meshes keep full resolution.

---

## 6. Rotation-field magnitude is physically unbounded 📋 (next lever)

**Finding.** The deformation is driven by rotations up to **163°** per cell, yet
the machine B-axis is clamped to `[-130°, +30°]` (`MIN_ROTATION`/`MAX_ROTATION`)
in the G-code stage. The smoothed rotation field is only clipped to **±360°**
(`MAX_POS_ROTATION`/`MAX_NEG_ROTATION = ±deg2rad(360)`), i.e. effectively
unbounded. Tilting a tet by >90° is what produces most of the 620 remaining
inversions, and those tilts aren't even reachable by the hardware.

**Hypotheses to try (in order of expected payoff / effort):**
1. **Clamp the deformation rotation field to the machine range** (or a sane
   ±60–90°) via `MAX_POS_ROTATION`/`MAX_NEG_ROTATION`. Should both match physics
   and sharply cut inversions. Cheapest test.
2. **ARAP iteration** — re-fit each cell's rotation to the current deformation so
   the field becomes compatible/integrable, instead of forcing the raw overhang
   target.
3. **Locally-injective / inversion-barrier solve** or the volumetric
   inside/outside graph of Zhou et al. [7] — strongest, most work.

**Metric:** inverted-tet count + vol-ratio range from the deformation diagnostic.
Baseline to beat: **620 inverted** (direct solve, ±360° clamp).

---

## Test assets
- `tree.stl` — self-intersecting; the workhorse for repair + deformation tests.
- `dino.stl` — small/fast (9k cells); good for solver-sensitivity sweeps.
- Distortion metric: per-tet `F_c = G_c X` via `igl.grad`; report `|σ−1|`
  percentiles + inverted-tet count.
</content>
</invoke>
