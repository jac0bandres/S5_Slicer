"""synthetic_test.py — verify s5_bench + s4_reference work on your machine.

Run this once to confirm the package is set up correctly. No mesh files, no
S5 pipeline, no pyvista — just numpy + scipy. Takes <1 second.

    python synthetic_test.py

Expected output ends with "All checks passed." and writes a small bench_results/
directory you can inspect (or delete). If anything fails, the error points at
which piece needs attention before you patch S5.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from s5_bench import bench, BenchConfig
import s4_reference as s4


def main():
    # Configure — same style as VizConfig at the top of S5.py
    BenchConfig.enabled = True
    BenchConfig.output_dir = "./bench_results"
    BenchConfig.run_tag = "synthetic_smoke_test"
    BenchConfig.run_s4_comparison = True
    BenchConfig.verbose = True

    # Build a toy problem: 100-cell 1D chain for smoothing, then a tiny cube
    # for deformation-family stages.
    rng = np.random.default_rng(0)

    # -------- smoothing (1D chain) ---------------------------------------
    n = 100
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    init = np.full(n, np.nan)
    init[0], init[-1] = 0.0, 1.0
    W = 30.0

    bench.set_global_context(test="chain", n=n)

    with bench.stage("smoothing"):
        # This is where S5.py's inline spsolve would normally live. We replicate
        # it here so we can verify bench's stage() wraps it correctly.
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        valid = ~np.isnan(init)
        ea, eb = edges[:, 0], edges[:, 1]
        deg = np.zeros(n); np.add.at(deg, ea, 1.0); np.add.at(deg, eb, 1.0)
        diag = W * deg + valid.astype(float) + 1e-10
        rows = np.concatenate([ea, eb, np.arange(n, dtype=np.int32)])
        cols = np.concatenate([eb, ea, np.arange(n, dtype=np.int32)])
        data = np.concatenate([-W * np.ones(len(ea)), -W * np.ones(len(eb)), diag])
        A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        b = np.where(valid, np.nan_to_num(init, nan=0.0), 0.0)
        s5_rot = spsolve(A, b)

    bench.reference(
        "smoothing",
        s4=lambda: s4.smooth_rotation_field(n, edges, init, W,
                                            max_nfev=2000, ftol=1e-12, xtol=1e-12),
        s5_result=s5_rot,
        correctness=bench.compare_scalar_fields,
    )

    # -------- deformation (cube 6-tet) -----------------------------------
    # Cube offset so cell centers have nontrivial radial direction.
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=np.float64) + np.array([5.0, 3.0, 0.0])
    cells = np.array([
        [0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7],
        [0, 4, 6, 7], [0, 2, 6, 7], [0, 2, 3, 7],
    ], dtype=np.int32)
    cell_centers = points[cells].mean(axis=1)
    rotation_field = rng.uniform(-0.05, 0.05, size=len(cells))

    bench.set_global_context(test="cube", n_cells=len(cells), n_points=len(points))

    # Simulate S5 deformation inline (parallel LSQR, 3 threads)
    with bench.stage("deformation"):
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from concurrent.futures import ThreadPoolExecutor

        rot_mats = s4._rotation_matrices_from_field(cell_centers, rotation_field)
        _N4 = np.eye(4) - 0.25 * np.ones((4, 4))
        old_verts = points[cells]
        old_t = np.einsum("cij,cjk->cik", rot_mats, (_N4 @ old_verts).transpose(0, 2, 1))
        n_c, n_p = len(cells), len(points)
        c_idx = np.arange(n_c)
        rs, cs, vs = [], [], []
        for i in range(4):
            for k in range(4):
                rs.append(c_idx * 4 + i); cs.append(cells[:, k])
                vs.append(np.full(n_c, 0.75 if i == k else -0.25))
        A = sparse.coo_matrix((np.concatenate(vs),
                               (np.concatenate(rs), np.concatenate(cs))),
                              shape=(n_c * 4, n_p)).tocsr()
        B = np.ascontiguousarray(old_t.transpose(0, 2, 1).reshape(n_c * 4, 3))
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(
                lambda j: lsqr(A, B[:, j], x0=points[:, j],
                               atol=1e-10, btol=1e-10, iter_lim=500)[0],
                range(3),
            ))
        s5_def = np.stack(results, axis=1)

    bench.reference(
        "deformation",
        s4=lambda: s4.solve_deformation(points, cells, cell_centers,
                                        rotation_field, max_nfev=5000,
                                        ftol=1e-12, xtol=1e-12),
        s5_result=s5_def,
        correctness=bench.compare_vector_fields,
    )

    # -------- barycentric (exact interior points) ------------------------
    n_pts = 200
    cell_idx = rng.integers(0, len(cells), size=n_pts)
    cell_verts_batch = points[cells[cell_idx]]
    w = rng.dirichlet(np.ones(4), size=n_pts)
    query = np.einsum("nk,nkj->nj", w, cell_verts_batch)

    bench.set_global_context(test="bary", n_pts=n_pts)

    with bench.stage("barycentric"):
        tet_d = cell_verts_batch[:, 3]
        T = (cell_verts_batch[:, :3] - cell_verts_batch[:, 3:4]).transpose(0, 2, 1)
        rhs = query - tet_d
        lam123 = np.linalg.solve(T, rhs[:, :, None]).squeeze(-1)
        s5_bary = np.empty((n_pts, 4))
        s5_bary[:, :3] = lam123
        s5_bary[:, 3] = 1.0 - lam123.sum(axis=1)

    bench.reference(
        "barycentric",
        s4=lambda: s4.compute_barycentric_point_transforms(cell_verts_batch, query),
        s5_result=s5_bary,
        correctness=bench.compare_vector_fields,
    )

    # -------- validity mask ---------------------------------------------
    positions = rng.standard_normal((1000, 3))
    nan_rows = rng.choice(1000, size=300, replace=False)
    positions[nan_rows, rng.integers(0, 3, size=300)] = np.nan

    bench.set_global_context(test="validity", n=len(positions))

    with bench.stage("validity"):
        s5_valid = ~np.any(np.isnan(positions), axis=1)

    bench.reference(
        "validity",
        s4=lambda: s4.check_validity_loop(positions),
        s5_result=s5_valid,
        correctness=bench.compare_bool_arrays,
    )

    # -------- write out and inspect -------------------------------------
    bench.finish()

    # Sanity assertions
    out = Path(BenchConfig.output_dir) / BenchConfig.run_tag
    for f in ["timings.csv", "correctness.csv", "env.json", "meta.json"]:
        assert (out / f).exists(), f"missing output file: {out / f}"
    print(f"\nWrote {out.resolve()}")

    # Read back and sanity-check the timings CSV
    import csv
    rows = list(csv.DictReader((out / "timings.csv").open()))
    n_s5 = sum(1 for r in rows if r["impl"] == "s5")
    n_s4 = sum(1 for r in rows if r["impl"] == "s4")
    print(f"Recorded {n_s5} S5 rows and {n_s4} S4 rows.")
    assert n_s5 >= 4 and n_s4 >= 4, "expected at least 4 of each"

    # Correctness CSV
    corr_rows = list(csv.DictReader((out / "correctness.csv").open()))
    print(f"Recorded {len(corr_rows)} correctness rows.")
    # Thresholds are stage-specific — some of these are exact refactors (1e-6
    # precision), others are algorithm substitutions or TRF-vs-direct-solver
    # pairs where S4 is known to leave a ~20–40% residual gap on rank-deficient
    # problems. We encode these expected tolerances explicitly.
    expected = {
        "smoothing":   ("max_abs", 1e-3),
        "deformation": ("cosine_mean_at_least", 0.99),
        "barycentric": ("max_abs", 1e-10),
        "validity":    ("exact_match", True),
    }
    for r in corr_rows:
        stage = r["stage"]
        if stage not in expected:
            continue
        check, bound = expected[stage]
        if check == "max_abs":
            val = float(r.get("max_abs") or 0)
            assert val < bound, f"{stage}: max_abs = {val} (bound {bound})"
        elif check == "cosine_mean_at_least":
            val = float(r.get("cosine_mean") or 0)
            assert val >= bound, f"{stage}: cosine_mean = {val} (need >= {bound})"
        elif check == "exact_match":
            val = r.get("exact_match", "").lower()
            assert val == "true", f"{stage}: exact_match = {val}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()