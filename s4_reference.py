"""s4_reference — S4_Slicer reference implementations as plain callables.

Each function here is a faithful extraction from main.ipynb, usable as the
`s4` argument to bench.reference(). All functions are pure (no side effects
on inputs, return new arrays) so you can call them alongside your S5 pipeline
without corrupting state.

Pattern:
    import s4_reference as s4
    bench.reference('smoothing',
        s4=lambda: s4.smooth_rotation_field(n_c, cell_face_nb, init, W),
        s5_result=rotation_field,
        correctness=bench.compare_scalar_fields)

Notes on correctness of S4 itself:

  - smooth_rotation_field     : correctly formulated TRF (sqrt(W)·diff residuals).
                                Use this as the fair baseline.
  - smooth_rotation_field_buggy : verbatim main.ipynb including the quartic
                                formulation bug (passes W*diff**2 to TRF).
                                Use only for the "combined penalty" diagnostic.

  - solve_deformation         : correctly formulated (returns linear residuals).
                                Reaches a suboptimal iterate at ftol=1e-14 on
                                rank-deficient problems — this is TRF, not a bug
                                in the extraction. Expect ~20–40% residual gap
                                vs. a direct solver on small meshes.
  - solve_deformation_buggy   : verbatim main.ipynb (returns ||M_c||_F^2 per
                                cell, which TRF then squares — the quartic bug).

  - compute_barycentric_point_transforms : uses UNSIGNED sub-volume method, which
                                is what main.ipynb does. This produces WRONG
                                coords for query points outside their cell
                                (np.abs(det) loses the sign). Documented bug,
                                not an extraction error. S5's T-matrix method
                                is both faster and correct.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R


# =============================================================================
# §4.1  Adjacency
# =============================================================================

def compute_adjacency(tet, already_run) -> dict:
    """Per-cell-type neighbor lists. Mirrors main.ipynb cell 2.

    Returns {'point': (m,2), 'edge': (m,2), 'face': (m,2)} with (lo,hi) order
    — matches s5's output format so `bench.compare_edge_sets` works directly.
    """
    
    out: dict[str, np.ndarray] = {}
    n = tet.number_of_cells
    for n_type in ["point", "edge", "face"]:
        pairs = []
        for c in range(n):
            for nb in tet.cell_neighbors(c, f"{n_type}s"):
                if nb > c:
                    pairs.append((c, nb))
        out[n_type] = (np.array(pairs, dtype=np.int32)
                       if pairs else np.zeros((0, 2), dtype=np.int32))
    return out


# =============================================================================
# §4.4  Rotation-field smoothing
# =============================================================================

def smooth_rotation_field(
    n_cells: int,
    cell_face_neighbours: np.ndarray,
    initial_rotation_field: np.ndarray,
    neighbor_loss_weight: float,
    max_nfev: int = 100,
    ftol: float = 1e-6,
    xtol: float = 1e-8,
) -> np.ndarray:
    """TRF-based Laplacian smoothing with correctly-formulated residuals.

    See main.ipynb `optimize_rotations`. The original code passes W·diff² to
    least_squares, which then squares it — the quartic bug. This version passes
    sqrt(W)·diff so TRF's internal squaring produces the intended quadratic.
    Use this for the fair baseline; use smooth_rotation_field_buggy to measure
    the combined TRF+formulation penalty.
    """
    W = float(neighbor_loss_weight)
    valid_mask = ~np.isnan(initial_rotation_field)
    valid_idx = np.where(valid_mask)[0]
    n_e = len(cell_face_neighbours)
    n_valid = len(valid_idx)
    init_0 = np.nan_to_num(initial_rotation_field, nan=0.0)

    def objective(r):
        nb = np.sqrt(W) * (r[cell_face_neighbours[:, 0]] - r[cell_face_neighbours[:, 1]])
        an = r[valid_idx] - init_0[valid_idx]
        return np.concatenate((nb, an))

    def jacobian(r):
        j = lil_matrix((n_e + n_valid, n_cells), dtype=np.float64)
        c1, c2 = cell_face_neighbours[:, 0], cell_face_neighbours[:, 1]
        j[np.arange(n_e), c1] = np.sqrt(W)
        j[np.arange(n_e), c2] = -np.sqrt(W)
        j[n_e + np.arange(n_valid), valid_idx] = 1.0
        return j.tocsr()

    def jac_sparsity():
        s = lil_matrix((n_e + n_valid, n_cells), dtype=np.int8)
        s[np.arange(n_e), cell_face_neighbours[:, 0]] = 1
        s[np.arange(n_e), cell_face_neighbours[:, 1]] = 1
        s[n_e + np.arange(n_valid), valid_idx] = 1
        return s.tocsr()

    res = least_squares(
        objective, np.zeros(n_cells),
        jac=jacobian, max_nfev=max_nfev,
        jac_sparsity=jac_sparsity(),
        method="trf", ftol=ftol, xtol=xtol,
    )
    return res.x


def smooth_rotation_field_buggy(
    n_cells: int,
    cell_face_neighbours: np.ndarray,
    initial_rotation_field: np.ndarray,
    neighbor_loss_weight: float,
    max_nfev: int = 100,
    ftol: float = 1e-6,
) -> np.ndarray:
    """Verbatim main.ipynb with the quartic formulation bug preserved."""
    W = float(neighbor_loss_weight)
    valid_idx = np.where(~np.isnan(initial_rotation_field))[0]
    n_e = len(cell_face_neighbours)
    n_valid = len(valid_idx)

    def objective(r):
        diff = r[cell_face_neighbours[:, 0]] - r[cell_face_neighbours[:, 1]]
        return np.concatenate((W * diff ** 2,
                               (r[valid_idx] - initial_rotation_field[valid_idx]) ** 2))

    def jacobian(r):
        j = lil_matrix((n_e + n_valid, n_cells), dtype=np.float64)
        c1, c2 = cell_face_neighbours[:, 0], cell_face_neighbours[:, 1]
        diff = r[c1] - r[c2]
        j[np.arange(n_e), c1] = 2 * W * diff
        j[np.arange(n_e), c2] = -2 * W * diff
        j[n_e + np.arange(n_valid), valid_idx] = \
            2 * (r[valid_idx] - initial_rotation_field[valid_idx])
        return j.tocsr()

    def jac_sparsity():
        s = lil_matrix((n_e + n_valid, n_cells), dtype=np.int8)
        s[np.arange(n_e), cell_face_neighbours[:, 0]] = 1
        s[np.arange(n_e), cell_face_neighbours[:, 1]] = 1
        s[n_e + np.arange(n_valid), valid_idx] = 1
        return s.tocsr()

    return least_squares(
        objective, np.zeros(n_cells),
        jac=jacobian, max_nfev=max_nfev,
        jac_sparsity=jac_sparsity(),
        method="trf", ftol=ftol,
    ).x


# =============================================================================
# §4.5  Mesh deformation
# =============================================================================

_N4 = np.eye(4) - 0.25 * np.ones((4, 4))


def _rotation_matrices_from_field(cell_centers: np.ndarray,
                                  rotation_field: np.ndarray) -> np.ndarray:
    cc_xy = cell_centers[:, :2]
    tangent = np.cross(np.array([0.0, 0.0, 1.0]),
                       np.column_stack([cc_xy, np.zeros(len(cc_xy))]))
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    safe = norms > 0
    tangent = np.where(safe, tangent / np.where(safe, norms, 1.0),
                       np.array([1.0, 0.0, 0.0]))
    return R.from_rotvec(rotation_field[:, None] * tangent).as_matrix()


def solve_deformation(
    points: np.ndarray,
    cells: np.ndarray,
    cell_centers: np.ndarray,
    rotation_field: np.ndarray,
    max_nfev: int = 1000,
    ftol: float = 1e-6,
    xtol: float = 1e-8,
) -> np.ndarray:
    """TRF on correctly-formulated linear residuals.

    Returns new_points of shape (n_p, 3). Does NOT mutate inputs.
    """
    n_c = len(cells)
    n_p = len(points)

    rot_mats = _rotation_matrices_from_field(cell_centers, rotation_field)
    old_verts = points[cells]
    old_transformed = np.einsum(
        "cij,cjk->cik", rot_mats, (_N4 @ old_verts).transpose(0, 2, 1)
    )

    c_idx = np.arange(n_c)
    rows_, cols_, vals_ = [], [], []
    for i in range(4):
        for k in range(4):
            rows_.append(c_idx * 4 + i)
            cols_.append(cells[:, k])
            vals_.append(np.full(n_c, 0.75 if i == k else -0.25, dtype=np.float64))
    A = sparse.coo_matrix(
        (np.concatenate(vals_), (np.concatenate(rows_), np.concatenate(cols_))),
        shape=(n_c * 4, n_p), dtype=np.float64,
    ).tocsr()
    B = old_transformed.transpose(0, 2, 1).reshape(n_c * 4, 3)
    J_const = sparse.block_diag([A, A, A], format="csr")

    def objective(x):
        V = x.reshape(n_p, 3)
        return np.concatenate([A @ V[:, j] - B[:, j] for j in range(3)])

    res = least_squares(
        objective, points.reshape(-1),
        jac=lambda x: J_const,
        max_nfev=max_nfev,
        method="trf", ftol=ftol, xtol=xtol,
    )
    return res.x.reshape(n_p, 3)


def solve_deformation_buggy(
    points: np.ndarray,
    cells: np.ndarray,
    cell_centers: np.ndarray,
    rotation_field: np.ndarray,
    max_nfev: int = 1000,
    ftol: float = 1e-6,
) -> np.ndarray:
    """Verbatim main.ipynb: passes per-cell ||M_c||_F^2 as residuals → quartic bug."""
    n_c = len(cells)
    n_p = len(points)

    rot_mats = _rotation_matrices_from_field(cell_centers, rotation_field)
    old_verts = points[cells]
    old_transformed = np.einsum(
        "cij,cjk->cik", rot_mats, (_N4 @ old_verts).transpose(0, 2, 1)
    )

    def objective(params):
        V = params.reshape(-1, 3)
        new_t = (_N4 @ V[cells]).transpose(0, 2, 1)
        return np.linalg.norm(new_t - old_transformed, axis=(1, 2)) ** 2

    rows_ = np.repeat(np.arange(n_c), 4 * 3)
    cols_ = np.repeat(cells, 3, axis=1).ravel() * 3 + np.tile([0, 1, 2], n_c * 4)
    sparsity = sparse.csr_matrix(
        (np.ones(len(rows_), dtype=np.int8), (rows_, cols_)),
        shape=(n_c, 3 * n_p),
    )
    res = least_squares(
        objective, points.reshape(-1),
        max_nfev=max_nfev,
        jac_sparsity=sparsity,
        method="trf", ftol=ftol, x_scale="jac",
    )
    return res.x.reshape(n_p, 3)


# =============================================================================
# §4.6  Barycentric transforms
# =============================================================================

def _tet_vol_abs(a, b, c, d):
    return np.abs(np.linalg.det(np.vstack([b - a, c - a, d - a]))) / 6.0


def compute_barycentric_point_transforms(
    cell_vertices: np.ndarray,   # (n, 4, 3)
    query_points: np.ndarray,    # (n, 3)
) -> np.ndarray:
    """Per-point unsigned sub-volume method, matching main.ipynb.

    Known bug (carries over from S4): np.abs() loses sign, so exterior query
    points reconstruct incorrectly. S5's T-matrix method fixes this.
    """
    n = len(query_points)
    out = np.empty((n, 4))
    for i in range(n):
        a, b, c, d = cell_vertices[i]
        p = query_points[i]
        total = _tet_vol_abs(a, b, c, d)
        if total == 0:
            out[i] = [0, 0, 0, 0]; continue
        out[i, 0] = _tet_vol_abs(p, b, c, d) / total
        out[i, 1] = _tet_vol_abs(a, p, c, d) / total
        out[i, 2] = _tet_vol_abs(a, b, p, d) / total
        out[i, 3] = _tet_vol_abs(a, b, c, p) / total
    return out


# =============================================================================
# §4.7  Vertex rotation recovery
# =============================================================================

def recover_vertex_rotations(
    undeformed_points: np.ndarray,
    deformed_points: np.ndarray,
    cells: np.ndarray,
    undeformed_cell_centers: np.ndarray,
    deformed_cell_centers: np.ndarray,
    max_rotation_deg: float = 30.0,
    min_rotation_deg: float = -130.0,
) -> np.ndarray:
    """Per-cell SVD inside a Python loop → scatter to vertices."""
    n_c = len(cells)
    n_p = max(deformed_points.shape[0], undeformed_points.shape[0])

    num_cells_per_vertex = np.zeros(n_p)
    for cell in cells:
        num_cells_per_vertex[cell] += 1

    vertex_rotations = np.zeros(n_p)

    for c_idx, cell in enumerate(cells):
        new_v = deformed_points[cell] - deformed_cell_centers[c_idx]
        old_v = undeformed_points[cell] - undeformed_cell_centers[c_idx]

        xy = undeformed_cell_centers[c_idx, :2]
        n_xy = np.linalg.norm(xy)
        if n_xy < 1e-12:
            plane_x = np.array([1.0, 0.0, 0.0])
        else:
            plane_x = np.array([xy[0] / n_xy, xy[1] / n_xy, 0.0])

        new_proj = np.stack([new_v @ plane_x, new_v[:, 2]], axis=-1)
        old_proj = np.stack([old_v @ plane_x, old_v[:, 2]], axis=-1)

        cov = new_proj.T @ old_proj
        U, _, Vt = np.linalg.svd(cov)
        Rm = U @ Vt
        theta = -np.arccos(np.clip(Rm[0, 0], -1.0, 1.0))
        if Rm[1, 0] < 0:
            theta = -theta
        theta = max(min(theta, np.deg2rad(max_rotation_deg)),
                    np.deg2rad(min_rotation_deg))

        for v in cell:
            vertex_rotations[v] += theta / num_cells_per_vertex[v]

    return vertex_rotations


# =============================================================================
# §4.8  Volume scaling
# =============================================================================

def compute_volume_scales(
    undeformed_points: np.ndarray,
    deformed_points: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    n_c = len(cells)
    out = np.empty(n_c)
    for c_idx, cell in enumerate(cells):
        wa, wb, wc, wd = deformed_points[cell]
        ua, ub, uc, ud = undeformed_points[cell]
        vw = _tet_vol_abs(wa, wb, wc, wd)
        vu = _tet_vol_abs(ua, ub, uc, ud)
        out[c_idx] = vu / (vw + 1e-15)
    return out


# =============================================================================
# §4.9  Validity check (per-iter allocation pattern)
# =============================================================================

def check_validity_loop(positions: np.ndarray) -> np.ndarray:
    """Simulates the inner-loop pattern from main.ipynb's G-code reformation
    (per-iteration np.any(np.isnan(p)) call → allocates a 3-element scratch
    every iter). S5 replaces this with a single vectorized pre-pass.
    """
    n = len(positions)
    valid = np.empty(n, dtype=bool)
    for i in range(n):
        valid[i] = not np.any(np.isnan(positions[i]))
    return valid