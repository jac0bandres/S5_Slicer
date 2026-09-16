"""Active support-free deformation entry point from the official S³ GUI."""
from dataclasses import dataclass, asdict
import numpy as np
from .mesh import TetMesh, unit
from .heat import growing_field
from .numerics import (fit_rotations, support_projection, smooth_quaternions,
                       solve_scales, least_squares_pinned)


@dataclass(frozen=True)
class SupportFreeConfig:
    # MainWindow::_updateFrameworkParameter, ring preset.
    outer_iterations: int = 2
    inner_iterations: int = 7
    critical_weight: float = 15.
    neighbor_scale_weight: float = 5.
    regular_scale_weight: float = 1.
    quaternion_smooth_weight: float = 40.
    quaternion_keep_weight: float = 1.
    support_angle: float = 30.
    bottom_height: float = 1.

    def __post_init__(self):
        if self.outer_iterations < 1 or self.inner_iterations < 1:
            raise ValueError('Iteration counts must be positive')
        if not 0 <= self.support_angle < 90:
            raise ValueError('Support angle must be in [0, 90)')
        for key, value in asdict(self).items():
            if not np.isfinite(value) or ('weight' in key and value <= 0):
                raise ValueError(f'Invalid parameter: {key}')
        if self.bottom_height <= 0:
            raise ValueError('Bottom height must be positive')


def detect_overhang(mesh, points, bottom, angle):
    # QMesh boundary winding has inward normals. Preserve the upstream sign.
    face_mask = mesh.boundary & (mesh.normals(points)[:,1] > np.sin(angle*3.1415926/180))
    face_mask[np.unique(mesh.cell_faces[bottom])] = False
    return face_mask, face_mask[mesh.cell_faces].any(axis=1) & ~bottom


def inverse_scalar_field(mesh, deformed, smooth_iterations=10):
    """MainWindow::inverseDeformation support-free branch, including smoothing."""
    height = deformed[:,1]
    span = np.ptp(height)
    if span < 1e-14:
        raise ValueError('Deformation has zero height range')
    raw = (height-height.min())/span
    vectors = unit(mesh.gradient(raw))
    # Intentionally sequential/in-place, just like _vectorField_smooth.
    for _ in range(smooth_iterations):
        for i, ns in enumerate(mesh.neighbors):
            vectors[i] = unit(vectors[i] + vectors[ns].sum(axis=0))
    a = -mesh.gradient_operator()/6
    guide = least_squares_pinned(a, vectors.ravel(), mesh.anchors,
                                np.zeros(len(mesh.anchors)))
    if np.ptp(guide) < 1e-14:
        raise ValueError('Reconstructed scalar field is constant')
    return 1-(guide-guide.min())/np.ptp(guide), raw


def support_free(mesh, config=None, *, initial_growing=None, callback=None):
    """Translate runASAP_SupportLess_test3; no Cura/S4 deformation is involved.

    Numerical parity with the complete official executable is not yet certified.
    Unsupported degenerate directions raise instead of propagating upstream NaNs.
    """
    cfg = config or SupportFreeConfig()
    original = mesh.points.copy()
    original[:,1] -= original[:,1].min()
    mesh = TetMesh(original, mesh.cells)
    bottom = (mesh.points[mesh.cells,1] < cfg.bottom_height).any(axis=1)
    normals = mesh.normals()
    if initial_growing is None:
        growing, heat = growing_field(mesh, cfg.bottom_height)
    else:
        growing = np.asarray(initial_growing, dtype=float)
        if growing.shape != (len(mesh.cells),3):
            raise ValueError('initial_growing must have shape (tet_count, 3)')
        unit(growing)
        heat = None
    points = mesh.points.copy()
    history = []
    for outer in range(cfg.outer_iterations):
        faces, overhang = detect_overhang(mesh, points, bottom, cfg.support_angle)
        rotations = fit_rotations(mesh, points)
        selected = np.flatnonzero(overhang)
        # Preserve _get_initial_overhang_faceNormal's first-face-only behavior.
        first_faces = mesh.cell_faces[selected][np.arange(len(selected)),
                          np.argmax(faces[mesh.cell_faces[selected]],axis=1)]
        for inner in range(cfg.inner_iterations):
            for ci, fi in zip(selected, first_faces):
                normal = rotations[ci] @ -normals[fi]
                direction = rotations[ci] @ growing[ci]
                rotations[ci] = support_projection(normal,direction,cfg.support_angle) @ rotations[ci]
            rotations = smooth_quaternions(mesh, rotations, overhang | bottom,
                                           cfg.quaternion_smooth_weight,cfg.quaternion_keep_weight)
        weights = np.where(overhang,cfg.critical_weight,1.)
        points, scales = solve_scales(mesh,rotations,weights,
                                     cfg.neighbor_scale_weight,cfg.regular_scale_weight)
        f = np.einsum('cij,cik->ckj',mesh.grad_basis,points[mesh.cells])
        det = np.linalg.det(f)
        remaining, _ = detect_overhang(mesh,points,bottom,cfg.support_angle)
        row = dict(outer=outer+1, targeted_cells=int(overhang.sum()),
                   overhang_faces=int(remaining.sum()), inverted_cells=int((det<0).sum()),
                   min_determinant=float(det.min()), max_determinant=float(det.max()))
        history.append(row)
        if callback:
            callback(row)
    # postProcess centers X/Z by node mean and moves the lowest Y to zero.
    points[:,[0,2]] -= points[:,[0,2]].mean(axis=0)
    points[:,1] -= points[:,1].min()
    scalar, raw_scalar = inverse_scalar_field(mesh,points)
    return dict(points=mesh.points, cells=mesh.cells, deformed=points, scales=scales,
                rotations=rotations, scalar=scalar, raw_scalar=raw_scalar,
                growing=growing, heat=heat, history=history, config=asdict(cfg))
