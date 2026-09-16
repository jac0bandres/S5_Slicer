"""Numerical translations of DeformTet.cpp's active support-free path.

The C++ residual weights are used as residual weights (their squares enter the
energy). These are deliberately not silently replaced with paper coefficients.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from .mesh import unit


def least_squares_pinned(a, b, anchors, values):
    """Remove translation gauges before solving normal equations."""
    free = np.ones(a.shape[1], dtype=bool)
    free[anchors] = False
    reduced = a[:, free].tocsc()
    rhs = b - a[:, anchors] @ values
    result = np.empty((a.shape[1],) + b.shape[1:])
    result[anchors] = values
    if free.any():
        result[free] = splu((reduced.T @ reduced).tocsc()).solve(reduced.T @ rhs)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite least-squares solution')
    return result


def matrix_to_quaternion(matrices):
    """Eigen Quaternion(Matrix3) branch/sign convention, output w,x,y,z.

    SciPy may choose a different representative of q/-q; that matters when the
    upstream algorithm smooths quaternion components without hemisphere fixing.
    """
    qs = []
    for m in matrices:
        q = np.empty(4)
        trace = np.trace(m)
        if trace > 0:
            t = np.sqrt(trace + 1)
            q[0] = 0.5*t
            q[1:] = np.array([m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]]) * (0.5/t)
        else:
            i = int(np.argmax(np.diag(m)))
            j, k = (i+1)%3, (i+2)%3
            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i+1] = 0.5*t
            q[0] = (m[k,j]-m[j,k])*(0.5/t)
            q[j+1] = (m[j,i]+m[i,j])*(0.5/t)
            q[k+1] = (m[k,i]+m[i,k])*(0.5/t)
        qs.append(q)
    return np.asarray(qs)


def from_two_vectors(a, b):
    a, b = unit(a), unit(b)
    d = np.clip(a @ b, -1, 1)
    if d < -1 + 1e-12:
        # Eigen uses SVD for this ambiguous case. Deterministic equivalent axis.
        _, _, vt = np.linalg.svd(np.vstack([a, b]), full_matrices=True)
        return Rotation.from_rotvec(np.pi * vt[-1]).as_matrix()
    cross = np.cross(a, b)
    q = np.r_[1+d, cross]
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def fit_rotations(mesh, points):
    current = points[mesh.cells]
    current -= current.mean(axis=1, keepdims=True)
    original = mesh.points[mesh.cells]
    original -= original.mean(axis=1, keepdims=True)
    f = (np.linalg.pinv(original) @ current).transpose(0, 2, 1)
    u, _, vt = np.linalg.svd(f)
    # Deliberately no determinant correction: matches the C++ U * V.transpose().
    return u @ vt


def smooth_quaternions(mesh, rotations, critical, smooth_weight, keep_weight):
    """_globalQuaternionSmooth1_supportLess: ||L_rw q||² plus anchors."""
    count = len(mesh.cells)
    rows, cols, vals = list(range(count)), list(range(count)), [smooth_weight]*count
    for i, ns in enumerate(mesh.neighbors):
        for j in ns:
            rows.append(i); cols.append(j); vals.append(-smooth_weight/len(ns))
    ids = np.flatnonzero(critical)
    rows.extend(count + np.arange(len(ids)))
    cols.extend(ids)
    vals.extend([keep_weight]*len(ids))
    a = sparse.coo_matrix((vals, (rows, cols)), shape=(count+len(ids), count)).tocsc()
    b = np.zeros((a.shape[0], 4))
    b[count:] = keep_weight * matrix_to_quaternion(rotations[ids])
    q = splu((a.T @ a).tocsc()).solve(a.T @ b)
    q = unit(q)
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def support_projection(normal, growing_direction, angle_degrees):
    """_cal_rotationMatrix_supportLess, preserving its two-candidate rule."""
    normal, growing_direction = unit(normal), unit(growing_direction)
    axis = np.cross(normal, growing_direction)
    if np.linalg.norm(axis) < 1e-14:
        # The C++ normalizes zero here; reject instead of fabricating a rotation.
        raise ValueError('Support projection has parallel normal/growing direction')
    axis = unit(axis)
    theta = np.deg2rad(90 + angle_degrees)
    candidates = Rotation.from_rotvec(np.array([theta, -theta])[:, None]*axis).apply(normal)
    dots = candidates @ growing_direction
    target = candidates[0 if dots[0] > dots[1] else 1]
    return from_two_vectors(target, np.array([0., 1., 0.]))


def scale_system(mesh, rotations, weights, neighbor_weight, regular_weight, axis):
    """C++ blocks 11/12/22/32; scale acts on the rotated frame per world axis."""
    m, n = len(mesh.cells), len(mesh.points)
    pairs = mesh.neighbor_pairs
    p = mesh.points[mesh.cells]
    p -= p.mean(axis=1, keepdims=True)
    frame = np.einsum('cij,ckj->cki', rotations, p)
    center = np.eye(4) - 0.25
    rows = np.broadcast_to(np.arange(4*m).reshape(m,4,1), (m,4,4)).ravel()
    cols = np.broadcast_to(mesh.cells[:,None,:], (m,4,4)).ravel()
    vals = (weights[:,None,None]*np.broadcast_to(center, (m,4,4))).ravel()
    rows = np.r_[rows, np.arange(4*m), np.repeat(4*m+np.arange(len(pairs)),2),
                 4*m+len(pairs)+np.arange(m)]
    cols = np.r_[cols, np.repeat(n+np.arange(m),4), n+pairs.ravel(), n+np.arange(m)]
    vals = np.r_[vals, (-weights[:,None]*frame[:,:,axis]).ravel(),
                 np.tile([neighbor_weight,-neighbor_weight],len(pairs)),
                 np.full(m, regular_weight)]
    a = sparse.coo_matrix((vals,(rows,cols)),shape=(5*m+len(pairs),n+m)).tocsr()
    b = np.zeros(a.shape[0]); b[4*m+len(pairs):] = regular_weight
    return a, b


def solve_scales(mesh, rotations, weights, neighbor_weight, regular_weight):
    answers = []
    for axis in range(3):
        a,b = scale_system(mesh, rotations, weights, neighbor_weight, regular_weight, axis)
        answers.append(least_squares_pinned(a,b,mesh.anchors,mesh.points[mesh.anchors,axis]))
    x = np.column_stack(answers)
    return x[:len(mesh.points)], x[len(mesh.points):]
