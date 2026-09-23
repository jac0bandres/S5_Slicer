"""Numerical translations of DeformTet.cpp's active support-free path.

The C++ residual weights are used as residual weights (their squares enter the
energy). These are deliberately not silently replaced with paper coefficients.
"""
import numpy as np
import time
from scipy import sparse
from scipy.sparse.linalg import splu, cg
from scipy.spatial.transform import Rotation
from .mesh import unit


class PinnedLeastSquares:
    """Factor a fixed system once; pinned values and right-hand sides may vary."""
    def __init__(self,a,anchors,*,progress=None,permc_spec='COLAMD'):
        self.progress=progress
        self.anchors=np.asarray(anchors,dtype=int).copy()
        self.free=np.ones(a.shape[1],dtype=bool)
        self.free[self.anchors]=False
        if progress:progress('Eliminating pinned variables')
        reduced=a[:,self.free].tocsc()
        self.transpose=reduced.T
        self.pinned=a[:,self.anchors].copy()
        self.factor=None
        if self.free.any():
            if progress:progress(f'Forming normal matrix · {reduced.shape[1]:,} free unknowns')
            normal=(self.transpose@reduced).tocsc()
            normal.eliminate_zeros()
            if progress:progress(f'Factoring sparse matrix · {normal.shape[0]:,} unknowns · {normal.nnz:,} nonzeros · {permc_spec}')
            self.factor=splu(normal,permc_spec=permc_spec)

    def solve(self,b,values):
        if self.progress:self.progress('Solving factored system')
        rhs=b-self.pinned@values
        result=np.empty((len(self.free),)+b.shape[1:])
        result[self.anchors]=values
        if self.factor is not None:
            result[self.free]=self.factor.solve(self.transpose@rhs)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite least-squares solution')
        return result


def least_squares_pinned(a, b, anchors, values):
    """Remove translation gauges before solving normal equations."""
    return PinnedLeastSquares(a,anchors).solve(b,values)


def iterative_least_squares_pinned(a,b,anchors,values,*,initial=None,progress=None,
                                   tolerance=1e-9,max_iterations=5000):
    """Column-scaled CG on pinned normal equations, without LU fill-in.

    Accept only a verified scaled normal residual. Nonconvergence is explicit;
    do not silently fall back to the potentially expensive direct factorization.
    """
    free=np.ones(a.shape[1],dtype=bool); free[anchors]=False
    result=np.empty(a.shape[1]); result[anchors]=values
    if not free.any():return result
    if progress:progress('Preparing iterative system and diagonal preconditioner')
    reduced=a[:,free].tocsr()
    rhs=b-a[:,anchors]@values
    norms=np.sqrt(np.asarray(reduced.power(2).sum(axis=0)).ravel())
    if np.any(norms<=0) or not np.isfinite(norms).all():
        raise ValueError('Invalid iterative system column norms')
    scaled=reduced@sparse.diags(1/norms)
    normal=(scaled.T@scaled).tocsr(); normal.eliminate_zeros()
    target=np.asarray(scaled.T@rhs).ravel()
    target_norm=float(np.linalg.norm(target))
    limit=tolerance*max(target_norm,1e-30)
    count=0; last_update=time.perf_counter()
    def monitor(x):
        nonlocal count,last_update
        count+=1
        if progress and (count==1 or time.perf_counter()-last_update>=.5):
            residual=np.linalg.norm(normal@x-target)/max(target_norm,1e-30)
            progress(f'CG iteration {count}/{max_iterations} · relative normal residual {residual:.3g} · target {tolerance:.1g}')
            last_update=time.perf_counter()
    if progress:progress(f'Iterative CG · {normal.shape[0]:,} unknowns · {normal.nnz:,} nonzeros')
    x,info=cg(normal,target,x0=None if initial is None else initial[free]*norms,
              rtol=tolerance,atol=0.,maxiter=max_iterations,callback=monitor)
    residual=float(np.linalg.norm(normal@x-target))
    if info!=0 or not np.isfinite(x).all() or not np.isfinite(residual) or residual>max(limit*2,1e-30):
        raise ValueError(f'Eq. 12 iterative solve did not converge after {count} iterations '
                         f'(relative normal residual {residual/max(target_norm,1e-30):.3g}). '
                         'Try the Direct (LU) Eq. 12 solver or increase scale_max_iterations in the JSON config.')
    result[free]=x/norms
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite iterative least-squares solution')
    if progress:progress(f'CG converged · {count} iterations · relative normal residual {residual/max(target_norm,1e-30):.3g}')
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


def from_two_vectors_batch(a,b):
    """Batched minimal rotations, preserving the scalar antiparallel fallback."""
    a,b=np.broadcast_arrays(unit(a),unit(b))
    dots=np.clip(np.einsum('ij,ij->i',a,b),-1,1)
    opposite=dots < -1+1e-12
    result=np.empty((len(a),3,3))
    regular=~opposite
    if regular.any():
        q=np.column_stack([1+dots[regular],np.cross(a[regular],b[regular])])
        result[regular]=Rotation.from_quat(unit(q),scalar_first=True).as_matrix()
    for i in np.flatnonzero(opposite):
        result[i]=from_two_vectors(a[i],b[i])
    return result


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
