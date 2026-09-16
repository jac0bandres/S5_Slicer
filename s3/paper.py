"""Equation-based S³ kernels (SIGGRAPH Asia 2022, Eqs. 7, 8, 12, 13).

This module is the paper implementation. ``deform.py`` is a separately labeled
C++ comparison port and is deliberately not called here.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from .mesh import unit
from .numerics import from_two_vectors, least_squares_pinned


def project_sphere_halfspaces(direction, a, b, tol=1e-10,*,preferred=None):
    """Nearest direction on S² intersect {a_i.d >= b_i}, by active-set enumeration.

    The optimum is interior, on one small circle, or at a pair of circle
    intersections. Enumerating those cases gives the global spherical projection
    in three dimensions, including nonconvex feasible subsets of the sphere.
    """
    d = unit(direction)
    a = np.asarray(a,dtype=float).reshape(-1,3)
    b = np.asarray(b,dtype=float)
    if len(a) != len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid halfspace data')
    if not len(a):
        return d
    lengths = np.linalg.norm(a,axis=1)
    if np.any(lengths < 1e-14):
        raise ValueError('Zero halfspace normal')
    a,b = a/lengths[:,None],b/lengths
    if np.any(b > 1+tol):
        raise ValueError('Empty spherical feasible region')
    candidates = [d]
    for n,c in zip(a,b):
        if abs(c)>1+tol:
            continue
        c = np.clip(c,-1,1)
        center=n*c
        radius=np.sqrt(max(0.,1-c*c))
        tangent=d-(d@n)*n
        if np.linalg.norm(tangent)<1e-12:
            tangent=np.zeros(3) if preferred is None else np.asarray(preferred)-(np.asarray(preferred)@n)*n
            if np.linalg.norm(tangent)<1e-12:
                basis=np.eye(3)[np.argmin(np.abs(n))]
                tangent=np.cross(n,basis)
        tangent=unit(tangent)
        candidates.extend([center+radius*tangent,center-radius*tangent])
    for i,j in combinations(range(len(a)),2):
        normals=a[[i,j]]
        axis=np.cross(*normals)
        if np.linalg.norm(axis)<1e-12:
            continue
        center=normals.T @ np.linalg.solve(normals@normals.T,b[[i,j]])
        r2=1-center@center
        if r2 < -tol:
            continue
        offset=np.sqrt(max(0.,r2))*unit(axis)
        candidates.extend([center+offset,center-offset])
    candidates=np.array(candidates)
    feasible=np.all(candidates@a.T >= b-tol,axis=1)
    candidates=candidates[feasible]
    if not len(candidates):
        raise ValueError('Empty spherical feasible region')
    scores=candidates@d
    if preferred is not None:
        tied=np.flatnonzero(scores>=scores.max()-tol)
        return candidates[tied[np.argmax(candidates[tied]@preferred)]]
    return candidates[np.argmax(scores)]


def project_printing_direction(direction, *, sf_normals=(), stress=None, sq_normal=None, sq_normals=(),
                               alpha=30., beta=10., gamma=10.,preferred=None):
    """Eq. 7: SF halfspaces, SR band, and SQ union of band and two poles."""
    d=unit(direction)
    a,b=[],[]
    for n in sf_normals:
        a.append(unit(n)); b.append(-np.sin(np.deg2rad(alpha)))
    if stress is not None:
        stress=unit(stress)
        a.extend([stress,-stress]); b.extend([-np.sin(np.deg2rad(beta))]*2)
    quality = list(sq_normals)
    if sq_normal is not None:
        quality.append(sq_normal)
    if not quality:
        return project_sphere_halfspaces(d,a,b,preferred=preferred),False
    quality=unit(np.asarray(quality))
    candidates=[]
    band_a=a+list(quality)+list(-quality)
    band_b=b+[-np.sin(np.deg2rad(gamma))]*(2*len(quality))
    try:
        candidates.append((project_sphere_halfspaces(d,band_a,band_b,preferred=preferred),False))
    except ValueError:
        pass
    for pole in np.vstack([quality,-quality]):
        sq_dots=np.abs(quality@pole)
        quality_ok=np.all((sq_dots<=np.sin(np.deg2rad(gamma))+1e-10)|(sq_dots>=1-1e-10))
        if quality_ok and (not len(a) or np.all(np.asarray(a)@pole >= np.asarray(b)-1e-10)):
            candidates.append((pole,True))
    if not candidates:
        raise ValueError('Incompatible fabrication constraints')
    return max(candidates,key=lambda item: item[0]@d)


def blend_quaternions(mesh, rotations, targets, keep_weights, edge_weights,*,fixed_identity=None):
    """Eq. 8, pairwise incidence energy, with normalized quaternion output.

    Targets use the same hemisphere as the current rotations; all initial
    rotations receive a consistent graph sign lift before a solve. Quaternion
    double-cover handling is a documented implementation choice.
    """
    m=len(mesh.cells); pairs=mesh.neighbor_pairs
    q=Rotation.from_matrix(rotations).as_quat(scalar_first=True)
    qt=Rotation.from_matrix(targets).as_quat(scalar_first=True)
    seen=np.zeros(m,dtype=bool)
    for root in range(m):
        if seen[root]: continue
        seen[root]=True; queue=[root]
        for i in queue:
            for j in mesh.neighbors[i]:
                if not seen[j]:
                    if q[i]@q[j]<0: q[j]*=-1
                    seen[j]=True; queue.append(j)
    qt[np.einsum('ij,ij->i',q,qt)<0]*=-1
    keep_weights=np.asarray(keep_weights,dtype=float)
    edge_weights=np.asarray(edge_weights,dtype=float)
    if np.any(keep_weights<0) or np.any(edge_weights<=0):
        raise ValueError('Invalid energy weights')
    ids=np.flatnonzero(keep_weights>0)
    rows=np.r_[np.repeat(np.arange(len(pairs)),2),len(pairs)+np.arange(len(ids))]
    cols=np.r_[pairs.ravel(),ids]
    vals=np.r_[np.repeat(np.sqrt(edge_weights),2)*np.tile([1,-1],len(pairs)),
               np.sqrt(keep_weights[ids])]
    a=sparse.coo_matrix((vals,(rows,cols)),shape=(len(pairs)+len(ids),m)).tocsr()
    rhs=np.zeros((a.shape[0],4)); rhs[len(pairs):]=np.sqrt(keep_weights[ids,None])*qt[ids]
    # Unconstrained components have a constant-quaternion minimizer. Preserve
    # their current root orientation rather than solving singular equations.
    graph=sparse.coo_matrix((np.ones(len(pairs)*2),
                            (np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(m,m))
    count,labels=connected_components(graph)
    fixed=np.asarray([] if fixed_identity is None else fixed_identity,dtype=int)
    anchors=[]
    for c in range(count):
        indices=np.flatnonzero(labels==c)
        if not np.any(keep_weights[indices]>0) and not np.isin(indices,fixed).any(): anchors.append(indices[0])
    anchors=np.r_[np.array(anchors,dtype=int),fixed]
    values=q[anchors].copy()
    if len(fixed):
        values[-len(fixed):]=0
        values[-len(fixed):,0]=np.where(q[fixed,0]<0,-1.,1.)
    solved=least_squares_pinned(a,rhs,anchors,values)
    return Rotation.from_quat(unit(solved),scalar_first=True).as_matrix()


def paper_scale_system(mesh, rotations, rigidity=1., compatibility=6.):
    """Eq. 12 exactly as printed: centered Xᵀ − R diag(s) centered Vᵀ.

    This is one coupled linear least-squares problem in 3*n+3*m unknowns;
    unlike the upstream axiswise implementation, rotation follows scaling.
    """
    if rigidity<=0 or compatibility<0:
        raise ValueError('Require positive rigidity and nonnegative compatibility')
    n,m=len(mesh.points),len(mesh.cells); pairs=mesh.neighbor_pairs
    p=mesh.points[mesh.cells]; p=p-p.mean(axis=1,keepdims=True)
    row_grid=np.arange(12*m).reshape(m,4,3)
    rows=[];cols=[];vals=[]
    for k in range(4):
        for j in range(4):
            rows.append(row_grid[:,k,:].ravel())
            cols.append((3*mesh.cells[:,j,None]+np.arange(3)).ravel())
            vals.append(np.full(3*m,(k==j)-0.25))
        for scale_axis in range(3):
            rows.append(row_grid[:,k,:].ravel())
            cols.append(np.repeat(3*n+3*np.arange(m)+scale_axis,3))
            vals.append((-rotations[:,:,scale_axis]*p[:,k,scale_axis,None]).ravel())
    offset=12*m
    neighbor_rows=offset+np.arange(3*len(pairs)).reshape(-1,3)
    for side,sign in [(0,1),(1,-1)]:
        rows.append(neighbor_rows.ravel())
        cols.append((3*n+3*pairs[:,side,None]+np.arange(3)).ravel())
        vals.append(np.full(3*len(pairs),sign*np.sqrt(compatibility)))
    offset+=3*len(pairs)
    rows.append(offset+np.arange(3*m));cols.append(3*n+np.arange(3*m))
    vals.append(np.full(3*m,np.sqrt(rigidity)))
    a=sparse.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
                       shape=(offset+3*m,3*n+3*m)).tocsr()
    b=np.zeros(a.shape[0]);b[offset:]=np.sqrt(rigidity)
    return a,b


def solve_paper_scales(mesh,rotations,rigidity=1.,compatibility=6.,*,fixed_vertices=None):
    a,b=paper_scale_system(mesh,rotations,rigidity,compatibility)
    vertices=mesh.anchors
    if fixed_vertices is not None:
        fixed_vertices=np.asarray(fixed_vertices,dtype=int)
        fixed_components=np.unique(mesh.vertex_components[fixed_vertices])
        gauges=mesh.anchors[~np.isin(mesh.vertex_components[mesh.anchors],fixed_components)]
        vertices=np.union1d(gauges,fixed_vertices)
    anchors=(3*vertices[:,None]+np.arange(3)).ravel()
    values=mesh.points[vertices].ravel()
    x=least_squares_pinned(a,b,anchors,values)
    n=len(mesh.points)
    return x[:3*n].reshape(-1,3),x[3*n:].reshape(-1,3)
