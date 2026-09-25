"""heatMethod.cpp's BoundCon=100 path, used by support-free preprocessing."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .mesh import unit


def growing_field(mesh, bottom_height=1.):
    n = len(mesh.points)
    p = mesh.points[mesh.cells]
    # Tet cotangent stiffness is equivalent to the FEM basis-gradient form.
    stiffness = mesh.volumes[:,None,None] * (mesh.grad_basis @ mesh.grad_basis.transpose(0,2,1))
    rr = np.broadcast_to(mesh.cells[:,:,None], stiffness.shape).ravel()
    cc = np.broadcast_to(mesh.cells[:,None,:], stiffness.shape).ravel()
    weight = sparse.coo_matrix((-stiffness.ravel(),(rr,cc)),shape=(n,n)).tocsr()
    mass = np.zeros(n)
    # QMeshTetra::CalVolume returns |det(edges)|, i.e. SIX times geometric
    # volume. Preserve that upstream mass convention rather than correcting it.
    np.add.at(mass, mesh.cells.ravel(), np.repeat(6*mesh.volumes/4,4))
    sources = mesh.points[:,1] < mesh.points[:,1].min()+bottom_height
    if not sources.any():
        raise ValueError('Heat field has no source vertices')
    edges = np.unique(np.sort(np.concatenate([mesh.cells[:,[i,j]] for i in range(4)
                                             for j in range(i+1,4)]),axis=1),axis=0)
    t = np.linalg.norm(mesh.points[edges[:,0]]-mesh.points[edges[:,1]],axis=1).mean()**2
    invmass = 1/mass; invmass[sources] = 0
    system = sparse.eye(n)-100*t*sparse.diags(invmass)@weight
    heat = spsolve(system.tocsc(),sources.astype(float))
    direction = unit(mesh.gradient(heat))
    # Opposite face normal points toward the vertex: area*n/3 = volume*grad(phi).
    local_div = -mesh.volumes[:,None]*np.einsum('cij,cj->ci',mesh.grad_basis,direction)
    div = np.zeros(n); np.add.at(div,mesh.cells.ravel(),local_div.ravel())
    div[sources] = 0
    system = sparse.diags((~sources).astype(float))@weight+sparse.diags(sources.astype(float))
    potential = spsolve(system.tocsc(),div)
    span = np.ptp(potential)
    if not np.isfinite(potential).all() or span < 1e-14:
        raise ValueError('Heat potential is constant/nonfinite')
    heat_field = (potential-potential.min())/span
    # The apparent smoothing loop in _cal_heatMethod changes display vectorField
    # only, not vectorField_4voxelOrder used by the deformation.
    # DeformTet::VolumeMatrix divides by 6*CalVolume, hence grad(phi)/6
    # for the positive orientation used in the official datasets.
    return mesh.gradient(1-heat_field)/6, heat_field
