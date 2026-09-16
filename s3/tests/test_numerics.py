import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from s3.mesh import TetMesh, read_tet, write_tet
from s3.numerics import (solve_scales, fit_rotations, matrix_to_quaternion,
                         smooth_quaternions, support_projection)
from s3.deform import inverse_scalar_field, support_free, SupportFreeConfig
from s3.layers import isosurface


@pytest.fixture
def mesh():
    # Two positive tetrahedra sharing one face, non-axis-aligned geometry.
    return TetMesh(np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],[0.,0.,2.],[2.,2.,2.]]),
                   np.array([[0,1,2,3],[4,2,1,3]]))


def test_affine_gradient_and_topology(mesh):
    direction = np.array([2.,-3.,5.])
    scalar = mesh.points @ direction + 7
    np.testing.assert_allclose(mesh.gradient(scalar),np.tile(direction,(2,1)))
    np.testing.assert_allclose((mesh.gradient_operator()@scalar).reshape(-1,3),
                               mesh.gradient(scalar))
    assert len(mesh.neighbor_pairs) == 1
    assert mesh.boundary.sum() == 6
    assert mesh.neighbors == [[1],[0]]


def test_identity_and_uniform_large_rotation(mesh):
    for angle in [0,160]:
        r = Rotation.from_rotvec(np.deg2rad(angle)*np.array([1.,2.,3.])/np.sqrt(14)).as_matrix()
        targets = np.broadcast_to(r,(2,3,3)).copy()
        points,scales = solve_scales(mesh,targets,np.ones(2),5.,1.)
        expected = (mesh.points-mesh.points[0])@r.T+mesh.points[0]
        np.testing.assert_allclose(points,expected,atol=2e-12)
        np.testing.assert_allclose(scales,1,atol=2e-12)
        np.testing.assert_allclose(fit_rotations(mesh,points),targets,atol=2e-12)


def test_scale_solve_against_independent_dense_design(mesh):
    rotations = Rotation.from_euler('xyz',[[15,30,40],[-20,10,50]],degrees=True).as_matrix()
    weights = [2.,3.]
    actual, actual_scales = solve_scales(mesh,rotations,np.array(weights),5.,1.)
    n,m = len(mesh.points),len(mesh.cells)
    for axis in range(3):
        rows,b = [],[]
        for ci,cell in enumerate(mesh.cells):
            v = mesh.points[cell]
            target = (v-v.mean(axis=0))@rotations[ci].T
            for local in range(4):
                row = np.zeros(n+m)
                for k,node in enumerate(cell):
                    row[node] = weights[ci]*((k==local)-0.25)
                row[n+ci] = -weights[ci]*target[local,axis]
                rows.append(row); b.append(0.)
        row = np.zeros(n+m); row[n:n+2] = [5.,-5.]
        rows.append(row); b.append(0.)
        for ci in range(m):
            row = np.zeros(n+m); row[n+ci]=1
            rows.append(row); b.append(1.)
        a = np.array(rows)
        expected = np.linalg.lstsq(a[:,1:],b,rcond=None)[0]
        np.testing.assert_allclose(actual[1:,axis],expected[:n-1],atol=2e-12)
        np.testing.assert_allclose(actual_scales[:,axis],expected[n-1:],atol=2e-12)


def test_quaternion_reconstruction_and_constant_field(mesh):
    matrices = Rotation.from_euler('xyz',[[1,2,3],[170,20,-80],[-20,175,10]],degrees=True).as_matrix()
    qs = matrix_to_quaternion(matrices)
    np.testing.assert_allclose(Rotation.from_quat(qs,scalar_first=True).as_matrix(),matrices,atol=1e-14)
    constant = np.repeat(matrices[:1],2,axis=0)
    smoothed = smooth_quaternions(mesh,constant,np.ones(2,dtype=bool),40.,1.)
    np.testing.assert_allclose(smoothed,constant,atol=1e-12)


def test_projection_satisfies_requested_angle():
    n = np.array([0.6,-0.8,0.])
    rotation = support_projection(n,np.array([0.1,0.9,0.2]),30.)
    np.testing.assert_allclose((rotation@n)[1],-0.5,atol=1e-14)
    with pytest.raises(ValueError,match='parallel'):
        support_projection(n,n,30.)


def test_disconnected_translation_gauges(mesh):
    points = np.vstack([mesh.points,mesh.points+10])
    cells = np.vstack([mesh.cells,mesh.cells+len(mesh.points)])
    disconnected = TetMesh(points,cells)
    x,s = solve_scales(disconnected,np.tile(np.eye(3),(4,1,1)),np.ones(4),5,1)
    np.testing.assert_allclose(x,points,atol=1e-11)
    np.testing.assert_allclose(s,1,atol=1e-11)


def test_scalar_reconstruction_and_slice_plane(mesh):
    field,raw = inverse_scalar_field(mesh,mesh.points)
    np.testing.assert_allclose(field,raw,atol=1e-13)
    points,triangles = isosurface(mesh,field.copy(),0.37)
    np.testing.assert_allclose(points[:,1],0.74,atol=1e-12)
    assert len(triangles) > 0
    assert len(np.unique(points,axis=0)) == len(points)


def test_tet_roundtrip_and_rejection(mesh,tmp_path):
    path=tmp_path/'mesh.tet'
    write_tet(path,mesh.points,mesh.cells)
    loaded=read_tet(path)
    np.testing.assert_array_equal(loaded.cells,mesh.cells)
    np.testing.assert_allclose(loaded.points,mesh.points)
    with pytest.raises(ValueError,match='Degenerate'):
        TetMesh(mesh.points[:4]*[1,1,0],mesh.cells[:1])


def test_full_identity_without_heat(mesh):
    cfg=SupportFreeConfig(outer_iterations=2,inner_iterations=2,bottom_height=3.)
    result=support_free(mesh,cfg,initial_growing=np.tile([0,1,0],(2,1)))
    expected=mesh.points.copy(); expected[:,[0,2]]-=expected[:,[0,2]].mean(axis=0)
    np.testing.assert_allclose(result['deformed'],expected,atol=1e-11)
    assert all(row['inverted_cells']==0 for row in result['history'])
