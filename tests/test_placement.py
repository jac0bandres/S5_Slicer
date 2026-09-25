from pathlib import Path
import numpy as np
import pytest
from s3.mesh import read_tet
from s3.placement import Placement


def test_scale_rotation_and_bed_placement_preserve_topology():
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    before=mesh.points.copy()
    placed=Placement(scale=2,rotation_x=90,center_xy=True,offset_x=30,offset_y=-10).apply(mesh)
    size=np.ptp(mesh.points,axis=0)
    np.testing.assert_allclose(np.ptp(placed.points,axis=0),2*size[[0,2,1]],atol=1e-12)
    np.testing.assert_allclose((placed.points[:,:2].min(axis=0)+placed.points[:,:2].max(axis=0))/2,[30,-10])
    assert placed.points[:,2].min()==0
    np.testing.assert_array_equal(placed.cells,mesh.cells)
    np.testing.assert_array_equal(placed.faces,mesh.faces)
    np.testing.assert_array_equal(mesh.points,before)
    assert np.linalg.det(Placement(rotation_x=90,rotation_y=25).rotation)==pytest.approx(1)


def test_directions_rotate_without_scaling_or_offsets():
    placement=Placement(scale=3,rotation_x=90,rotation_z=90,offset_x=100)
    np.testing.assert_allclose(placement.rotate_vectors([[0,1,0],[1,0,0]]),
                               [[0,0,1],[0,1,0]],atol=1e-12)


@pytest.mark.parametrize('kwargs',[dict(scale=0),dict(scale=-1),dict(rotation_z=np.nan),dict(offset_x=np.inf)])
def test_invalid_placement(kwargs):
    with pytest.raises(ValueError):Placement(**kwargs)
