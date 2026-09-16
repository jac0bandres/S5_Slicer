import numpy as np
import pytest
from s3.surface import Surface
from s3.toolpaths import ToolpathConfig,SurfaceFEM,refine_surface,generate_toolpaths


def square(size=4.,offset=(0,0,0)):
    return Surface(np.array([[0,0,0],[size,0,0],[size,size,0],[0,size,0]])+offset,
                   [[0,1,2],[0,2,3]],cell_ids=np.array([0,0]))


def test_planar_boundary_distance_and_contours():
    layer=square()
    refined=refine_surface(layer,.2,20000)
    fem=SurfaceFEM(refined)
    distance,negative=fem.boundary_distance()
    exact=np.minimum(refined.points[:,:2],4-refined.points[:,:2]).min(axis=1)
    assert np.mean(np.abs(distance-exact))<.12
    assert negative==0
    paths,report=generate_toolpaths(layer,ToolpathConfig(spacing=.4,waypoint_distance=.13))
    assert len(paths)>=4 and all(p.closed for p in paths)
    for path in paths:
        np.testing.assert_allclose(path.points[:,2],0,atol=1e-12)
        np.testing.assert_allclose(path.normals,np.tile([0,0,1],(len(path.points),1)),atol=1e-12)
        assert np.linalg.norm(np.diff(path.points,axis=0),axis=1).max()<=.13+1e-12
        assert np.all(path.points[:,:2]>0) and np.all(path.points[:,:2]<4)
    assert not report['coverage_certified']


def test_disconnected_layers_never_bridge_islands():
    left,right=square(),square(offset=(10,0,0))
    layer=Surface(np.vstack([left.points,right.points]),np.vstack([left.faces,right.faces+4]))
    paths,report=generate_toolpaths(layer,ToolpathConfig(spacing=.5))
    assert report['components']==2
    assert len(paths)>=4
    for path in paths:
        assert np.ptp(path.points[:,0])<4
        assert path.closed


def test_hybrid_paths_follow_constant_principal_stress():
    paths,report=generate_toolpaths(square(6),ToolpathConfig(mode='hybrid',spacing=.5),
                                    stress=[[1.,0,0]],stress_mask=[True])
    interior=[p for p in paths if p.kind=='stress']
    assert len(interior)>=5
    assert report['stress_angle_mean_degrees']<1e-4
    for path in interior:
        assert not path.closed
        assert np.ptp(path.points[:,1])<1e-8
        assert path.points[:,0].min()>.8 and path.points[:,0].max()<5.2
    assert any(p.kind=='contour' for p in paths)


def test_noncritical_hybrid_layer_uses_contours_and_missing_stress_fails():
    cfg=ToolpathConfig(mode='hybrid',spacing=.5)
    paths,report=generate_toolpaths(square(),cfg,stress=[[1.,0,0]],stress_mask=[False])
    assert report['mode']=='contour' and all(p.kind=='contour' for p in paths)
    with pytest.raises(ValueError,match='require'):
        generate_toolpaths(square(),cfg)


def test_curved_waypoints_stay_on_facets():
    layer=Surface([[0,0,0],[3,0,0],[3,3,2],[0,3,0]],[[0,1,2],[0,2,3]])
    paths,_=generate_toolpaths(layer,ToolpathConfig(spacing=.4,waypoint_distance=.1))
    assert paths
    for path in paths:
        assert layer.distances(path.points).max()<1e-10
        midpoints=(path.points[:-1]+path.points[1:])/2
        assert layer.distances(midpoints).max()<1e-10
        np.testing.assert_allclose(np.linalg.norm(path.normals,axis=1),1,atol=1e-12)


def test_budget_and_closed_surface_are_explicit():
    with pytest.raises(ValueError,match='refinement'):
        generate_toolpaths(square(),ToolpathConfig(spacing=.01,max_faces=100))
    closed=Surface([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    with pytest.raises(ValueError,match='boundary'):
        generate_toolpaths(closed)


def test_annular_paths_preserve_hole():
    count=32;angles=np.arange(count)*2*np.pi/count
    points=np.vstack([np.column_stack([r*np.cos(angles),r*np.sin(angles),np.zeros(count)]) for r in (1.,3.)])
    faces=[]
    for i in range(count):
        j=(i+1)%count
        faces.extend([[i,count+i,count+j],[i,count+j,j]])
    layer=Surface(points,faces)
    paths,_=generate_toolpaths(layer,ToolpathConfig(spacing=.25,waypoint_distance=.1))
    assert len(paths)>=4 and all(p.closed for p in paths)
    for path in paths:
        mid=(path.points[:-1]+path.points[1:])/2
        assert np.linalg.norm(mid[:,:2],axis=1).min()>1
        assert layer.distances(mid).max()<1e-10


def test_stress_sign_flips_do_not_reverse_infill():
    layer=square(6);layer.cell_ids=np.array([0,1])
    paths,report=generate_toolpaths(layer,ToolpathConfig(mode='hybrid',spacing=.5),
                                  stress=[[1.,0,0],[-1.,0,0]],stress_mask=[True,True])
    assert report['stress_angle_max_degrees']<1e-4
    assert any(p.kind=='stress' for p in paths)


def test_critical_stress_normal_to_layer_rejects_hybrid():
    with pytest.raises(ValueError,match='normal to'):
        generate_toolpaths(square(6),ToolpathConfig(mode='hybrid'),stress=[[0,0,1.]],stress_mask=[True])
