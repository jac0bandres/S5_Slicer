import numpy as np
from s3.surface import Surface,point_triangle_squared,separation_bounds,triangle_triangle_squared


def test_point_triangle_regions():
    tri=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]])
    for point,distance2 in [([.2,.2,2],4),([2,0,0],1),([1,1,0],.5),([-.1,-.2,0],.05)]:
        np.testing.assert_allclose(point_triangle_squared(point,tri),[distance2],atol=1e-14)


def test_nearest_surface_matches_exhaustive():
    rng=np.random.default_rng(42)
    triangles=rng.normal(size=(50,3,3))
    s=Surface(triangles.reshape(-1,3),np.arange(150).reshape(-1,3))
    for point in rng.normal(size=(20,3)):
        d,_=s.nearest(point)
        np.testing.assert_allclose(d,np.sqrt(point_triangle_squared(point,triangles).min()),atol=1e-14)


def test_parallel_surface_distance_bounds():
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    bottom=Surface(points,[[0,1,2]])
    top=Surface(points+[0,0,.2],[[0,1,2]])
    result=separation_bounds(top,bottom,tolerance=1e-8)
    assert result['resolved']
    for key in ['min_lower','min_upper','max_lower','max_upper']:
        np.testing.assert_allclose(result[key],.2,atol=1e-14)


def test_unresolved_bounds_remain_explicit():
    bottom=Surface([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
    top=Surface([[.3,-.2,.1],[.5,.6,.4],[1.3,.2,.2]],[[0,1,2]])
    result=separation_bounds(top,bottom,tolerance=1e-14,max_refinements=0)
    assert not result['resolved']
    rng=np.random.default_rng(3)
    weights=rng.dirichlet([1,1,1],1000)
    samples=bottom.distances(weights@top.points)
    assert result['min_lower']<=samples.min()+1e-12
    assert result['max_upper']>=samples.max()-1e-12


def test_threshold_decisions_do_not_require_precise_extrema():
    bottom=Surface([[0,0,0],[2,0,0],[0,2,0]],[[0,1,2]])
    top=Surface([[0,0,.2],[2,0,.3],[0,2,.2]],[[0,1,2]])
    result=separation_bounds(top,bottom,tolerance=1e-10,minimum_threshold=.15,maximum_threshold=.4)
    assert result['decision_resolved'] and result['refinements']==0
    assert result['min_lower']>=.15 and result['max_upper']<=.4


def test_triangle_minimum_interior_piercing_and_skew_edges():
    base=np.array([[0.,0,0],[2,0,0],[0,2,0]])
    piercing=np.array([[.3,.3,-1],[.3,.3,1],[.5,.3,1]])
    parallel=base+[0,0,.2]
    np.testing.assert_allclose(triangle_triangle_squared(base,np.array([piercing,parallel])),[0,.04],atol=1e-12)
    # Closest points lie in the interiors of perpendicular edges.
    a=np.array([[-1.,0,0],[1,0,0],[0,-1,0]])
    b=np.array([[0.,-1,1],[0,1,1],[0,0,2]])
    np.testing.assert_allclose(triangle_triangle_squared(a,b[None]),[1.],atol=1e-12)


def test_triangle_minimum_is_symmetric_and_bounds_dense_samples():
    rng=np.random.default_rng(271)
    source=rng.normal(size=(3,3));targets=rng.normal(size=(12,3,3))
    actual=triangle_triangle_squared(source,targets)
    inverse=np.array([triangle_triangle_squared(t,source[None])[0] for t in targets])
    np.testing.assert_allclose(actual,inverse,atol=1e-12)
    samples=rng.dirichlet([1,1,1],200)@source
    sampled=np.min([point_triangle_squared(p,targets) for p in samples],axis=0)
    assert np.all(actual<=sampled+1e-12)
