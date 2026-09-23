import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from s3.mesh import TetMesh
from s3.paper import (project_sphere_halfspaces,project_printing_direction,
                      blend_quaternions,paper_scale_system,solve_paper_scales)
from s3.pipeline import PaperConfig,Objectives,run_paper,concavity_weights
from s3.paper import QuaternionBlendWorkspace
from s3.numerics import from_two_vectors,from_two_vectors_batch


@pytest.fixture
def mesh():
    return TetMesh([[0,0,0],[2,0,0],[0,2,0],[0,0,2],[2,2,2]],[[0,1,2,3],[4,2,1,3]])


def test_sf_projection_analytic():
    d,cover=project_printing_direction([0,0,1],sf_normals=[[0,0,-1]],alpha=30)
    np.testing.assert_allclose(d[2],.5,atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(d),1)
    assert not cover


def test_sr_band_and_sq_union():
    d,_=project_printing_direction([0,0,1],stress=[0,0,1],beta=10)
    np.testing.assert_allclose(d[2],np.sin(np.deg2rad(10)),atol=1e-14)
    near_pole=np.array([.1,0,1]); near_pole/=np.linalg.norm(near_pole)
    d,cover=project_printing_direction(near_pole,sq_normal=[0,0,1])
    np.testing.assert_allclose(d,[0,0,1]);assert cover
    d,cover=project_printing_direction([1,0,0],sq_normal=[0,0,1])
    np.testing.assert_allclose(d,[1,0,0]);assert not cover


def test_hybrid_projection_beats_dense_sphere_search():
    rng=np.random.default_rng(6)
    sphere=rng.normal(size=(100000,3));sphere/=np.linalg.norm(sphere,axis=1)[:,None]
    sf=np.array([.2,.4,-.8]);sf/=np.linalg.norm(sf)
    stress=np.array([.5,-.3,.2]);stress/=np.linalg.norm(stress)
    feasible=sphere[(sphere@sf>=-.5)&(np.abs(sphere@stress)<=np.sin(np.deg2rad(10)))]
    for direction in [[0,0,1],[1,0,0],[-.3,.8,.1]]:
        direction=np.array(direction,dtype=float);direction/=np.linalg.norm(direction)
        d,_=project_printing_direction(direction,sf_normals=[sf],stress=stress)
        assert d@sf>=-.5-1e-10
        assert abs(d@stress)<=np.sin(np.deg2rad(10))+1e-10
        assert d@direction>=np.max(feasible@direction)-1e-10


def test_infeasible_halfspaces_are_explicit():
    with pytest.raises(ValueError,match='Empty'):
        project_sphere_halfspaces([0,0,1],[[1,0,0],[-1,0,0]],[.5,.5])


def test_eq8_pairwise_weighted_solution(mesh):
    rotations=np.tile(np.eye(3),(2,1,1))
    targets=Rotation.from_euler('z',[[0],[70]],degrees=True).as_matrix()
    keep=np.array([2.,3.]);edges=np.array([5.])
    actual=blend_quaternions(mesh,rotations,targets,keep,edges)
    qt=Rotation.from_matrix(targets).as_quat(scalar_first=True)
    q=np.linalg.solve(np.array([[7.,-5.],[-5.,8.]]),keep[:,None]*qt)
    expected=Rotation.from_quat(q,scalar_first=True).as_matrix()
    np.testing.assert_allclose(actual,expected,atol=1e-14)


def test_eq8_reuses_factor_and_invalidates_changed_constraints(mesh):
    workspace=QuaternionBlendWorkspace()
    rotations=Rotation.from_euler('z',[[20],[60]],degrees=True).as_matrix()
    targets=Rotation.from_euler('z',[[10],[90]],degrees=True).as_matrix()
    keep=np.array([2.,3.]); edges=np.array([5.])
    def solve(fixed=None):
        actual=blend_quaternions(mesh,rotations,targets,keep,edges,
                                 fixed_identity=fixed,workspace=workspace)
        expected=blend_quaternions(mesh,rotations,targets,keep,edges,fixed_identity=fixed)
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        return workspace.solver.factor
    factor=solve()
    targets[:]=Rotation.from_euler('x',[[30],[45]],degrees=True).as_matrix()
    assert solve() is factor
    keep[0]=10.  # Final SQ-C adjustment; arrays can be mutated in place.
    changed=solve()
    assert changed is not factor
    edges[0]=7.
    assert solve() is not changed
    factor=workspace.solver.factor
    assert solve([0]) is not factor
    assert solve([0,1]) is None  # Every variable pinned.
    keep[:]=0
    factor=solve()  # Gauge anchor on a component without targets.
    rotations[:]=Rotation.from_euler('y',[[40],[80]],degrees=True).as_matrix()
    assert solve() is factor  # Anchor values must still follow current rotations.


def test_batched_rotations_match_scalar():
    rng=np.random.default_rng(51)
    a=np.vstack([rng.normal(size=(100,3)),[0,0,1],[0,0,-1],[1e-8,0,-1]])
    b=np.array([0.,0.,1.])
    expected=np.array([from_two_vectors(v,b) for v in a])
    np.testing.assert_allclose(from_two_vectors_batch(a,b),expected,atol=1e-14)


def test_projection_fast_path_and_preferred_boundary_tie():
    direction=np.array([1.,0.,0.])
    np.testing.assert_array_equal(project_sphere_halfspaces(direction,[[1,0,0]],[.5]),direction)
    # A nearby boundary candidate is tied within tolerance and preferred.
    a=np.array([[1.,1e-6,0.]])
    a/=np.linalg.norm(a)
    b=[a[0,0]-1e-12]
    result=project_sphere_halfspaces(direction,a,b,preferred=[0,1,0])
    assert result[1]>0
    assert np.all(a@result>=np.array(b)-1e-10)


def test_eq12_residual_matches_written_energy(mesh):
    rng=np.random.default_rng(2)
    rotations=Rotation.random(2,random_state=rng).as_matrix()
    x=mesh.points+rng.normal(size=mesh.points.shape)*.2
    scales=rng.uniform(.5,1.5,(2,3))
    wr,wc=1.3,6.2
    a,b=paper_scale_system(mesh,rotations,wr,wc)
    computed=np.linalg.norm(a@np.r_[x.ravel(),scales.ravel()]-b)**2
    expected=0.
    for ci,cell in enumerate(mesh.cells):
        v=mesh.points[cell];v=v-v.mean(axis=0)
        xx=x[cell];xx=xx-xx.mean(axis=0)
        expected+=np.linalg.norm(xx.T-rotations[ci]@np.diag(scales[ci])@v.T)**2
    expected+=wr*np.linalg.norm(scales-1)**2+wc*np.linalg.norm(scales[0]-scales[1])**2
    np.testing.assert_allclose(computed,expected,rtol=1e-14)


def test_eq12_uniform_rigid_and_stationarity(mesh):
    rotations=np.tile(Rotation.from_euler('xyz',[30,20,160],degrees=True).as_matrix(),(2,1,1))
    x,s=solve_paper_scales(mesh,rotations)
    np.testing.assert_allclose(x,mesh.points@rotations[0].T,atol=1e-12)
    np.testing.assert_allclose(s,1,atol=1e-12)
    rotations[1]=Rotation.from_euler('xyz',[-10,5,30],degrees=True).as_matrix()
    x,s=solve_paper_scales(mesh,rotations)
    a,b=paper_scale_system(mesh,rotations)
    derivative=a.T@(a@np.r_[x.ravel(),s.ravel()]-b)
    np.testing.assert_allclose(derivative[3:],0,atol=1e-12)


@pytest.mark.parametrize('fixed',[None,[0,1,2],[0,1,2,3,4]])
def test_eq12_iterative_matches_direct(mesh,fixed):
    rotations=Rotation.from_euler('xyz',[[20,10,40],[-10,30,5]],degrees=True).as_matrix()
    direct=solve_paper_scales(mesh,rotations,fixed_vertices=fixed,solver='direct')
    messages=[]
    iterative=solve_paper_scales(mesh,rotations,fixed_vertices=fixed,solver='iterative',
                                tolerance=1e-11,progress=messages.append)
    for actual,expected in zip(iterative,direct):
        np.testing.assert_allclose(actual,expected,atol=1e-8)
    assert any('CG converged' in message for message in messages)
    if fixed is not None:np.testing.assert_array_equal(iterative[0][fixed],mesh.points[fixed])


def test_eq12_iterative_failure_is_explicit(mesh):
    rotations=Rotation.from_euler('xyz',[[20,10,40],[-10,30,5]],degrees=True).as_matrix()
    with pytest.raises(ValueError,match='did not converge'):
        solve_paper_scales(mesh,rotations,solver='iterative',max_iterations=1,tolerance=1e-14)


def test_eq12_checks_true_residual(mesh,monkeypatch):
    monkeypatch.setattr('s3.numerics.cg',lambda a,b,**kwargs: (np.zeros(len(b)),0))
    with pytest.raises(ValueError,match='did not converge'):
        solve_paper_scales(mesh,np.tile(np.eye(3),(2,1,1)),solver='iterative')


def test_eq12_iterative_disconnected_gauges(mesh):
    disconnected=TetMesh(np.vstack([mesh.points,mesh.points+[10,0,0]]),
                         np.vstack([mesh.cells,mesh.cells+len(mesh.points)]))
    rotations=Rotation.from_euler('z',[[10],[20],[30],[40]],degrees=True).as_matrix()
    direct=solve_paper_scales(disconnected,rotations,solver='direct')
    iterative=solve_paper_scales(disconnected,rotations,solver='iterative')
    np.testing.assert_allclose(iterative[0],direct[0],atol=1e-7)
    np.testing.assert_array_equal(iterative[0][disconnected.anchors],disconnected.points[disconnected.anchors])


def test_iterative_pipeline_matches_direct():
    from pathlib import Path
    from s3.mesh import read_tet
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    direct=run_paper(mesh,PaperConfig(max_outer_iterations=2,scale_solver='direct'))
    iterative=run_paper(mesh,PaperConfig(max_outer_iterations=2,scale_solver='iterative'))
    np.testing.assert_allclose(iterative['deformed'],direct['deformed'],atol=1e-6)
    assert iterative['stop_reason']==direct['stop_reason']
    assert iterative['build_plate']['below_plate_vertices']==0


@pytest.mark.parametrize('settings',[dict(scale_solver='other'),dict(scale_tolerance=0),
    dict(scale_max_iterations=0),dict(scale_max_iterations=1.5)])
def test_invalid_scale_solver_config(settings):
    with pytest.raises(ValueError):PaperConfig(**settings)


def test_height_transfer_and_objective_metric(mesh):
    # This analytic case measures the unconstrained paper objective.
    result=run_paper(mesh,PaperConfig(max_outer_iterations=3,fix_build_plate=False))
    np.testing.assert_array_equal(result['scalar'],result['deformed'][:,2])
    problem=Objectives(mesh,PaperConfig())
    pi,_=problem.metric(result['scalar'])
    assert pi<1e-8
    assert result['stop_reason']=='objective_tolerance'
    assert result['initial_pi']>pi


def test_progress_reports_work_before_outer_iteration_completes(mesh):
    events=[]
    rows=[]
    cfg=PaperConfig(max_outer_iterations=3,fix_build_plate=False)
    def completed(row):
        assert any('Eq. 7' in e['message'] for e in events)
        assert any('Eq. 8' in e['message'] for e in events)
        assert any('Eq. 12' in e['message'] for e in events)
        assert any('Factoring sparse matrix' in e['message'] and 'nonzeros' in e['message'] for e in events)
        assert any('Solving factored system' in e['message'] for e in events)
        rows.append(row)
    result=run_paper(mesh,cfg,callback=completed,progress_callback=events.append)
    assert rows==result['history']
    assert events[0]['message']=='Preparing build-plate constraints'
    assert any('Inner 1/7' in e['message'] and 'active cells' in e['message'] for e in events)
    assert events[-1]['message'].startswith('Finished · ')
    assert all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(events,events[1:]))
    baseline=run_paper(mesh,cfg)
    np.testing.assert_array_equal(result['deformed'],baseline['deformed'])
    assert result['stop_reason']==baseline['stop_reason']


def test_sr_requires_supplied_critical_region(mesh):
    with pytest.raises(ValueError,match='SR requires'):
        run_paper(mesh,PaperConfig(weight_sf=0,weight_sr=1))


def test_concavity_distinguishes_valley_and_ridge():
    # Shared face lies in x=0; scalar z-|x| has concave (valley) level sets.
    m=TetMesh([[-1,0,0],[0,-1,0],[0,1,0],[0,0,2],[1,0,0]],[[0,1,2,3],[4,2,1,3]])
    valley=m.points[:,2]-np.abs(m.points[:,0])
    ridge=m.points[:,2]+np.abs(m.points[:,0])
    wv,_=concavity_weights(m,valley,120,1)
    wr,_=concavity_weights(m,ridge,120,1)
    assert wv[0]>1
    assert wr[0]==1


def test_isolated_unconstrained_quaternion_component():
    m=TetMesh([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2,3]])
    rotations=np.array([Rotation.from_euler('xyz',[20,30,40],degrees=True).as_matrix()])
    actual=blend_quaternions(m,rotations,rotations,np.zeros(1),np.empty(0))
    np.testing.assert_allclose(actual,rotations,atol=1e-14)


def test_nonintegral_iteration_count_rejected():
    with pytest.raises(ValueError,match='integers'):
        PaperConfig(inner_iterations=1.5)


@pytest.mark.parametrize('weights',[(0.,1.,0.),(0.,0.,1.),(.3,.3,.4)])
def test_sr_sq_and_hybrid_pipeline(weights):
    m=TetMesh([[0,0,0],[2,0,0],[0,2,0],[0,0,2]],[[0,1,2,3]])
    cfg=PaperConfig(weight_sf=weights[0],weight_sr=weights[1],weight_sq=weights[2],
                    max_outer_iterations=4,fix_build_plate=False)
    kwargs=dict(sf_faces=np.array([0]),sq_faces=np.array([0]))
    if weights[1]:
        kwargs.update(stress=np.array([[1.,1.,1.]]),stress_mask=np.array([True]))
    result=run_paper(m,cfg,**kwargs)
    problem=Objectives(m,cfg,**kwargs)
    metric,_=problem.metric(result['scalar'])
    assert metric<1e-8
    assert result['stop_reason']=='objective_tolerance'
    if result['history']:
        assert result['history'][-1]['inverted_cells']==0


def test_cantilever_keeps_base_and_grows_from_it():
    from pathlib import Path
    from s3.mesh import read_tet
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    result=run_paper(mesh)
    base=result['build_plate']['base_vertices']
    np.testing.assert_allclose(result['deformed'][base],mesh.points[base],atol=1e-12)
    assert not result['build_plate']['floating_minimum_vertices']
    assert result['build_plate']['below_plate_vertices']==0
    assert result['history'][-1]['pi']<result['initial_pi']*.2
    assert all(row['min_determinant']>0 for row in result['history'])
    # Fixing three non-collinear base points rules out a nontrivial global
    # rigid motion; material away from the base must actually deform.
    assert np.linalg.norm(result['deformed']-mesh.points,axis=1).max()>1


def test_floating_minimum_plateaus_are_detected():
    from s3.pipeline import floating_minima
    from s3.mesh import read_tet
    from pathlib import Path
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    base=np.flatnonzero(mesh.points[:,2]==0)
    assert not floating_minima(mesh,mesh.points[:,2],base)
    field=mesh.points[:,2].copy()
    field[mesh.points[:,0]==30]=-1
    assert floating_minima(mesh,field,base)


def test_plate_normalization_preserves_nonzero_input_base():
    from s3.mesh import read_tet
    from pathlib import Path
    source=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    mesh=TetMesh(source.points+[0,0,17],source.cells)
    result=run_paper(mesh)
    base=result['build_plate']['base_vertices']
    np.testing.assert_allclose(result['deformed'][base],source.points[base],atol=1e-12)


def test_guided_projection_preserves_nearest_angle():
    direction=np.array([0.,0.,1.])
    projected,_=project_printing_direction(direction,sf_normals=[[0,0,-1]],preferred=[1,0,1])
    np.testing.assert_allclose(projected,[np.sqrt(3)/2,0,.5],atol=1e-12)


def test_fixed_base_backtracks_an_inverting_proposal(monkeypatch):
    from pathlib import Path
    from s3.mesh import read_tet
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    def inverted(mesh,*args,**kwargs):
        points=mesh.points.copy();points[:,2]*=-1
        return points,np.ones((len(mesh.cells),3))
    monkeypatch.setattr('s3.pipeline.solve_paper_scales',inverted)
    result=run_paper(mesh,PaperConfig(max_outer_iterations=1))
    assert result['history'][0]['step_fraction']==.25
    assert result['history'][0]['min_determinant']>0
    assert not result['build_plate']['floating_minimum_vertices']
    assert result['build_plate']['below_plate_vertices']==0
