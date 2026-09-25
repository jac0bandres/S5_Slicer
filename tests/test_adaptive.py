import numpy as np
import pytest
from s3.mesh import TetMesh
from s3.adaptive import AdaptiveConfig, adaptive_layers, trim_inserted
from s3.surface import Surface,conform_surface


def test_height_layers_have_measured_spacing_and_report_uncovered_cap():
    mesh=TetMesh([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2,3]])
    layers,report=adaptive_layers(mesh,mesh.points[:,2],AdaptiveConfig(minimum=.2,maximum=.3))
    np.testing.assert_allclose([s.level for s in layers],[.2,.4,.6,.8])
    assert report['spacing_passed']
    assert not report['coverage_certified']
    np.testing.assert_allclose(report['remaining_scalar_range'],.2)


def test_trimming_removes_close_layer_and_keeps_separated_layer():
    points=np.array([[0.,0.,0.],[1,0,0],[0,1,0]])
    bottom=Surface(points,[[0,1,2]])
    cfg=AdaptiveConfig()
    close=Surface(points+[0,0,.1],[[0,1,2]],inserted=True)
    far=Surface(points+[0,0,.2],[[0,1,2]],inserted=True)
    removed,info=trim_inserted(close,[bottom],cfg)
    assert removed is None and info['unresolved_pieces']==0
    kept,info=trim_inserted(far,[bottom],cfg)
    assert len(kept.faces)==1 and info['removed_pieces']==0


def test_inserted_partial_layers_keep_source_cells():
    points=np.array([[0.,0,0],[2,0,0],[0,2,0],[0,0,1]])
    mesh=TetMesh(np.vstack([points,points+[10,0,0]]),[[0,1,2,3],[4,5,6,7]])
    scalar=np.r_[points[:,2],2*points[:,2]]
    layers,report=adaptive_layers(mesh,scalar,AdaptiveConfig(minimum=.2,maximum=.3))
    inserted=[layer for layer in layers if layer.inserted]
    assert inserted and report['spacing_passed']
    assert any(row.get('removed_pieces',0)>0 for row in report['trim_history'])
    assert all(layer.cell_ids is not None for layer in layers)
    assert any(np.all(layer.cell_ids==0) for layer in inserted)


def test_hanging_edges_are_conformed_without_false_boundaries():
    layer=Surface([[0,0,0],[2,0,0],[2,2,0],[0,2,0],[1,1,0]],
                  [[0,1,2],[0,4,3],[4,2,3]],cell_ids=[0,1,1])
    fixed=conform_surface(layer)
    from s3.toolpaths import topology
    edges,counts,_=topology(fixed)
    boundary=edges[counts==1]
    assert len(boundary)==4
    assert set(fixed.cell_ids)=={0,1}
    assert len(fixed.faces)==6
    np.testing.assert_allclose(fixed.points[:,2],0)


def test_first_layer_uses_physical_height_for_compressed_scalar():
    from s3.layers import first_bed_layer,fixed_layers
    mesh=TetMesh([[0,0,0],[1,0,0],[0,1,0],[0,0,2]],[[0,1,2,3]])
    scalar=mesh.points[:,2]*.1
    first,report=first_bed_layer(mesh,scalar,.2)
    np.testing.assert_allclose(first.points[:,2],.2,atol=1e-10)
    assert first.level==pytest.approx(.02)
    assert report['bed_spacing_passed']
    layers,_=fixed_layers(mesh,scalar,12,.2)
    assert len(layers)==12
    assert layers[0].points[:,2].max()<=.2+1e-10
    adaptive,report=adaptive_layers(mesh,scalar,AdaptiveConfig(minimum=.2,maximum=.3))
    assert adaptive[0].points[:,2].max()<=.2+1e-10
    assert report['first_layer']['maximum_z']<=.2+1e-10


def test_first_layer_limits_all_components_and_rejects_elevated_start():
    from s3.layers import first_bed_layer
    left=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,2.]])
    mesh=TetMesh(np.vstack([left,left+[10,0,0]]),[[0,1,2,3],[4,5,6,7]])
    field=np.r_[left[:,2],.1*left[:,2]]
    layer,_=first_bed_layer(mesh,field,.2)
    assert set(layer.cell_ids)=={0,1}
    assert layer.points[:,2].max()<=.2+1e-10
    elevated=TetMesh(np.vstack([left,left+[10,0,1]]),mesh.cells)
    with pytest.raises(ValueError,match='starts above'):
        first_bed_layer(elevated,field,.2)


def test_scaled_cantilever_first_layer_stays_below_height():
    from pathlib import Path
    from s3.mesh import read_tet
    from s3.placement import Placement
    from s3.pipeline import PaperConfig,run_paper
    from s3.layers import first_bed_layer
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    for scale in [1.,3.,4.]:
        placed=Placement(scale=scale).apply(mesh)
        result=run_paper(placed,PaperConfig(max_outer_iterations=5))
        first,report=first_bed_layer(placed,result['scalar'],.2)
        assert first.points[:,2].min()>=0
        assert first.points[:,2].max()<=.2+1e-10
        assert report['bed_spacing_passed']
