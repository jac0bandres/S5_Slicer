"""Cura owns path planning; reformation preserves its motion/extrusion semantics."""
import re
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('pyvista')
from s3.cura import CuraConfig,TetReformer,reform_gcode,slice_with_cura
from s3.machine import S5Machine


def tetra():
    return np.array([[0.,0,0],[10,0,0],[0,10,0],[0,0,10]]),np.array([[0,1,2,3]])


def motion(text):
    return [dict((k,float(v)) for k,v in re.findall(r'([XZBCEF])(-?[\d.]+)',line))
            for line in text.splitlines() if line.startswith('G1 X')]


def test_affine_barycentric_mapping_including_exterior_points():
    original,cells=tetra()
    transform=np.array([[1.2,.1,0],[0,.8,.1],[.2,0,1.4]])
    offset=np.array([3.,-2.,4.])
    reformer=TetReformer(original,original@transform.T+offset,cells)
    expected=np.array([[1.,1.,1.],[2.,1.,3.],[12.,2.,1.]])
    actual,_,ratio,outside=reformer.map(expected@transform.T+offset)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    np.testing.assert_allclose(ratio,1/np.linalg.det(transform),atol=1e-12)
    np.testing.assert_array_equal(outside,[False,False,True])


def test_modal_extrusion_retractions_g92_and_relative_coordinates():
    original,cells=tetra()
    reformer=TetReformer(original,original,cells)
    raw='''G21
G90
M82
G92 E10
M104 S210
;LAYER:0
G0 X1 Y1 Z1 F600
;TYPE:WALL-OUTER
G1 X3 E12
G1 E11 F1200
G1 E12
G92 E0
G91
M83
G1 X1 E.5 F600
G1 E-1 F1500
G1 E1
M106 S128
M104 S0
'''
    text,report=reform_gcode(raw,reformer,CuraConfig(segment_length=.4))
    assert 'M104 S210' in text and ';TYPE:WALL-OUTER' in text and 'M106 S128' in text
    assert 'M82' not in text and 'G91' not in text
    assert text.count('G1 E-1.00000000')==2
    assert text.count('G1 E1.00000000')==2
    assert report['source_deposition_filament']==pytest.approx(2.5)
    assert report['reformed_deposition_filament']==pytest.approx(2.5)
    depositing=[m for m in motion(text) if m.get('E',0)>0]
    position,_=S5Machine().forward([depositing[-1][a] for a in 'XZBC'])
    np.testing.assert_allclose(position,[4,1,1],atol=1e-5)
    assert sum(1/m['F'] for m in depositing)==pytest.approx(3/600)


def test_volume_correction_does_not_scale_unretractions():
    original,cells=tetra()
    reformer=TetReformer(original,original*2,cells)
    text,report=reform_gcode('M83\nG0 X1 Y1 Z1 F600\nG1 X3 E2\nG1 E-1\nG1 E1',
                             reformer,CuraConfig(segment_length=.4))
    assert report['reformed_deposition_filament']==pytest.approx(.25)
    assert 'G1 E-1.00000000' in text and 'G1 E1.00000000' in text


def box_result(height=1.):
    points=np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,0],
                     [0,0,height],[10,0,height],[0,10,height],[10,10,height]],float)
    cells=np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],[0,6,4,7],[0,4,5,7],[0,5,1,7]])
    return dict(points=points,cells=cells,deformed=points.copy(),scalar=points[:,2].copy())


@pytest.mark.skipif(not Path('/usr/bin/CuraEngine').exists(),reason='CuraEngine not installed')
def test_real_cura_layers_and_extruder_diameter(tmp_path):
    cfg=CuraConfig(layer_height=.2,first_layer_height=.2)
    text,report=slice_with_cura(box_result(),tmp_path/'small',cfg)
    assert report['layers']==5
    assert ';TYPE:WALL-OUTER' in text and ';TYPE:SKIN' in text
    assert 'M190 S55' in text
    assert (tmp_path/'small/cura_deformed.gcode').exists()
    assert (tmp_path/'small/cura.log').exists()
    assert report['source_deposition_filament']==pytest.approx(report['reformed_deposition_filament'])
    _,larger=slice_with_cura(box_result(),tmp_path/'large',replace(cfg,filament_diameter=3.5))
    assert report['source_deposition_filament']/larger['source_deposition_filament']==pytest.approx(4.,rel=.001)
    depositing=[m for m in motion(text) if m.get('E',0)>0]
    p,_=S5Machine().forward([depositing[0][a] for a in 'XZBC'])
    assert p[2]==pytest.approx(.2,abs=1e-5)
