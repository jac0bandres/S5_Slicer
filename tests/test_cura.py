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


def test_first_layer_extrusion_uses_physical_bead_dimensions():
    class DeformedReformer:
        def map(self, points):
            points=np.asarray(points,dtype=float).copy()
            points[:,0]*=2
            return points,np.zeros(len(points)),np.full(len(points),.25),np.zeros(len(points),bool)

    raw='''G90
M83
;LAYER:0
G0 X1 Y0 Z0.1 F600
G1 X2 E0.01
;LAYER:1
G0 Z0.2
G1 X3 E0.01
'''
    text,report=reform_gcode(raw,DeformedReformer(),
        CuraConfig(segment_length=1.,first_layer_height=.2),source_first_layer_height=.1)
    first,second=text.split(';LAYER:1')
    assert sum(m.get('E',0) for m in motion(first))==pytest.approx(.02)
    assert sum(m.get('E',0) for m in motion(second))==pytest.approx(.0025)
    assert report['source_first_layer_height']==pytest.approx(.1)


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
    assert report['cura_settings']['retraction_hop_enabled'] is False
    assert report['source_deposition_filament']==pytest.approx(report['reformed_deposition_filament'])
    _,larger=slice_with_cura(box_result(),tmp_path/'large',replace(cfg,filament_diameter=3.5))
    assert report['source_deposition_filament']/larger['source_deposition_filament']==pytest.approx(4.,rel=.001)
    depositing=[m for m in motion(text) if m.get('E',0)>0]
    p,_=S5Machine().forward([depositing[0][a] for a in 'XZBC'])
    assert p[2]==pytest.approx(.2,abs=1e-5)


@pytest.mark.skipif(not Path('/usr/bin/CuraEngine').exists(),reason='CuraEngine not installed')
def test_postprocessed_brim_is_flat_and_precedes_first_layer(tmp_path):
    cfg = CuraConfig(brim_width=1.2, brim_gap=0.1, z_hop=0.5)
    text, report = slice_with_cura(box_result(), tmp_path/'brim', cfg)
    assert report['cura_settings']['adhesion_type'] == 'none'
    assert report['brim_loops'] >= 3
    assert report['brim_filament'] > 0
    assert ';TYPE:SKIRT-BRIM' not in (tmp_path/'brim/cura_deformed.gcode').read_text()
    brim = text.split('; S3 physical bed brim', 1)[1].split(';TYPE:', 1)[0]
    deposits = [row for row in motion(brim) if row.get('E', 0) > 0]
    assert deposits
    assert 'G1 E1.00000000' in brim and 'G1 E-1.00000000' in brim
    tips = np.array([S5Machine().forward([row[a] for a in 'XZBC'])[0] for row in deposits])
    np.testing.assert_allclose(tips[:, 2], cfg.first_layer_height, atol=1e-5)
    assert all(row['B'] == 0 for row in deposits)
    assert text.index('; S3 physical bed brim') < text.index(';TYPE:WALL-OUTER')


class IslandReformer:
    """Known endpoint poses with deliberately unusable extrapolation in the gap."""
    def map(self, points):
        points = np.asarray(points).copy()
        outside = (points[:, 0] > 12) & (points[:, 0] < 28)
        tilts = np.where(points[:, 0] >= 28, np.deg2rad(60.), np.deg2rad(20.))
        points[outside, 2] = 1000.
        tilts[outside] = 1000.
        return points, tilts, np.ones(len(points)), outside


@pytest.mark.parametrize('travel_command', ['G0', 'G1'])
def test_island_bridge_ignores_air_mapping_and_limits_travel_speed(travel_command):
    raw = f'''G90
M83
G0 X10 Y0 Z8 F1200
G1 X11 E.1
G1 Z2 E.1
G1 E-.7 F1500
; BRIDGE
{travel_command} X30 F7200
G1 E.7 F1500
; PRINT
G1 X31 E.1 F1200
'''
    cfg = CuraConfig(segment_length=.5)
    text, report = reform_gcode(raw, IslandReformer(), cfg)
    before, rest = text.split('; BRIDGE')
    bridge = rest.split('; PRINT')[0]
    rows = motion(before)[-1:] + motion(bridge)
    poses = np.array([[r[a] for a in 'XZBC'] for r in rows])
    seconds = np.array([60/r['F'] for r in rows[1:]])
    rates = np.abs(np.diff(poses, axis=0))/seconds[:, None]
    assert rates[:, :2].max() <= cfg.travel_speed + .01
    assert rates[:, 2:].max() <= cfg.rotary_speed + .01
    assert np.abs(np.diff(poses[:, 2])).max() <= 1.00002
    tips = np.array([S5Machine().forward(p)[0] for p in poses])
    assert tips[:, 2].max() < 10  # Poisoned exterior Z must never reach output.
    # Traversal of the gap stays above the highest deposited point (Z=8).
    crossing = (tips[:, 0] > 12) & (tips[:, 0] < 27)
    assert crossing.any()
    assert np.abs(poses[crossing, 2]).max() <= 45.00001
    assert tips[crossing, 2].min() >= 8
    assert all(r.get('E', 0) == 0 for r in motion(bridge))
    assert report['air_travel_bridges'] == 1
    assert report['air_travel_samples_replaced'] > 20
    assert report['source_deposition_filament'] == pytest.approx(.3)
    assert sum(r.get('E', 0) for r in motion(text)) == pytest.approx(.3, abs=1e-6)
    # Arrive and remove the hop before unretracting, preserving the print start.
    np.testing.assert_allclose(tips[-1], [30, 0, 2], atol=1e-5)


def test_compensated_hop_tracks_partial_retractions_and_returns_before_prime():
    raw = '''G90
M83
G0 X10 Y0 Z2 F1200
G1 X11 E.1
; HOP
G1 E-.7 F1500
G1 E.2
; PARTIAL
G1 E.5
; DONE
'''
    text, _ = reform_gcode(raw, IslandReformer())
    before, rest = text.split('; HOP')
    partial, rest = rest.split('; PARTIAL')
    restored = rest.split('; DONE')[0]
    original = motion(before)[-1]
    lifted = motion(partial)
    assert len(lifted) == 1  # Partial repayment must leave the hop in place.
    b = np.deg2rad(original['B'])
    assert lifted[0]['X']-original['X'] == pytest.approx(-np.sin(b), abs=1e-5)
    assert lifted[0]['Z']-original['Z'] == pytest.approx(np.cos(b), abs=1e-5)
    lowered = motion(restored)[-1]
    np.testing.assert_allclose([lowered[a] for a in 'XZBC'], [original[a] for a in 'XZBC'], atol=1e-5)
    assert restored.index('G1 X') < restored.index('G1 E0.50000000')


def test_exterior_travel_endpoint_is_preserved_at_end_of_file():
    original, cells = tetra()
    text, report = reform_gcode('M83\nG0 X1 Y1 Z1\nG1 X2 E.1\nG0 X20',
                               TetReformer(original, original, cells))
    last = motion(text)[-1]
    point, _ = S5Machine().forward([last[a] for a in 'XZBC'])
    np.testing.assert_allclose(point, [20, 1, 1], atol=1e-5)
    assert report['air_travel_bridges'] == 1


def test_center_crossing_is_retimed_for_rotary_axis():
    original, cells = tetra()
    text, _ = reform_gcode('G90\nG0 X-5 Y0 Z1 F7200\n; CROSS\nG0 X5',
                           TetReformer(original, original, cells))
    before, crossing = text.split('; CROSS')
    rows = motion(before)[-1:] + motion(crossing)
    poses = np.array([[r[a] for a in 'XZBC'] for r in rows])
    seconds = np.array([60/r['F'] for r in rows[1:]])
    assert (np.abs(np.diff(poses[:, 3]))/seconds).max() <= 30.001
    np.testing.assert_allclose(S5Machine().forward(poses[-1])[0], [5, 0, 1], atol=1e-5)


@pytest.mark.parametrize('kwargs', [dict(z_hop=-1), dict(z_hop=np.nan),
                                    dict(travel_speed=0), dict(rotary_speed=np.inf),
                                    dict(brim_width=-1), dict(brim_gap=np.nan)])
def test_invalid_travel_configuration(kwargs):
    with pytest.raises(ValueError):
        CuraConfig(**kwargs)
