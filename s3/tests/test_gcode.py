"""Validate emitted motion, filament volume, and all-or-nothing preflight."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pytest
from s3.gcode import GCodeConfig, render_gcode, write_gcode
from s3.machine import S5Machine
from s3.toolpaths import Toolpath


def stroke(points, closed=False, normals=None):
    points = np.asarray(points, dtype=float)
    return Toolpath(points, np.tile([0., 0., 1.], (len(points), 1)) if normals is None else normals,
                    'contour', closed, 0.)


def moves(text):
    return [dict((axis, float(value)) for axis, value in re.findall(r'([XZBCEF])(-?[\d.]+)', line))
            for line in text.splitlines() if line.startswith('G1 X') and ' F' in line]


def test_closed_stroke_extrusion_and_inverse_time():
    curve = stroke([[10, 0, .2], [13, 0, .2], [13, 4, .2]], closed=True)
    cfg = GCodeConfig(print_speed=10)
    text, report = render_gcode([[curve]], cfg)
    deposition = [m for m in moves(text) if 'E' in m]
    expected = np.array([[13, 0, .2], [13, 4, .2], [10, 0, .2]])
    actual = [S5Machine().forward([m[a] for a in 'XZBC'])[0] for m in deposition]
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert report['deposition_length'] == pytest.approx(12.)
    volume = sum(m['E'] for m in deposition) * np.pi * (.875 ** 2)
    assert volume == pytest.approx(12 * .4 * .2, abs=1e-7)
    assert all(60 / m['F'] >= length / 10 - 1e-8 for m, length in zip(deposition, [3, 4, 5]))
    assert 'G93' in text and 'M83' in text
    assert text.endswith('G94\nM104 S0\nM140 S0\n')


def test_disconnected_strokes_retract_lift_and_never_extrude_travel():
    paths = [[stroke([[10, 0, 1], [11, 0, 1]]), stroke([[20, 5, 2], [21, 5, 2]])]]
    text, report = render_gcode(paths,GCodeConfig(layer_height=2.))
    second = text.split(';LAYER:0 STROKE:1')[1]
    commands = moves(second)
    assert commands[0]['Z'] == report['travel_actuator_z']
    assert commands[1]['Z'] == report['travel_actuator_z']
    assert all('E' not in m for m in commands[:3])
    assert 'G1 E-1.00000 F1500.00000' in second
    assert 'G1 E1.00000 F1500.00000' in second
    assert report['deposition_length'] == pytest.approx(2)


def test_tilt_compensation_and_yaw_seam():
    machine = S5Machine()
    expected = np.array([[60., 5., -45., 179.], [60., 5., -45., 181.]])
    points, normals = zip(*(machine.forward(p) for p in expected))
    text, _ = render_gcode([[stroke(points, normals=np.array(normals))]],GCodeConfig(layer_height=100.))
    last = next(m for m in moves(text) if 'E' in m)
    np.testing.assert_allclose([last[a] for a in 'XZBC'], expected[1], atol=1e-5)


def test_preflight_preserves_existing_file_on_unreachable_normal(tmp_path):
    target = tmp_path/'print.gcode'
    target.write_text('existing')
    invalid = stroke([[10, 0, 1], [11, 0, 1]], normals=np.array([[0, 0, 1], [0, 1, 1]]))
    with pytest.raises(ValueError, match='Layer 1, stroke 1, waypoint 2.*unreachable'):
        write_gcode(target, [[invalid]], GCodeConfig(enforce_machine_limits=True))
    assert target.read_text() == 'existing'


@pytest.mark.parametrize('curves', [[],
                                   [stroke([[0, 0, 0], [0, 0, 0]])],
                                   [stroke([[0, 0, 0], [np.nan, 0, 1]])]])
def test_invalid_geometry(curves):
    with pytest.raises(ValueError):
        render_gcode([curves])


@pytest.mark.parametrize('settings', [dict(print_speed=0), dict(flow_multiplier=-1),
                                       dict(travel_clearance=0), dict(rotary_speed=np.inf)])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        GCodeConfig(**settings)


@pytest.mark.skipif(not Path('/usr/bin/CuraEngine').exists(),reason='CuraEngine not installed')
def test_cli_generates_gcode_and_report(tmp_path):
    pytest.importorskip('pyvista')
    config = tmp_path/'paper.json'
    config.write_text(json.dumps(dict(objective_tolerance=1e6)))
    output = tmp_path/'output'
    result = subprocess.run([sys.executable, '-m', 's3',
        str(Path(__file__).parent/'fixtures/cantilever.tet'), '-o', str(output),
        '--config', str(config), '--layer-height', '.5', '--gcode'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'G93' in (output/'print.gcode').read_text()
    report = json.loads((output/'report.json').read_text())
    assert report['gcode']['layers'] > 12 and not report['print_ready']


def test_custom_scripts_and_disabled_thermal_commands():
    paths = [[stroke([[10, 0, 1], [11, 0, 1]])]]
    text, _ = render_gcode(paths, GCodeConfig(layer_height=1.), start_gcode='G28\nM82', end_gcode='M400')
    assert 'M82\nG21\nG90\nM83\nG94' in text
    assert 'M109' not in text and text.endswith('G94\nM400\n')


def test_export_does_not_gate_on_bed_height(tmp_path):
    target=tmp_path/'print.gcode'
    floating=stroke([[10,0,2.],[11,0,2.]])
    report=write_gcode(target,[[],[floating]],GCodeConfig(layer_height=.2))
    assert 'G93' in target.read_text()
    assert report['first_layer_maximum_z']==pytest.approx(2.)
    below=stroke([[10,0,-1.],[11,0,-1.]])
    text,report=render_gcode([[below]])
    assert 'Z-1.00000' in text


def test_default_export_projects_normals_and_does_not_enforce_b_limits():
    curve=stroke([[10,0,.2],[11,0,.2]],normals=np.array([[0,1,1],[1,0,0]]))
    text,report=render_gcode([[curve]])
    assert not report['machine_limits_enforced']
    assert report['orientation_mismatch_waypoints']==1
    assert report['maximum_orientation_error_degrees']==pytest.approx(45.)
    assert report['outside_b_limits_waypoints']==1
    last=next(m for m in moves(text) if 'E' in m)
    assert last['B']==pytest.approx(90.)
    position,_=S5Machine().forward([last[a] for a in 'XZBC'])
    np.testing.assert_allclose(position,[11,0,.2],atol=1e-5)
