"""Reproduce the diagnostic Squirtle figures in this directory.

Run from the repository root with a Python environment containing the UI/Cura
dependencies plus matplotlib. CuraEngine 5.0.0 and its definitions are needed
for the reformation comparison.

    .venv-s3/bin/python -m pip install matplotlib
    .venv-s3/bin/python docs/figures/render_squirtle.py
"""
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import pyvista as pv
import trimesh

from s3.cura import CuraConfig, slice_with_cura
from s3.machine import S5Machine
from s3.mesh import TetMesh
from s3.pipeline import PaperConfig, run_paper
from s3.placement import Placement
from s3.stl_import import tetrahedralize
from s3.toolpaths import ToolpathConfig
from s3.visual_pipeline import build_visualization


OUT = Path(__file__).resolve().parent
STL = ROOT / 'research/s5/input_models/Squirtle.stl'
COLORS = ['#f59e0b', '#e879f9', '#38bdf8', '#34d399', '#f97373', '#a3e635', '#fbbf24']
WORD = re.compile(r'([XYZBCEF])(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')


def surface(ax, points, faces, color, alpha=.9):
    xyz = points[faces]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    size = np.linalg.norm(normals, axis=1)
    normals = normals / np.maximum(size[:, None], 1e-12)
    light = np.array([.3, -.6, 1.])
    light /= np.linalg.norm(light)
    shade = .54 + .38 * np.abs(normals @ light)
    rgb = np.array(matplotlib.colors.to_rgb(color))
    colors = np.column_stack([np.clip(rgb[None, :] * shade[:, None], 0, 1),
                              np.full(len(faces), alpha)])
    collection = Poly3DCollection(xyz, facecolors=colors, edgecolors='none',
                                  linewidths=0, zsort='average')
    ax.add_collection3d(collection)


def frame(ax, points, *, title=None, show_axes=False):
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(high-low) * .64
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(max(-2, center[2]-radius), center[2]+radius))
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev=22, azim=-61)
    ax.set_proj_type('ortho')
    if show_axes:
        ax.set_xlabel('X / mm'); ax.set_ylabel('Y / mm'); ax.set_zlabel('Z / mm')
    else:
        ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13, color='#17202e', pad=4, weight='semibold')


def save(fig, name):
    fig.patch.set_facecolor('white')
    fig.savefig(OUT / name, dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.18)
    plt.close(fig)


def render_deformation(result, faces):
    fig = plt.figure(figsize=(11.2, 5.6))
    for index, (points, title, color) in enumerate([
        (result['points'], 'Input volume mesh', '#64748b'),
        (result['deformed'], 'S³ deformed geometry', '#06b6d4'),
    ]):
        ax = fig.add_subplot(1, 2, index+1, projection='3d')
        surface(ax, points, faces, color)
        frame(ax, result['points'], title=title)
    save(fig, 'squirtle_deformation.png')


def draw_lines(ax, segments, points, faces, color, title):
    surface(ax, points, faces, '#94a3b8', alpha=.11)
    if segments:
        ax.add_collection3d(Line3DCollection(np.asarray(segments), colors=color,
                            linewidths=.75, alpha=.86))
    frame(ax, points, title=title)


def cura_segments(text):
    pos = np.zeros(3); layer = -1; result = []
    for line in text.splitlines():
        if line.startswith(';LAYER:'):
            layer = int(line.split(':', 1)[1]); continue
        if not line.startswith(('G0 ', 'G1 ')):
            continue
        words = dict((name, float(value)) for name, value in WORD.findall(line.split(';')[0]))
        target = pos.copy()
        for i, axis in enumerate('XYZ'):
            if axis in words: target[i] = words[axis]
        if words.get('E', 0) > 0 and layer >= 0 and not np.allclose(pos, target):
            result.append((layer, [pos.copy(), target.copy()]))
        pos = target
    return result


def reformed_segments(text):
    machine = S5Machine()
    pose = None; layer = -1; result = []
    for line in text.splitlines():
        if line.startswith(';LAYER:'):
            layer = int(line.split(':', 1)[1]); continue
        if not line.startswith('G1 X'):
            continue
        words = dict((name, float(value)) for name, value in WORD.findall(line.split(';')[0]))
        if not all(name in words for name in 'XZBC'):
            continue
        target = np.array([words[name] for name in 'XZBC'])
        if pose is not None and words.get('E', 0) > 0 and layer >= 0:
            a = machine.forward(pose)[0]
            b = machine.forward(target)[0]
            result.append((layer, [a, b]))
        pose = target
    return result


def render_reformation(result, faces, raw, reformed, report):
    offset = np.asarray(report['deformed_offset'])
    deformed = result['deformed'] - offset
    physical = result['points']
    raw_lines = cura_segments(raw)
    physical_lines = reformed_segments(reformed)
    # Display every fourth Cura layer so the individual paths remain visible.
    selected = set(range(0, int(report['layers']), 4))
    fig = plt.figure(figsize=(11.2, 5.6))
    for index, (lines, points, title) in enumerate([
        (raw_lines, deformed, 'Cura paths on deformed geometry'),
        (physical_lines, physical, 'Reformed nozzle-tip paths'),
    ]):
        ax = fig.add_subplot(1, 2, index+1, projection='3d')
        segments = [segment for layer, segment in lines if layer in selected]
        colors = [COLORS[(layer//4) % len(COLORS)] for layer, _ in lines if layer in selected]
        draw_lines(ax, segments, points, faces, colors, title)
    save(fig, 'squirtle_reformation.png')
    print(f'Cura layers: {report["layers"]}; deposition segments: '
          f'{len(raw_lines)} source, {len(physical_lines)} reformed')


def render_contours(result, faces, layers, paths):
    fig = plt.figure(figsize=(7.8, 6.2))
    ax = fig.add_subplot(111, projection='3d')
    surface(ax, result['points'], faces, '#64748b', alpha=.10)
    for index in range(len(layers)):
        for curve in paths[index]:
            points = curve.points
            if len(points) < 2:
                continue
            segments = np.stack([points[:-1], points[1:]], axis=1)
            ax.add_collection3d(Line3DCollection(segments, colors=COLORS[index % len(COLORS)],
                                linewidths=1.2, alpha=.9))
    frame(ax, result['points'], title='Diagnostic curved-layer contour paths')
    save(fig, 'squirtle_contours.png')


def main():
    original = trimesh.load(STL, force='mesh')
    pv_surface = pv.PolyData(np.asarray(original.vertices),
                             np.column_stack([np.full(len(original.faces), 3),
                                              original.faces]).ravel())
    reduced = pv_surface.decimate(.88, volume_preservation=True)
    reduced_faces = np.asarray(reduced.faces).reshape(-1, 4)[:, 1:]
    reduced_stl = trimesh.Trimesh(vertices=np.asarray(reduced.points),
                                  faces=reduced_faces, process=False).export(file_type='stl')
    mesh = tetrahedralize(reduced_stl)
    mesh = Placement(scale=.5, center_xy=True, drop_to_bed=True).apply(mesh)
    result = run_paper(mesh, PaperConfig(inner_iterations=3, max_outer_iterations=4),
                       callback=lambda row: print('outer', row['outer'], 'Pi', row['pi']))
    boundary = TetMesh(result['points'], result['cells'])
    faces = boundary.faces[boundary.boundary]
    render_deformation(result, faces)
    layers, paths, report = build_visualization(result, count=9,
        toolpath_config=ToolpathConfig(spacing=1.3, waypoint_distance=1.2),
        first_layer_height=.4)
    render_contours(result, faces, layers, paths)
    cfg = CuraConfig(layer_height=.8, first_layer_height=.4, line_width=.6,
                     wall_count=1, infill_density=0, segment_length=1.2)
    with tempfile.TemporaryDirectory(prefix='s3-squirtle-') as directory:
        reformed, cura_report = slice_with_cura(result, directory, cfg,
                                                settings={'top_layers': 1, 'bottom_layers': 1})
        raw = (Path(directory) / 'cura_deformed.gcode').read_text()
        render_reformation(result, faces, raw, reformed, cura_report)
    print('S³ stop:', result['stop_reason'], 'worst angle:',
          result['history'][-1]['worst_angle_degrees'],
          'contour curves:', report['total_curves'])


if __name__ == '__main__':
    main()
