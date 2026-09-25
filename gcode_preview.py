"""Workbench previews for requested paths and the S5 writer's emitted motion."""
import re
import numpy as np
import plotly.graph_objects as go
from .machine import S5Machine
from .visualization import layout


def compatibility_preview(paths, machine=None):
    """Show requested geometry even when a machine pose cannot be generated."""
    machine = machine or S5Machine()
    lines = []
    failures = []
    bad_points = []
    previous_c = 0.
    count = 0
    for li, curves in enumerate(paths):
        for ci, curve in enumerate(curves):
            points = np.asarray(curve.points)
            lines.extend(points.tolist())
            if curve.closed and len(points):
                lines.append(points[0].tolist())
            lines.append([None, None, None])
            for wi, (point, normal) in enumerate(zip(points, curve.normals)):
                count += 1
                try:
                    if point[2] < 0:
                        raise ValueError('Point is below the bed (Z=0)')
                    pose = machine.inverse(point, normal, previous_c=previous_c)
                    previous_c = pose[3]
                except ValueError as exc:
                    failures.append(dict(layer=li+1, stroke=ci+1, waypoint=wi+1, reason=str(exc)))
                    bad_points.append(point)
    fig = go.Figure()
    xyz = np.asarray(lines, dtype=object).reshape(-1, 3)
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines',
        name='Requested deposition paths', line=dict(color='#f59e0b', width=3), connectgaps=False))
    if bad_points:
        xyz = np.asarray(bad_points)
        labels = [f'Layer {f["layer"]}, stroke {f["stroke"]}, waypoint {f["waypoint"]}: {f["reason"]}' for f in failures]
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
            name='Exact-pose mismatch', marker=dict(color='#ef4444', size=3), text=labels, hoverinfo='text'))
    fig.update_layout(title='Requested paths and S5 compatibility (not generated G-code)')
    return layout(fig), dict(waypoints=count, failed_waypoints=len(failures), failures=failures)


def gcode_motion_preview(text, machine=None):
    """Reconstruct this writer's explicit X/Z/B/C G1 moves in bed coordinates.

    Samples linear joint interpolation, including travel, rather than plotting
    actuator X as Cartesian X. Startup before a complete pose is known is omitted.
    This is a viewer for our generated dialect, not an arbitrary G-code parser.
    """
    machine = machine or S5Machine()
    axes = {}
    previous = None
    deposition, travel = [], []
    for line in text.splitlines():
        code = line.split(';', 1)[0].strip()
        if not code.startswith('G1 '):
            continue
        words = {a: float(v) for a, v in re.findall(r'([XZBCE])(-?\d+(?:\.\d+)?)', code)}
        if not any(a in words for a in 'XZBC'):
            continue
        axes.update({a: words[a] for a in 'XZBC' if a in words})
        if not all(a in axes for a in 'XZBC'):
            continue
        pose = np.array([axes[a] for a in 'XZBC'])
        if previous is not None:
            samples = min(181, max(2, int(np.ceil(np.max(np.abs(pose[2:]-previous[2:])) / 2)) + 1))
            points = [machine.forward(previous + t*(pose-previous))[0].tolist()
                      for t in np.linspace(0, 1, samples)]
            target = deposition if words.get('E', 0) > 0 else travel
            target.extend(points + [[None, None, None]])
        previous = pose
    fig = go.Figure()
    for points, name, color in ((deposition, 'G-code deposition', '#f59e0b'),
                                 (travel, 'G-code travel', '#38bdf8')):
        xyz = np.asarray(points, dtype=object).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines',
            name=name, line=dict(color=color, width=3), connectgaps=False))
    fig.update_layout(title='Generated G-code: reconstructed nozzle-tip motion')
    return layout(fig)
