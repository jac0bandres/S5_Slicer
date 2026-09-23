"""Cura slicing and S4/S5 tetrahedral G-code reformation for S3.

The signed barycentric map, radial-plane rotation recovery, and volume-based
extrusion correction follow S5.py / s4_reference.py (GPL-3.0, Joshua Bird's S4).
Cura owns deposition planning; this module transforms its existing moves.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from .mesh import TetMesh
from .machine import S5Machine
from .layers import first_bed_layer

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = Path('/usr/share/cura/resources/definitions')


def definition(name):
    system = DEFINITIONS/name
    return str(system if system.exists() else ROOT/'config'/name)


@dataclass(frozen=True)
class CuraConfig:
    engine: str = '/usr/bin/CuraEngine'
    printer_definition: str = definition('fdmprinter.def.json')
    extruder_definition: str = definition('fdmextruder.def.json')
    profile: str = str(ROOT/'config/core.def.json')
    layer_height: float = .2
    first_layer_height: float = .2
    line_width: float = .4
    filament_diameter: float = 1.75
    print_speed: float = 30.
    infill_density: float = 20.
    wall_count: int = 2
    nozzle_temperature: float = 240.
    bed_temperature: float = 55.
    segment_length: float = .6
    nozzle_offset: float = 41.5
    max_extrusion_multiplier: float = 10.

    def __post_init__(self):
        for name in ('layer_height', 'first_layer_height', 'line_width', 'filament_diameter',
                     'print_speed', 'segment_length', 'max_extrusion_multiplier'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 <= self.infill_density <= 100 or not isinstance(self.wall_count, int) or self.wall_count < 1:
            raise ValueError('Invalid Cura infill density or wall count')
        for name in ('nozzle_temperature', 'bed_temperature', 'nozzle_offset'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


def load_profile(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    settings = data.get('settings', {}).get('global', {}).get('all', data)
    result = {}
    for key, value in settings.items():
        if isinstance(value, dict):
            value = value.get('value') if value.get('value') is not None else value.get('default_value')
        if value is not None:
            result[key] = value
    return result


class TetReformer:
    """Vectorized signed barycentric mapping with S4 radial tilt recovery."""
    def __init__(self, original, deformed, cells):
        import pyvista as pv
        from s4_reference import recover_vertex_rotations
        self.original = np.asarray(original, dtype=float)
        self.deformed = np.asarray(deformed, dtype=float)
        self.cells = np.asarray(cells, dtype=int)
        old = self.original[self.cells]
        new = self.deformed[self.cells]
        self.grid = pv.UnstructuredGrid(np.column_stack([np.full(len(cells), 4), cells]).ravel(),
                                        np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8), self.deformed)
        self.inverse = np.linalg.inv((new[:, :3] - new[:, 3:4]).transpose(0, 2, 1))
        self.rotation = recover_vertex_rotations(self.original, self.deformed, self.cells,
            old.mean(axis=1), new.mean(axis=1), max_rotation_deg=180., min_rotation_deg=-180.)
        self.volume_ratio = np.abs(np.linalg.det(old[:, 1:]-old[:, :1]) /
                                   np.linalg.det(new[:, 1:]-new[:, :1]))

    def map(self, points):
        """Exterior points use the closest tetrahedron's affine extension (S5)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        cells = np.asarray(self.grid.find_containing_cell(points)).reshape(-1)
        outside = cells < 0
        if outside.any():
            cells[outside] = np.asarray(self.grid.find_closest_cell(points[outside])).reshape(-1)
        v = self.cells[cells]
        xyz = np.einsum('nij,nj->ni', self.inverse[cells], points-self.deformed[v[:, 3]])
        bary = np.column_stack([xyz, 1-xyz.sum(axis=1)])
        mapped = np.einsum('ni,nij->nj', bary, self.original[v])
        tilt = np.einsum('ni,ni->n', bary, self.rotation[v])
        return mapped, tilt, self.volume_ratio[cells], outside


def reform_gcode(text, reformer, config=None):
    """Transform Cura linear moves; retain comments and non-motion commands.

    Normalize absolute/relative XYZ/E and G92 resets. Split moves at segment_length,
    distribute Cura's E and move time, and apply S4 volume correction only to
    positive depositing motion, never E-only retractions/unretractions. Exterior
    points are affine-extrapolated and counted, never silently discarded.
    """
    cfg = config or CuraConfig()
    machine = S5Machine(nozzle_offset=cfg.nozzle_offset)
    pos = np.zeros(3); coordinate_offset = np.zeros(3)
    epos = 0.; feed = 1200.; absolute = True; relative_e = False
    records = []; positions = []
    layer = None; layers = set()
    word_pattern = re.compile(r'([A-Za-z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))')
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(';LAYER:'):
            layer = int(stripped.split(':', 1)[1])
        code, _, comment = line.partition(';')
        code = re.sub(r'\([^)]*\)', '', code).strip()
        words = [(a.upper(), float(v)) for a, v in word_pattern.findall(code)]
        if not words:
            records.append(('text', line)); continue
        command = next(((a, int(v)) for a, v in words if a in ('G', 'M', 'T')), None)
        values = dict(words)
        if command in (('G', 0), ('G', 1)):
            if 'F' in values:
                feed = values['F']
            if feed <= 0:
                raise ValueError(f'Cura line {number}: nonpositive feed')
            target = pos.copy()
            for axis, name in enumerate('XYZ'):
                if name in values:
                    target[axis] = values[name]+coordinate_offset[axis] if absolute else pos[axis]+values[name]
            extrusion = None
            if 'E' in values:
                extrusion = values['E'] if relative_e else values['E']-epos
                epos = epos+values['E'] if relative_e else values['E']
            distance = float(np.linalg.norm(target-pos))
            if comment:
                records.append(('text', ';'+comment))
            if distance <= 1e-12:
                records.append(('extruder', extrusion, feed))
            else:
                count = max(1, int(np.ceil(distance/cfg.segment_length)))
                for i in range(count):
                    point = pos+(target-pos)*(i+1)/count
                    records.append(('move', len(positions), None if extrusion is None else extrusion/count,
                                    distance/count/feed, layer))
                    positions.append(point)
                if extrusion is not None and extrusion > 0 and layer is not None:
                    layers.add(layer)
            pos = target
        elif command in (('G', 90), ('G', 91)):
            absolute = command[1] == 90
        elif command in (('M', 82), ('M', 83)):
            relative_e = command[1] == 83
        elif command == ('G', 92):
            if 'E' in values:
                epos = values['E']
            for axis, name in enumerate('XYZ'):
                if name in values:
                    coordinate_offset[axis] = pos[axis]-values[name]
        elif command == ('G', 28):
            pos[:] = 0.; coordinate_offset[:] = 0.
            records.append(('text', line))
        elif command in (('G', 2), ('G', 3), ('G', 20), ('G', 93)):
            raise ValueError(f'Cura line {number}: unsupported source command {command}; use linear millimetre G-code')
        elif command == ('G', 94):
            continue
        else:
            records.append(('text', line))
    if not positions:
        raise ValueError('Cura produced no XYZ moves')
    mapped, tilts, ratios, outside = reformer.map(positions)
    out = ['; S3 deformation / Cura toolpaths / S4-S5 reformation', 'G21', 'G90', 'M83', 'G94']
    previous_c = 0.; motion_count = 0; total_e = 0.; source_e = 0.
    for row in records:
        if row[0] == 'text':
            out.append(row[1])
        elif row[0] == 'extruder':
            _, extrusion, feed = row
            if extrusion is not None:
                out.extend(['G94', f'G1 E{extrusion:.8f} F{feed:.5f}'])
        else:
            _, index, extrusion, minutes, _ = row
            point = mapped[index]; tilt = tilts[index]
            yaw = np.arctan2(point[1], point[0]) if np.linalg.norm(point[:2]) > 1e-10 else np.deg2rad(previous_c)
            direction = [np.sin(tilt)*np.cos(yaw), np.sin(tilt)*np.sin(yaw), np.cos(tilt)]
            pose = machine.inverse(point, direction, previous_c=previous_c, strict=False)
            previous_c = pose[3]
            words = ' '.join(f'{a}{v:.5f}' for a, v in zip('XZBC', pose))
            if extrusion is not None:
                if extrusion > 0:
                    source_e += extrusion
                    extrusion *= min(ratios[index], cfg.max_extrusion_multiplier)
                    total_e += extrusion
                words += f' E{extrusion:.8f}'
            out.extend(['G93', f'G1 {words} F{1/minutes:.8f}', 'G94'])
            motion_count += 1
    report = dict(source='Cura / S4-S5 deform-reform', layers=len(layers), layer_indices=sorted(layers),
                  moves=motion_count, exterior_waypoints=int(outside.sum()),
                  source_deposition_filament=source_e, reformed_deposition_filament=total_e,
                  minimum_volume_ratio=float(ratios.min()), maximum_volume_ratio=float(ratios.max()),
                  config=asdict(cfg))
    return '\n'.join(out)+'\n', report


def slice_with_cura(result, directory, config=None, *, settings=None, callback=None):
    """Write deformed STL, Cura G-code/log, reformed print.gcode and report."""
    cfg = config or CuraConfig()
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    original = TetMesh(result['points'], result['cells'])
    deformed = np.array(result['deformed'], dtype=float, copy=True)
    offset = np.r_[(deformed[:, :2].min(axis=0)+deformed[:, :2].max(axis=0))/2, deformed[:, 2].min()]
    deformed -= offset
    reformer = TetReformer(original.points, deformed, original.cells)
    surface = reformer.grid.extract_surface(algorithm='dataset_surface').triangulate()
    stl = root/'cura_deformed.stl'
    surface.save(stl)
    opts = load_profile(cfg.profile)
    opts.update(layer_height=cfg.layer_height, layer_height_0=cfg.first_layer_height,
        line_width=cfg.line_width, wall_line_width=cfg.line_width, wall_line_width_0=cfg.line_width,
        wall_line_width_x=cfg.line_width, skin_line_width=cfg.line_width, infill_line_width=cfg.line_width,
        initial_layer_line_width_factor=100, material_diameter=cfg.filament_diameter,
        speed_print=cfg.print_speed, speed_wall=cfg.print_speed, speed_wall_0=cfg.print_speed,
        speed_wall_x=cfg.print_speed, speed_infill=cfg.print_speed, speed_topbottom=cfg.print_speed,
        infill_sparse_density=cfg.infill_density,
        infill_line_distance=cfg.line_width*100/cfg.infill_density if cfg.infill_density else 0,
        wall_line_count=cfg.wall_count,
        material_print_temperature=cfg.nozzle_temperature, material_print_temperature_layer_0=cfg.nozzle_temperature,
        material_bed_temperature=cfg.bed_temperature, material_bed_temperature_layer_0=cfg.bed_temperature,
        machine_heated_bed=cfg.bed_temperature>0,
        machine_start_gcode=opts.get('machine_start_gcode') or 'G28\nG90\nM83',
        machine_end_gcode=opts.get('machine_end_gcode') or 'M104 S0\nM140 S0')
    for name,value in dict(top_layers=4,bottom_layers=4,adhesion_type='none',support_enable=False).items():
        opts.setdefault(name,value)
    opts.update(settings or {})
    # Coordinate conventions required by the inverse map, not slicer heuristics.
    opts.update(machine_center_is_zero=True, center_object=False, mesh_position_x=0,
                mesh_position_y=0, mesh_position_z=0, machine_gcode_flavor='RepRap (Marlin/Sprinter)',
                relative_extrusion=True)
    # Preserve the requested physical first-layer upper bound under reformation.
    first, bed_report = first_bed_layer(original, deformed[:, 2], cfg.first_layer_height)
    opts['layer_height_0'] = min(float(opts['layer_height_0']), first.level)
    raw = root/'cura_deformed.gcode'
    command = [cfg.engine, 'slice', '-j', cfg.printer_definition, '-j', cfg.extruder_definition]
    flags=[]
    for name, value in opts.items():
        value = str(value).lower() if isinstance(value, bool) else str(value)
        flags.extend(['-s', f'{name}={value}'])
    # Cura's extruder train owns material diameter and other extrusion settings.
    # Apply overrides in both scopes so train defaults cannot shadow the profile.
    command.extend(flags)
    command.extend(['-e0'])
    command.extend(flags)
    command.extend(['-l', str(stl.resolve()), '-o', str(raw.resolve())])
    if callback: callback('Slicing deformed model with Cura…')
    env = dict(os.environ); env.setdefault('OMP_NUM_THREADS', '2')
    try:
        process = subprocess.run(command, capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        raise RuntimeError(f'CuraEngine not found: {cfg.engine}') from exc
    (root/'cura.log').write_text(process.stdout+'\n'+process.stderr)
    if process.returncode or not raw.exists() or not raw.stat().st_size:
        raise RuntimeError('CuraEngine failed; see cura.log.\n'+'\n'.join(line[:500] for line in process.stderr.splitlines()[-15:]))
    if callback: callback('Reforming Cura moves through the tetrahedral map…')
    output, report = reform_gcode(raw.read_text(), reformer, cfg)
    report.update(deformed_offset=offset.tolist(), cura_settings=opts, first_layer=bed_report)
    (root/'print.gcode').write_text(output)
    (root/'cura_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return output, report
