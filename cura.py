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

ROOT = Path(__file__).resolve().parent
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
    z_hop: float = 1.
    travel_speed: float = 60.  # linear actuator mm/s
    rotary_speed: float = 30.  # degrees/s
    brim_width: float = 0.  # physical bed brim; zero disables it
    brim_gap: float = 0.

    def __post_init__(self):
        for name in ('layer_height', 'first_layer_height', 'line_width', 'filament_diameter',
                     'print_speed', 'segment_length', 'max_extrusion_multiplier', 'travel_speed', 'rotary_speed'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 <= self.infill_density <= 100 or not isinstance(self.wall_count, int) or self.wall_count < 1:
            raise ValueError('Invalid Cura infill density or wall count')
        for name in ('nozzle_temperature', 'bed_temperature', 'nozzle_offset', 'z_hop',
                     'brim_width', 'brim_gap'):
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
        from research.s4.s4_reference import recover_vertex_rotations
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


def reform_gcode(text, reformer, config=None, *, source_first_layer_height=None):
    """Transform Cura linear moves; retain comments and non-motion commands.

    Normalize absolute/relative XYZ/E and G92 resets. Split moves at segment_length,
    distribute Cura's E and move time. First-layer E follows mapped bead length
    and bed gap; later layers use S4 volume correction. E-only retractions and
    unretractions are unchanged. Exterior travel samples are replaced by S5
    clearance bridges; depositing points and
    exterior travel endpoints retain affine extrapolation. Retraction adds a
    nozzle-compensated hop. Travel timing also bounds linear and rotary speed.
    """
    cfg = config or CuraConfig()
    if source_first_layer_height is None:
        source_first_layer_height = cfg.first_layer_height
    if not np.isfinite(source_first_layer_height) or source_first_layer_height <= 0:
        raise ValueError('source_first_layer_height must be finite and positive')
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
                                    distance/count/feed, layer, distance/count))
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
    previous_point = None; previous_tilt = 0.; previous_pose = None
    retracted = 0.; highest_printed = 0.
    air_index = None; air_minutes = 0.; air_samples = 0; bridges = 0

    def emit(point, tilt, minutes, extrusion=None, *, smooth=False):
        nonlocal previous_point, previous_tilt, previous_pose, previous_c, motion_count
        point = np.asarray(point, dtype=float)
        if smooth:
            tilt = .2 * tilt + .8 * previous_tilt
        start = point if previous_point is None else previous_point.copy()
        start_tilt = previous_tilt
        count = max(1, int(np.ceil(abs(tilt-start_tilt) / np.deg2rad(1))))
        hop = cfg.z_hop if retracted > 1e-9 else 0.
        for step in range(1, count+1):
            fraction = step/count
            p = start + (point-start)*fraction
            b = start_tilt + (tilt-start_tilt)*fraction
            yaw = np.arctan2(p[1], p[0]) if np.linalg.norm(p[:2]) > 1e-10 else np.deg2rad(previous_c)
            direction = [np.sin(b)*np.cos(yaw), np.sin(b)*np.sin(yaw), np.cos(b)]
            pose = machine.inverse(p, direction, previous_c=previous_c, strict=False)
            # S5 uses nozzle_offset + hop: the physical tip backs away along
            # the tilted nozzle, rather than adding only a machine-Z offset.
            pose[:2] += [-np.sin(b)*hop, np.cos(b)*hop]
            duration = minutes/count
            if previous_pose is not None and (extrusion is None or extrusion <= 0):
                delta = np.abs(pose-previous_pose)
                duration = max(duration, float(delta[:2].max())/cfg.travel_speed/60,
                               float(delta[2:].max())/cfg.rotary_speed/60)
            if duration <= 1e-12:
                continue
            words = ' '.join(f'{a}{v:.5f}' for a, v in zip('XZBC', pose))
            if extrusion is not None:
                words += f' E{extrusion/count:.8f}'
            out.extend(['G93', f'G1 {words} F{1/duration:.8f}', 'G94'])
            previous_pose = pose
            previous_c = pose[3]
            motion_count += 1
        previous_point = point.copy()
        previous_tilt = tilt

    def finish_air(index=None, minutes=0.):
        nonlocal air_index, air_minutes, bridges
        if air_index is None:
            return
        endpoint = air_index if index is None else index
        destination = mapped[endpoint].copy()
        tilt = float(np.clip(tilts[endpoint], -np.pi/4, np.pi/4))
        height = max(highest_printed, previous_point[2], destination[2])
        lifted = previous_point.copy(); lifted[2] = height
        emit(lifted, previous_tilt, 0.)
        raised = destination.copy(); raised[2] = height
        emit(raised, tilt, air_minutes+minutes)
        emit(destination, tilt, 0.)
        air_index = None; air_minutes = 0.
        bridges += 1

    for row in records:
        if row[0] == 'text':
            # Preserve ordering for executable commands (including homing),
            # while allowing comments within a continuous travel bridge.
            if row[1].strip() and not row[1].lstrip().startswith(';'):
                finish_air()
            out.append(row[1])
            if row[1].strip().upper().startswith('G28'):
                previous_point = None; previous_pose = None; previous_c = 0.; previous_tilt = 0.
        elif row[0] == 'extruder':
            _, extrusion, feed = row
            if extrusion is not None:
                finish_air()
                # Lower before unretracting; lift after retracting. Track debt
                # instead of relying on S5's hard-coded E == +/-1 sentinel.
                if extrusion > 0:
                    retracted = max(0., retracted-extrusion)
                    if previous_point is not None:
                        emit(previous_point, previous_tilt, 0.)
                out.extend(['G94', f'G1 E{extrusion:.8f} F{feed:.5f}'])
                if extrusion < 0:
                    retracted -= extrusion
                    if previous_point is not None:
                        emit(previous_point, previous_tilt, 0.)
        else:
            _, index, extrusion, minutes, move_layer, source_length = row
            depositing = extrusion is not None and extrusion > 0
            if not depositing and (extrusion is None or extrusion == 0) and outside[index] and previous_point is not None:
                if air_index is None:
                    lifted = previous_point.copy()
                    lifted[2] = max(highest_printed, lifted[2])
                    emit(lifted, float(np.clip(previous_tilt, -np.pi/4, np.pi/4)), 0.)
                air_index = index
                air_minutes += minutes
                air_samples += 1
                continue
            if air_index is not None:
                if not depositing and (extrusion is None or extrusion == 0):
                    finish_air(index, minutes)
                    continue
                finish_air()
            if extrusion is not None:
                if depositing:
                    # A combined XYZ/E move may also repay retraction debt.
                    repayment = min(retracted, extrusion)
                    retracted -= repayment
                    if repayment and previous_point is not None:
                        emit(previous_point, previous_tilt, 0.)
                    source_e += extrusion-repayment
                    if move_layer == 0 and previous_point is not None:
                        # Cura's first-layer E already includes its chosen line
                        # width and flow. Map only the dimensions a fixed-width
                        # physical bead changes: path length and bed gap. The
                        # tet volume ratio also includes lateral deformation,
                        # which can starve a thin first layer.
                        mapped_length = float(np.linalg.norm(mapped[index]-previous_point))
                        bed_gap = max(0., min(cfg.first_layer_height,
                                              .5*(mapped[index, 2]+previous_point[2])))
                        multiplier = mapped_length/source_length*bed_gap/source_first_layer_height
                    else:
                        multiplier = ratios[index]
                    extrusion = repayment + (extrusion-repayment)*min(multiplier, cfg.max_extrusion_multiplier)
                    total_e += extrusion-repayment
                elif extrusion < 0:
                    retracted -= extrusion
            emit(mapped[index], float(tilts[index]), minutes, extrusion, smooth=True)
            if depositing:
                highest_printed = max(highest_printed, float(mapped[index, 2]))
    finish_air()
    report = dict(source='Cura / S4-S5 deform-reform', layers=len(layers), layer_indices=sorted(layers),
                  moves=motion_count, exterior_waypoints=int(outside.sum()),
                  air_travel_samples_replaced=air_samples, air_travel_bridges=bridges,
                  source_deposition_filament=source_e, reformed_deposition_filament=total_e,
                  minimum_volume_ratio=float(ratios.min()), maximum_volume_ratio=float(ratios.max()),
                  source_first_layer_height=float(source_first_layer_height), config=asdict(cfg))
    return '\n'.join(out)+'\n', report


def add_bed_brim(gcode, cfg):
    """Add flat S5 brim loops around the reformed first-layer outer walls."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    machine = S5Machine(nozzle_offset=cfg.nozzle_offset)
    rows = gcode.splitlines()
    layer = None; role = None; previous = None; footprint = []
    marker = None; saved_pose = None; retraction = 0.
    pattern = re.compile(r'([XZBCEF])([-+]?(?:\d+(?:\.\d*)?|\.\d+))')
    for i, line in enumerate(rows):
        if line.startswith(';LAYER:'):
            layer = int(line.partition(':')[2])
            if layer == 0 and marker is None:
                marker = i
                saved_pose = previous
        elif line.startswith(';TYPE:'):
            role = line.partition(':')[2].strip()
        if line.startswith('G1 X'):
            values = {key: float(value) for key, value in pattern.findall(line)}
            pose = np.array([values[key] for key in 'XZBC'])
            point, _ = machine.forward(pose)
            if layer == 0 and role == 'WALL-OUTER' and values.get('E', 0) > 0 and previous is not None:
                start, _ = machine.forward(previous)
                if np.linalg.norm(point[:2]-start[:2]) > 1e-7:
                    footprint.append(LineString([start[:2], point[:2]]))
            previous = pose
        elif marker is None and line.startswith('G1 E'):
            amount = float(dict(pattern.findall(line))['E'])
            retraction = max(0., retraction-amount)
    if marker is None or not footprint:
        raise ValueError('Cannot form brim: Cura produced no first-layer outer-wall moves')
    base = unary_union(footprint).buffer(cfg.line_width/2, cap_style=2, join_style=2)
    count = max(1, int(np.ceil(cfg.brim_width/cfg.line_width)))
    loops = []
    for n in range(count):
        outline = base.buffer(cfg.brim_gap + cfg.line_width*(n+.5), join_style=2)
        polygons = [outline] if outline.geom_type == 'Polygon' else list(outline.geoms)
        loops.extend(np.asarray(poly.exterior.coords)[:, :2] for poly in polygons if not poly.is_empty)
    if not loops:
        raise ValueError('Cannot form brim: first-layer footprint is empty')

    lines = [';TYPE:SKIRT-BRIM', '; S3 physical bed brim']
    previous_pose = saved_pose if saved_pose is not None else np.zeros(4)
    previous_c = float(saved_pose[3]) if saved_pose is not None else 0.
    filament_area = np.pi*(cfg.filament_diameter/2)**2
    extruded = 0.

    def move(xy, height, extrusion=0.):
        nonlocal previous_pose, previous_c, extruded
        point = np.array([xy[0], xy[1], height])
        pose = machine.inverse(point, [0., 0., 1.], previous_c=previous_c)
        if not extrusion and previous_pose is not None and np.allclose(pose, previous_pose, atol=1e-8):
            return
        if previous_pose is not None:
            delta = np.abs(pose-previous_pose)
            seconds = max(float(delta[:2].max())/cfg.travel_speed,
                          float(delta[2:].max())/cfg.rotary_speed,
                          float(np.linalg.norm(point[:2]-machine.forward(previous_pose)[0][:2])) /
                          (cfg.print_speed if extrusion else cfg.travel_speed), 1e-5)
        else:
            seconds = 1.
        words = ' '.join(f'{axis}{value:.5f}' for axis, value in zip('XZBC', pose))
        if extrusion:
            words += f' E{extrusion:.8f}'
            extruded += extrusion
        lines.extend(['G93', f'G1 {words} F{60/seconds:.8f}', 'G94'])
        previous_pose = pose
        previous_c = pose[3]

    for number, loop in enumerate(reversed(loops)):  # outermost first
        start = loop[0]
        move(start, cfg.first_layer_height + cfg.z_hop)
        move(start, cfg.first_layer_height)
        if number == 0 and retraction:
            lines.extend(['G94', f'G1 E{retraction:.8f} F{min(1500., cfg.print_speed*60):.5f}'])
        for point in loop[1:]:
            length = float(np.linalg.norm(point-start))
            move(point, cfg.first_layer_height,
                 length*cfg.line_width*cfg.first_layer_height/filament_area)
            start = point
        if number == len(loops)-1 and retraction:
            lines.extend(['G94', f'G1 E{-retraction:.8f} F{min(1500., cfg.print_speed*60):.5f}'])
        move(start, cfg.first_layer_height + cfg.z_hop)
    if saved_pose is not None:
        target, _ = machine.forward(saved_pose)
        move(target[:2], max(target[2], cfg.first_layer_height + cfg.z_hop))
        words = ' '.join(f'{axis}{value:.5f}' for axis, value in zip('XZBC', saved_pose))
        delta = np.abs(saved_pose-previous_pose)
        seconds = max(float(delta[:2].max())/cfg.travel_speed,
                      float(delta[2:].max())/cfg.rotary_speed, 1e-5)
        lines.extend(['G93', f'G1 {words} F{60/seconds:.8f}', 'G94'])
    else:
        first_move = next((line for line in rows[marker+1:] if line.startswith('G1 X')), None)
        if first_move:
            values = {key: float(value) for key, value in pattern.findall(first_move)}
            target_pose = np.array([values[key] for key in 'XZBC'])
            target, _ = machine.forward(target_pose)
            move(target[:2], max(target[2], cfg.first_layer_height + cfg.z_hop))
            delta = np.abs(target_pose-previous_pose)
            seconds = max(float(delta[:2].max())/cfg.travel_speed,
                          float(delta[2:].max())/cfg.rotary_speed, 1e-5)
            words = ' '.join(f'{axis}{value:.5f}' for axis, value in zip('XZBC', target_pose))
            lines.extend(['G93', f'G1 {words} F{60/seconds:.8f}', 'G94'])
    rows[marker+1:marker+1] = lines
    return '\n'.join(rows)+'\n', dict(brim_loops=len(loops), brim_filament=extruded,
                                      brim_moves=sum(line.startswith('G1 X') for line in lines))


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
    if cfg.brim_width:
        opts['adhesion_type'] = 'none'
    # Reformation owns the compensated hop; Cura must not add another one.
    opts.update(retraction_hop_enabled=False, retraction_hop_after_extruder_switch=False)
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
    output, report = reform_gcode(raw.read_text(), reformer, cfg,
                                  source_first_layer_height=float(opts['layer_height_0']))
    if cfg.brim_width:
        output, brim_report = add_bed_brim(output, cfg)
        report.update(brim_report)
        report['moves'] += brim_report['brim_moves']
    report.update(deformed_offset=offset.tolist(), cura_settings=opts, first_layer=bed_report)
    (root/'print.gcode').write_text(output)
    (root/'cura_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return output, report
