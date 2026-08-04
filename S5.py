import numpy as np
import tetgen
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import pickle
import time
import base64
import os
import argparse
import pyvista as pv
import subprocess
import configparser
import igl
import potpourri3d as pp3d
import json
import pymeshfix
import trimesh

import warnings

warnings.simplefilter("ignore")
CURA_LOGS_DIR = os.path.abspath("./cura_logs")

from s5_viz import viz, VizConfig
from s5_bench import bench, BenchConfig
import s4_reference as s4ref

VizConfig.enabled = False
VizConfig.output_dir = "./plots"
VizConfig.interactive = False

BenchConfig.enabled = False
BenchConfig.output_dir = "./bench_results"
BenchConfig.run_tag = "pi_3mm_subdiv_0"
BenchConfig.run_s4_comparison = True
BenchConfig.record_correctness = True
BenchConfig.track_memory = False

total_time = time.time()
plotter = pv.Plotter()


def reorient_mesh(mesh: trimesh.Trimesh, euler_string: str):
    """
    euler_string format:
        "90,0,180"
    interpreted as X,Y,Z degrees
    """
    rx, ry, rz = (np.radians(float(a)) for a in euler_string.split(","))

    transform = trimesh.transformations.euler_matrix(rx, ry, rz)

    mesh.apply_transform(transform)

    print(f"  [reorient] Euler X,Y,Z = {euler_string}°")

    return mesh


def _mesh_worker(backend, points, faces, kwargs, out_path):
    """Tetrahedralize in a child process and pickle the resulting pv grid (or
    the error) to `out_path`.

    Two backends:
      • "tetgen"   — fast, quality-controlled, but brittle: it does not merely
        raise on pathological surfaces (self-intersections, degenerate facets)
        — it can *segfault*, an uncatchable hard crash.
      • "ftetwild" — FloatTetWild: envelope-based, robust to self-intersecting /
        non-manifold / broken input by design (it does its own winding-number
        inside/outside extraction).  Slower, approximate boundary.

    Running either in a forked child contains a crash: the child dies, the
    output file is never written, and the parent treats that as a recoverable
    failure and moves to the next fallback tier.
    """
    import os
    import pickle
    import warnings

    warnings.simplefilter("ignore")
    # Silence the backends' native (C++) logging — all results and errors come
    # back through the pickle file, so the child's stdout/stderr is noise.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    import numpy as np
    import pyvista as pv

    try:
        if backend == "tetgen":
            import tetgen

            t = tetgen.TetGen(pv.PolyData(points, faces))
            t.tetrahedralize(**kwargs)
            grid = t.grid
        else:  # "ftetwild"
            import wildmeshing as wm

            tri = faces.reshape(-1, 4)[:, 1:].astype(np.int32)
            tet = wm.Tetrahedralizer(**kwargs)
            tet.set_mesh(np.asarray(points, np.float64), tri)
            tet.tetrahedralize()
            # Winding-number inside/outside (floodfill=False).  Flood fill seeds
            # from outside and leaks through holes on high-genus lattices —
            # e.g. it collapsed a frame to 2 tets — so it is unreliable here;
            # the winding-number test is robust to topology.
            VT, TT, _ = tet.get_tet_mesh(floodfill=False)
            VT = np.asarray(VT, float)
            TT = np.asarray(TT, np.int64)
            if len(TT) == 0:
                raise RuntimeError("FloatTetWild produced an empty mesh")
            cells = np.hstack([np.full((len(TT), 1), 4, np.int64), TT]).ravel()
            grid = pv.UnstructuredGrid(
                cells, np.full(len(TT), pv.CellType.TETRA, np.uint8), VT
            )
        with open(out_path, "wb") as fh:
            pickle.dump({"ok": True, "grid": grid}, fh)
    except RuntimeError as e:
        with open(out_path, "wb") as fh:
            pickle.dump({"ok": False, "error": str(e).strip()}, fh)


def infer_rotation_multiplier(tet, base_multiplier=2.0):
    # Get the distribution of overhang angles
    overhangs = tet.cell_data["overhang_angle"]
    # Filter for valid surface faces
    valid_overhangs = overhangs[~np.isnan(overhangs)]

    if len(valid_overhangs) == 0:
        return base_multiplier

    # If the mesh has many extreme overhangs, reduce the multiplier
    # to prevent "over-tilting" which causes mesh destruction.
    max_ov = np.rad2deg(np.nanmax(valid_overhangs))

    if max_ov > 135:  # Very steep/in-air sections
        return base_multiplier * 0.7
    elif max_ov < 100:  # Shallow overhangs
        return base_multiplier * 1.5
    return base_multiplier


# ── Unlayered-Slicer / RepRapFirmware G-code dialect ────────────────────────
# S5's native output is the polar Core-R-Theta dialect (C/X/Z/B + inverse-time
# feed). The Unlayered-Slicer fork instead drives a Cartesian XYZ + rotary B/C
# machine that resolves tool-tip kinematics (RTCP), with relative extrusion,
# standalone mm/min feedrate lines, and PrusaSlicer/Orca-style annotations.
# The B/C convention below matches the fork's trafo_to_bc() (verified against an
# example export); see vise.md §9.3-9.5.

# CuraEngine ;TYPE roles -> the Orca/PrusaSlicer role names the example uses.
_CURA_TO_ORCA_TYPE = {
    "WALL-OUTER": "Outer wall",
    "WALL-INNER": "Inner wall",
    "SKIN": "Top surface",
    "FILL": "Sparse infill",
    "SUPPORT": "Support",
    "SUPPORT-INTERFACE": "Support interface",
    "SKIRT": "Skirt",
    "SKIRT-BRIM": "Skirt",
    "PRIME-TOWER": "Prime tower",
}


def nozzle_to_bc(n):
    """Nozzle axis -> (B, C) degrees, matching Unlayered-Slicer's trafo_to_bc().

    C (yaw) = atan2(n_y, n_x); B (pitch) = atan2(-n_z, hypot(n_x, n_y)). A
    vertical axis is the degenerate case C=0, B=-/+90 (flat printing is B=-90).
    """
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    if abs(nx) < 1e-6 and abs(ny) < 1e-6:
        return (-90.0 if nz > 0 else 90.0), 0.0
    return (
        np.degrees(np.arctan2(-nz, np.hypot(nx, ny))),
        np.degrees(np.arctan2(ny, nx)),
    )


def _nozzle_axes_from_rotations(positions, rotations):
    """Per-point nozzle axis n = R(rotation, tangent) . zhat (vectorized).

    Mirrors calculate_rotation_matrices(): the tilt is `rotation` radians about
    the tangential axis t = zhat x radial_xy, so n leans in the radial-vertical
    plane. Returns an (N, 3) array; NaN rows propagate from NaN inputs.
    """
    xy = positions[:, :2]
    t = np.cross(
        np.array([0.0, 0.0, 1.0]),
        np.column_stack([xy, np.zeros(len(xy))]),
    )
    tn = np.linalg.norm(t, axis=1)
    degenerate = tn < 1e-9
    t[degenerate] = [1.0, 0.0, 0.0]
    tn[degenerate] = 1.0
    t /= tn[:, None]
    rotvecs = rotations[:, None] * t
    return R.from_rotvec(rotvecs).apply([0.0, 0.0, 1.0])


def write_unlayered_gcode(
    out_path,
    points,
    *,
    line_width=0.4,
    layer_height=0.2,
    nozzle_temp=240,
    bed_temp=55,
    pressure_advance=0.0,
    zhop=0.5,
    start_gcode=None,
    end_gcode=None,
):
    """Write continuous 5-axis nonplanar G-code in the Unlayered-Slicer / RRF
    dialect from S5's reformed toolpath points.

    Each extruding vertex emits `G1 X Y Z B C E` (tool-tip in the bed frame,
    B/C from the local nozzle axis via nozzle_to_bc). Conventions follow the
    example export: relative extrusion (M83), feedrate on its own `G1 F` line
    only when it changes (mm/min, not inverse-time), and PrusaSlicer/Orca
    annotations (;LAYER_CHANGE, ;Z, ;HEIGHT, ;TYPE:<role>, per-move ;WIDTH).

    points: the `new_gcode_points` list, each a dict with at least
        position (3,), rotation (rad), command, extrusion, feed, travelling,
        extrusion_multiplier, and optionally layer (int) / type (Cura role str).
    """
    positions = np.array([p["position"] for p in points], dtype=float)
    rotations = np.array(
        [p["rotation"] if p["rotation"] is not None else np.nan for p in points],
        dtype=float,
    )
    axes = _nozzle_axes_from_rotations(positions, rotations)
    nx, ny, nz = axes[:, 0], axes[:, 1], axes[:, 2]
    vertical = (np.abs(nx) < 1e-6) & (np.abs(ny) < 1e-6)
    B = np.degrees(np.arctan2(-nz, np.hypot(nx, ny)))
    C = np.degrees(np.arctan2(ny, nx))
    B[vertical] = np.where(nz[vertical] > 0, -90.0, 90.0)
    C[vertical] = 0.0

    out = []
    if start_gcode is not None:
        out.append(start_gcode.rstrip("\n"))
    else:
        out += [
            "; generated by S5_Slicer (Unlayered-Slicer / RRF dialect)",
            "G21 ; mm",
            "G90 ; absolute XYZ",
            "M83 ; relative extrusion",
        ]
        if bed_temp:
            out.append(f"M140 S{bed_temp} ; set bed temperature")
            out.append(f"M190 S{bed_temp} ; wait for bed")
        if nozzle_temp:
            out.append(f"M104 S{nozzle_temp} ; set nozzle temperature")
            out.append(f"M109 S{nozzle_temp} ; wait for nozzle")
        out.append("G28 ; home")
        if pressure_advance:
            out.append(f"M572 D0 S{pressure_advance} ; pressure advance")
        out += [
            "G92 E0",
            "G1 Z10 B-90 C0 F8000 ; lift, level nozzle",
        ]

    last_f = None
    last_w = None
    last_layer = None
    last_type = None
    n_moves = 0
    for i, p in enumerate(points):
        pos = positions[i]
        if np.any(np.isnan(pos)) or pos[2] < 0:
            continue

        layer = p.get("layer")
        if layer is not None and layer != last_layer:
            out += [
                ";LAYER_CHANGE",
                f";Z:{pos[2]:.3f}",
                f";HEIGHT:{layer_height:.3f}",
            ]
            last_layer = layer

        role = _CURA_TO_ORCA_TYPE.get(p.get("type"))
        if role is not None and role != last_type:
            out.append(f";TYPE:{role}")
            last_type = role

        x, y, z = pos
        if p.get("travelling"):
            z += zhop

        f = p.get("feed")
        if f is not None:
            fi = int(round(f))
            if fi != last_f:
                out.append(f"G1 F{fi}")
                last_f = fi

        extrusion = p.get("extrusion")
        if extrusion is not None and extrusion > 0:
            w = line_width * float(p.get("extrusion_multiplier", 1.0) or 1.0)
            if last_w is None or abs(w - last_w) > 0.02:
                out.append(f";WIDTH:{w:.3f}")
                last_w = w

        line = f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} B{B[i]:.3f} C{C[i]:.3f}"
        if extrusion is not None:
            line += f" E{extrusion:.5f}"
        out.append(line)
        n_moves += 1

    if end_gcode is not None:
        out.append(end_gcode.rstrip("\n"))
    else:
        out += [
            "M104 S0 ; nozzle off",
            "M140 S0 ; bed off",
            "M106 S0 ; fan off",
            "; S5 end",
        ]

    with open(out_path, "w") as fh:
        fh.write("\n".join(s for s in out if s != "") + "\n")
    print(f"Saving to {out_path} ({n_moves} moves, Unlayered dialect)")


def slice(
    model_path,
    config_path,
    extruder_path,
    printer_path,
    output_path,
    cura_path,
    offset,
    scale,
    rotation_multiplier,
    neighbor_loss_weight,
    max_overhang,
    nozzle_offset,
    reorient=None,
    gcode_dialect="rtheta",
    start_gcode_path=None,
    end_gcode_path=None,
):
    total_t = time.time()
    prev_time = total_t

    if model_path:
        model_name = os.path.basename(model_path).split(".")[0]
        print(f"Slicing {model_name}...")
        model_name = f"{model_name}_{time.time()}"
    else:
        print("No model path specified.")
        exit()

    os.makedirs(output_path, exist_ok=True)

    def encode_object(obj):
        return base64.b64encode(pickle.dumps(obj)).decode("utf-8")

    def decode_object(encoded_str):
        return pickle.loads(base64.b64decode(encoded_str))

    up_vector = np.array([0, 0, 1])

    mesh = trimesh.load(model_path, force="mesh")

    if reorient:
        mesh = reorient_mesh(mesh, reorient)

    faces = np.asarray(mesh.faces)
    faces_pv = np.hstack([np.full((len(faces), 1), 3, np.int64), faces]).ravel()
    surf = pv.PolyData(np.asarray(mesh.vertices, float), faces_pv).triangulate().clean()
    if surf.n_open_edges > 0:
        surf = surf.fill_holes(surf.length).triangulate().clean()

    min_ratio = 1.5

    QUAL = dict(order=1, mindihedral=20, minratio=min_ratio)
    max_volume = None
    if max_volume is not None:
        QUAL["maxvolume"] = max_volume

    def _run(s, backend, kwargs):
        """Tetrahedralize `s` with `backend` in an isolated subprocess; return
        the pv grid.

        Raises RuntimeError on either a backend error *or* a hard crash
        (segfault), so the caller's fallback tiers handle both uniformly — a
        bad mesh can never bring down the whole pipeline.
        """
        import multiprocessing as mp
        import tempfile
        import pickle

        fd, out = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        os.remove(out)  # worker (re)creates it only on success/clean-error
        proc = mp.get_context("fork").Process(
            target=_mesh_worker,
            args=(
                backend,
                np.asarray(s.points, float),
                np.asarray(s.faces),
                kwargs,
                out,
            ),
        )
        proc.start()
        proc.join()
        if not os.path.exists(out):
            raise RuntimeError(
                f"{backend} crashed (exit code {proc.exitcode}) — unrecoverable "
                "self-intersection or degenerate facet"
            )
        try:
            with open(out, "rb") as fh:
                res = pickle.load(fh)
        finally:
            os.remove(out)
        if not res["ok"]:
            raise RuntimeError(res["error"])
        grid = res["grid"]
        # Reject a degenerate/disconnected result (e.g. a few stray tets with no
        # shared vertices — a "soup").  A valid conforming tet mesh shares
        # vertices heavily (n_points ≈ n_cells/5); a soup has n_points ≈ 4·n_cells.
        if grid.n_cells < 4 or grid.n_points > 3 * grid.n_cells:
            raise RuntimeError(
                f"{backend} returned a degenerate mesh "
                f"({grid.n_cells} cells, {grid.n_points} points)"
            )
        return grid

    def _repair(s):
        """Make a surface watertight, manifold and self-intersection-free so
        TetGen's boundary recovery can succeed.

        TetGen aborts on raw CAD/STL surfaces that contain self-intersections or
        near-degenerate facets (`recoversubfaces`).  Two-stage repair:

          1. trimesh — cheap topological cleanup: weld coincident vertices, drop
             degenerate and duplicate faces, prune unreferenced vertices and make
             the winding consistent.
          2. pymeshfix (Attene's MeshFix) — the heavy lifting: removes
             self-intersections and returns a watertight manifold.  We keep every
             connected component (`remove_smallest_components=False`) so
             multi-part models (e.g. a tree's separate branches) are not silently
             gutted, and join the closest ones so the result is a single PLC.
        """
        import pymeshfix

        v = np.asarray(s.points, float)
        f = s.faces.reshape(-1, 4)[:, 1:]

        tm = trimesh.Trimesh(v, f, process=True)
        tm.update_faces(tm.nondegenerate_faces())
        tm.update_faces(tm.unique_faces())
        tm.remove_unreferenced_vertices()
        tm.fix_normals()

        mfix = pymeshfix.MeshFix(np.asarray(tm.vertices, float), np.asarray(tm.faces))
        mfix.repair(joincomp=True, remove_smallest_components=False)
        v2, f2 = mfix.points, mfix.faces
        return pv.PolyData(
            v2, np.hstack([np.full((len(f2), 1), 3, np.int64), f2]).ravel()
        )

    def _cleanup_tetgen_scratch():
        # TetGen drops `_skipped.face` / `_skipped.node` into the cwd whenever
        # boundary recovery fails (e.g. self-intersections); remove them so they
        # don't accumulate as stray untracked files in the project dir.
        for fn in ("_skipped.face", "_skipped.node"):
            try:
                os.remove(fn)
            except OSError:
                pass

    # Tiered, geometry-preserving fallback.  TetGen is brittle on raw CAD STLs
    # (self-intersections, near-coplanar/degenerate facets → `recoversubfaces`
    # / `split_segment`, sometimes a segfault).  Each tier keeps full mesh
    # resolution — we never decimate, which would throw away geometry.  A looser
    # `epsilon` makes TetGen merge near-coincident input, which both converts
    # some hard crashes into recoverable states and lets borderline meshes pass.
    _fixed = {}

    def _repaired():
        if "m" not in _fixed:
            _fixed["m"] = _repair(surf)
        return _fixed["m"]

    # FloatTetWild params: epsilon = surface envelope (× bbox diag), edge_length_r
    # = target edge (× bbox diag).  Smaller edge_length_r preserves more surface
    # detail at the cost of tet count / time; 0.05 is a robust default for the
    # deformation field.  fTetWild is the robust catch-all — it ingests the raw
    # surface (self-intersections and all) and does its own inside/outside.
    FTW = dict(epsilon=1e-3, edge_length_r=0.05)
    tiers = [
        ("tetgen as-is", lambda: _run(surf, "tetgen", QUAL)),
        ("tetgen repaired", lambda: _run(_repaired(), "tetgen", QUAL)),
        ("fTetWild (robust)", lambda: _run(surf, "ftetwild", FTW)),
    ]

    input_tet = None
    for name, attempt in tiers:
        try:
            input_tet = attempt()
            if name != "tetgen as-is":
                print(f"  tetrahedralization succeeded with tier: {name}")
            break
        except RuntimeError as e:
            print(f"  [{name}] failed: {str(e).strip()}")
        finally:
            _cleanup_tetgen_scratch()

    if input_tet is None:
        raise RuntimeError(
            "Could not tetrahedralize the mesh — even FloatTetWild failed. "
            "The input is likely empty, non-watertight beyond recovery, or has "
            "zero volume. Inspect the model before slicing."
        )

    input_tet = input_tet.scale(scale)

    bench.set_global_context(
        model=model_path.split("/")[-1].rsplit(".", 1)[0],
        n_cells=input_tet.number_of_cells,
        n_points=input_tet.number_of_points,
    )

    PART_OFFSET = np.array(offset)
    x_min, x_max, y_min, y_max, z_min, z_max = input_tet.bounds
    input_tet.points -= (
        np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]) + PART_OFFSET
    )
    viz.input_mesh(input_tet)
    # plotter.add_mesh(input_tet, show_edges=True)

    print("Finding cell neighbors")
    cells_only = input_tet.cells.reshape(-1, 5)[:, 1:]
    num_cells = input_tet.n_cells
    num_points = input_tet.n_points

    cell_indices = np.repeat(np.arange(num_cells), 4)
    point_indices = cells_only.ravel()
    data = np.ones(len(cell_indices), dtype=int)

    C_P = sparse.csr_matrix(
        (data, (cell_indices, point_indices)), shape=(num_cells, num_points)
    )
    A = (C_P @ C_P.T).tocoo()

    mask = A.row != A.col
    rows = A.row[mask]
    cols = A.col[mask]
    counts = A.data[mask]

    sym_mask = rows < cols

    for n_type, condition in [
        ("point", counts >= 1),
        ("edge", counts >= 2),
        ("face", counts == 3),
    ]:
        final_mask = condition & sym_mask
        input_tet.field_data[f"cell_{n_type}_neighbours"] = np.column_stack(
            (cols[final_mask], rows[final_mask])
        )

    print(f"Neighbors done: {round(time.time() - prev_time, 2)}s")
    prev_time = time.time()

    def build_surface_geodesic_field(tet, bottom_cells):
        """Distance field via potpourri3d heat method, gradient via libigl's
        per-face gradient operator.

        Returns (cell_distance_to_bottom, cell_gradient).
        """

        surface = tet.extract_surface()
        V = np.asarray(surface.points, dtype=np.float64)
        z_min = V[:, 2].min()
        bottom_vertex_set = np.where(V[:, 2] < z_min + 0.3)[0]
        F = surface.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
        orig_face_cell_ids = surface.cell_data["vtkOriginalCellIds"]

        # Find bottom surface vertices
        # is_bottom_cell = np.zeros(tet.number_of_cells, dtype=bool)
        # is_bottom_cell[bottom_cells] = True
        # bottom_faces = is_bottom_cell[orig_face_cell_ids]
        # bottom_vertex_set = np.unique(F[bottom_faces].ravel())

        # Scalar distance via heat method
        heat_solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        vert_dist = heat_solver.compute_distance_multisource(bottom_vertex_set.tolist())

        # Per-face gradient via libigl's standard operator
        G = igl.grad(V, F)  # sparse (3*n_faces, n_verts)
        face_grad_flat = G @ vert_dist  # (3*n_faces,)
        face_grad = face_grad_flat.reshape(3, -1).T  # (n_faces, 3)

        face_dist = vert_dist[F].mean(axis=1)

        # Map face quantities back to tet cells (mean over each cell's surface faces)
        n_cells = tet.number_of_cells
        cell_distance_to_bottom = np.full(n_cells, np.inf)
        cell_gradient = np.full((n_cells, 3), np.nan)

        dist_sum = np.zeros(n_cells)
        grad_sum = np.zeros((n_cells, 3))
        cnt = np.zeros(n_cells, dtype=int)
        np.add.at(dist_sum, orig_face_cell_ids, face_dist)
        np.add.at(grad_sum, orig_face_cell_ids, face_grad)
        np.add.at(cnt, orig_face_cell_ids, 1)

        surface_mask = cnt > 0
        cell_distance_to_bottom[surface_mask] = (
            dist_sum[surface_mask] / cnt[surface_mask]
        )
        cell_gradient[surface_mask] = grad_sum[surface_mask] / cnt[surface_mask, None]

        valid_grad = cell_gradient[surface_mask]
        mags = np.linalg.norm(valid_grad, axis=1)
        print(
            f"Surface gradient magnitude: "
            f"min={mags.min():.3f}, median={np.median(mags):.3f}, "
            f"mean={mags.mean():.3f}, max={mags.max():.3f}"
        )
        viz.heat_field(tet, cell_distance_to_bottom, cell_gradient)

        return cell_distance_to_bottom, cell_gradient

    def update_tet_attributes(tet):
        with bench.stage("adjacency"):
            print("updating tet attributes...")
            curr_t = time.time()
            surface_mesh = tet.extract_surface()
            cell_to_face = decode_object(tet.field_data["cell_to_face"])

            cells = tet.cells.reshape(-1, 5)[:, 1:]
            tet.add_field_data(cells, "cells")
            cell_vertices = tet.points
            tet.add_field_data(cell_vertices, "cell_vertices")
            faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
            tet.add_field_data(faces, "faces")
            face_vertices = surface_mesh.points
            tet.add_field_data(face_vertices, "face_vertices")

            tet.cell_data["face_normal"] = np.full((tet.number_of_cells, 3), np.nan)
            keys = list(cell_to_face.keys())
            values = list(cell_to_face.values())
            all_face_indices = np.concatenate(values)
            cell_counts = [len(f) for f in values]
            all_cell_ids = np.repeat(keys, cell_counts)
            all_normals = surface_mesh.face_normals[all_face_indices]
            sort_idx = np.lexsort((all_normals[:, 2], all_cell_ids))
            sorted_ids = all_cell_ids[sort_idx]
            sorted_normals = all_normals[sort_idx]
            unique_ids, first_indices = np.unique(sorted_ids, return_index=True)
            tet.cell_data["face_normal"][unique_ids] = sorted_normals[first_indices]
            final_normals = tet.cell_data["face_normal"]
            tet.cell_data["face_normal"] = final_normals / np.clip(
                np.linalg.norm(final_normals, axis=1)[:, None], 1e-15, None
            )

            tet.cell_data["face_center"] = np.empty((tet.number_of_cells, 3))
            tet.cell_data["face_center"][:, :] = np.nan
            surface_mesh_cell_centers = surface_mesh.cell_centers().points

            all_centers = surface_mesh_cell_centers[all_face_indices]
            sort_idx_centers = np.lexsort((all_centers[:, 2], all_cell_ids))
            sorted_ids_centers = all_cell_ids[sort_idx_centers]
            sorted_centers = all_centers[sort_idx_centers]
            _, first_indices_centers = np.unique(sorted_ids_centers, return_index=True)

            tet.cell_data["face_center"] = np.full((tet.number_of_cells, 3), np.nan)
            tet.cell_data["face_center"][keys] = sorted_centers[first_indices_centers]

            tet.cell_data["cell_center"] = tet.cell_centers().points
            bottom_cell_threshold = np.nanmin(tet.cell_data["face_center"][:, 2]) + 0.3
            bottom_cells_mask = (
                tet.cell_data["face_center"][:, 2] < bottom_cell_threshold
            )
            tet.cell_data["is_bottom"] = bottom_cells_mask
            bottom_cells = np.where(bottom_cells_mask)[0]
            face_normals = tet.cell_data["face_normal"].copy()
            face_normals[bottom_cells_mask] = np.nan
            dots = np.sum(face_normals * up_vector, axis=1)
            tet.cell_data["overhang_angle"] = np.arccos(np.clip(dots, -1.0, 1.0))

            print(f"{time.time() - curr_t}")
            face_edges_s5 = tet.field_data["cell_face_neighbours"]

        bench.reference(
            "adjacency",
            s4=lambda: s4ref.compute_adjacency(tet)["face"],
            s5_result=face_edges_s5,
            correctness=bench.compare_edge_sets,
        )
        return tet

    def calculate_tet_attributes(tet):
        print("calc tet attributes...")
        curr_t = time.time()
        surface_mesh = tet.extract_surface()
        cells = tet.cells.reshape(-1, 5)[:, 1:]
        tet.add_field_data(cells, "cells")
        tet.add_field_data(tet.points, "cell_vertices")

        faces = surface_mesh.faces.reshape(-1, 4)[:, 1:]
        tet.add_field_data(faces, "faces")
        tet.add_field_data(surface_mesh.points, "face_vertices")

        from collections import defaultdict

        cell_to_face = defaultdict(list)
        face_to_cell = defaultdict(list)

        orig_cell_ids = surface_mesh.cell_data["vtkOriginalCellIds"]
        for face_idx, tet_cell_idx in enumerate(orig_cell_ids):
            cell_to_face[tet_cell_idx].append(face_idx)
            face_to_cell[face_idx].append(tet_cell_idx)

        tet.add_field_data(encode_object(dict(cell_to_face)), "cell_to_face")
        tet.add_field_data(encode_object(dict(face_to_cell)), "face_to_cell")

        tet.cell_data["has_face"] = np.zeros(tet.number_of_cells, dtype=int)
        tet.cell_data["has_face"][list(cell_to_face.keys())] = 1

        tet = update_tet_attributes(tet)

        bottom_cells_mask = tet.cell_data["is_bottom"]
        bottom_cells = np.where(bottom_cells_mask)[0]
        tet.cell_data["overhang_angle"][bottom_cells] = np.nan
        print(f"Bottom cells: {len(bottom_cells)}")
        bc_xy = input_tet.cell_data["face_center"][bottom_cells, :2]
        bc_z = input_tet.cell_data["face_center"][bottom_cells, 2]
        print(
            f"Bottom cell XY extent: x=[{bc_xy[:, 0].min():.1f}, {bc_xy[:, 0].max():.1f}], "
            f"y=[{bc_xy[:, 1].min():.1f}, {bc_xy[:, 1].max():.1f}]"
        )
        print(f"Bottom cell Z values: {np.unique(np.round(bc_z, 1))[:20]}")
        print(
            f"Threshold was: {np.nanmin(input_tet.cell_data['face_center'][:, 2]) + 0.3:.2f}"
        )

        print(time.time() - curr_t)

        return tet, bottom_cells_mask, bottom_cells

    input_tet, _, bottom_cells = calculate_tet_attributes(input_tet)

    # Geodesic field on the true surface (replaces old graph + Dijkstra setup)
    surface_distance, surface_gradient = build_surface_geodesic_field(
        input_tet, bottom_cells
    )
    input_tet.cell_data["surface_distance"] = surface_distance
    input_tet.cell_data["surface_gradient"] = surface_gradient

    viz.surface_gradient_field(input_tet)
    viz.radial_projection_field(input_tet)

    undeformed_tet = input_tet.copy()
    viz.overhang_analysis(input_tet)

    def calculate_path_length_to_base_gradient(
        tet,
        MAX_OVERHANG,
        INITIAL_ROTATION_FIELD_SMOOTHING,
        SET_INITIAL_ROTATION_TO_ZERO,
    ):
        grad = tet.cell_data[
            "surface_gradient"
        ]  # already unit vectors from build_surface_geodesic_field
        cc = tet.cell_data["cell_center"]
        dist = tet.cell_data["surface_distance"]

        xy_centroid = cc[:, :2].mean(axis=0)
        r_xy = cc[:, :2] - xy_centroid
        r_n = np.linalg.norm(r_xy, axis=1, keepdims=True) + 1e-8
        r_hat = r_xy / r_n

        # Use gradient magnitude in XY directly, signed by whether it points outward
        grad_xy_mag = np.linalg.norm(grad[:, :2], axis=1)
        outward_sign = np.sign(np.sum(r_hat * grad[:, :2], axis=1))
        raw = -outward_sign * grad_xy_mag  # magnitude from geodesic, sign from radial
        # dist_threshold = np.nanpercentile(dist, 5)  # bottom 15% by distance
        # near_base_mask = dist < dist_threshold
        # raw[near_base_mask] = 0.0

        # Also zero out NaN gradient cells (interior cells)
        # raw[np.isnan(grad[:, 0])] = 0.0

        return raw

    # def calculate_path_length_to_base_gradient(tet, MAX_OVERHANG,
    #                                             INITIAL_ROTATION_FIELD_SMOOTHING,
    #                                             SET_INITIAL_ROTATION_TO_ZERO):
    #     grad = tet.cell_data['surface_gradient']  # already unit vectors from build_surface_geodesic_field
    #     cc   = tet.cell_data['cell_center']
    #     dist = tet.cell_data['surface_distance']
    #
    #     xy_centroid = cc[:, :2].mean(axis=0)
    #     r_xy = cc[:, :2] - xy_centroid
    #     r_n  = np.linalg.norm(r_xy, axis=1, keepdims=True) + 1e-8
    #     r_hat = r_xy / r_n

    #     # Use gradient magnitude in XY directly, signed by whether it points outward
    #     grad_xy_mag = np.linalg.norm(grad[:, :2], axis=1)
    #     outward_sign = np.sign(np.sum(r_hat * grad[:, :2], axis=1))
    #     raw = -outward_sign * grad_xy_mag  # magnitude from geodesic, sign from radial
    #     #dist_threshold = np.nanpercentile(dist, 5)  # bottom 15% by distance
    #     #near_base_mask = dist < dist_threshold
    #     #raw[near_base_mask] = 0.0

    #     # Also zero out NaN gradient cells (interior cells)
    #     #raw[np.isnan(grad[:, 0])] = 0.0

    #     return raw

    def calculate_initial_rotation_field(
        tet,
        MAX_OVERHANG,
        ROTATION_MULTIPLIER,
        STEEP_OVERHANG_COMPENSATION,
        INITIAL_ROTATION_FIELD_SMOOTHING,
        SET_INITIAL_ROTATION_TO_ZERO,
        MAX_POS_ROTATION,
        MAX_NEG_ROTATION,
    ):
        print("calculate_initial_rotation_field")
        curr_t = time.time()

        initial_rotation_field = np.abs(
            np.deg2rad(90 + MAX_OVERHANG) - tet.cell_data["overhang_angle"]
        )

        path_length_to_base_gradient = calculate_path_length_to_base_gradient(
            tet,
            MAX_OVERHANG,
            INITIAL_ROTATION_FIELD_SMOOTHING,
            SET_INITIAL_ROTATION_TO_ZERO,
        )

        if STEEP_OVERHANG_COMPENSATION:
            in_air_mask = tet.cell_data.get(
                "in_air", np.zeros(tet.number_of_cells, dtype=bool)
            )
            initial_rotation_field[in_air_mask] += 2 * (
                np.deg2rad(180) - tet.cell_data["overhang_angle"][in_air_mask]
            )

        initial_rotation_field *= path_length_to_base_gradient
        initial_rotation_field = np.clip(
            initial_rotation_field * ROTATION_MULTIPLIER,
            -np.deg2rad(360),
            np.deg2rad(360),
        )
        initial_rotation_field = np.clip(
            initial_rotation_field, MAX_NEG_ROTATION, MAX_POS_ROTATION
        )

        tet.cell_data["initial_rotation_field"] = initial_rotation_field
        print(time.time() - curr_t)
        return initial_rotation_field

    def calculate_rotation_matrices(tet, rotation_field):
        print("calculate_rotation_matrices")
        curr_t = time.time()
        tangential_vectors = np.cross(
            np.array([0, 0, 1]), tet.cell_data["cell_center"][:, :2]
        )
        tangential_vectors /= np.linalg.norm(tangential_vectors, axis=1)[:, None]
        tangential_vectors[np.isnan(tangential_vectors).any(axis=1)] = [1, 0, 0]

        rotation_matrices = R.from_rotvec(
            rotation_field[:, None] * tangential_vectors
        ).as_matrix()
        print(time.time() - curr_t)

        return rotation_matrices

    def optimize_rotations(
        tet,
        NEIGHBOR_LOSS_WEIGHT,
        MAX_OVERHANG,
        ROTATION_MULTIPLIER,
        ITERATIONS,
        SAVE_GIF,
        STEEP_OVERHANG_COMPENSATION,
        INITIAL_ROTATION_FIELD_SMOOTHING,
        SET_INITIAL_ROTATION_TO_ZERO,
        MAX_POS_ROTATION,
        MAX_NEG_ROTATION,
    ):
        print("optimized_rotations")
        cell_face_nb = tet.field_data["cell_face_neighbours"]
        n_c = tet.number_of_cells
        W = float(NEIGHBOR_LOSS_WEIGHT)

        with bench.stage("smoothing"):
            curr_t = time.time()
            initial_rotation_field = calculate_initial_rotation_field(
                tet,
                MAX_OVERHANG,
                ROTATION_MULTIPLIER,
                STEEP_OVERHANG_COMPENSATION,
                INITIAL_ROTATION_FIELD_SMOOTHING,
                SET_INITIAL_ROTATION_TO_ZERO,
                MAX_POS_ROTATION,
                MAX_NEG_ROTATION,
            )

            # OPT: ReplaceottomF iterative least_squares with a single direct sparse solve.
            #
            # Why this works: the original objective is Laplacian smoothing —
            #   W * Σ_(i,j face-neighbors) (r_i − r_j)²  +  Σ_(valid i) (r_i − init_i)²
            # This is purely quadratic, so its minimiser satisfies the linear system:
            #   (W · L_graph  +  diag(valid_mask)) r  =  diag(valid_mask) · r_init
            # where L_graph is the weighted graph Laplacian over face-neighbor pairs.
            #
            # Previously least_squares rebuilt a full COO→CSR Jacobian on each of up to
            # ITERATIONS TRF evaluations.  spsolve does one UMFPACK LU factorisation and
            # back-substitution — typically < 0.1 s vs 20 + s.

            from scipy.sparse.linalg import spsolve as _spsolve

            cell_face_nb = tet.field_data["cell_face_neighbours"]
            n_c = tet.number_of_cells
            valid_mask = ~np.isnan(initial_rotation_field)
            W = float(NEIGHBOR_LOSS_WEIGHT)

            ea = cell_face_nb[:, 0]  # one endpoint of each unique face-edge
            eb = cell_face_nb[:, 1]
            n_e = len(ea)

            # Degree = number of face-neighbours per cell (each undirected edge counted once,
            # so we add 1 to both endpoints)
            degree = np.zeros(n_c, dtype=np.float64)
            np.add.at(degree, ea, 1.0)
            np.add.at(degree, eb, 1.0)

            # Diagonal: W * degree + anchor-weight(1 if valid else 0) + tiny eps for stability
            diag_v = W * degree + valid_mask.astype(np.float64) + 1e-10

            all_r = np.concatenate([ea, eb, np.arange(n_c, dtype=np.int32)])
            all_c = np.concatenate([eb, ea, np.arange(n_c, dtype=np.int32)])
            all_d = np.concatenate([-W * np.ones(n_e), -W * np.ones(n_e), diag_v])

            A_rot = sparse.coo_matrix(
                (all_d, (all_r, all_c)), shape=(n_c, n_c), dtype=np.float64
            ).tocsr()
            b_rot = np.where(
                valid_mask, np.nan_to_num(initial_rotation_field, nan=0.0), 0.0
            )

            rotation_field = _spsolve(A_rot, b_rot)
            viz.initial_rotation_field(tet, initial_rotation_field)
            viz.smoothed_rotation_field(tet, rotation_field)
            viz.rotation_field_comparison(tet, initial_rotation_field, rotation_field)

            print(time.time() - curr_t)

        bench.reference(
            "smoothing",
            s4=lambda: s4ref.smooth_rotation_field(
                n_c, cell_face_nb, initial_rotation_field, W, max_nfev=100, ftol=1e-6
            ),
            s5_result=rotation_field,
            correctness=bench.compare_scalar_fields,
        )
        return rotation_field

    NEIGHBOR_LOSS_WEIGHT = neighbor_loss_weight
    # MAX_OVERHANG = 30
    MAX_OVERHANG = max_overhang
    # ROTATION_MULTIPLIER = 2
    ROTATION_MULTIPLIER = rotation_multiplier
    SET_INITIAL_ROTATION_TO_ZERO = False
    INITIAL_ROTATION_FIELD_SMOOTHING = 30
    MAX_POS_ROTATION = np.deg2rad(360)
    MAX_NEG_ROTATION = np.deg2rad(-360)
    ITERATIONS = 100
    SAVE_GIF = True
    STEEP_OVERHANG_COMPENSATION = True

    rotation_field = optimize_rotations(
        undeformed_tet,
        NEIGHBOR_LOSS_WEIGHT,
        MAX_OVERHANG,
        ROTATION_MULTIPLIER,
        ITERATIONS,
        SAVE_GIF,
        STEEP_OVERHANG_COMPENSATION,
        INITIAL_ROTATION_FIELD_SMOOTHING,
        SET_INITIAL_ROTATION_TO_ZERO,
        MAX_POS_ROTATION,
        MAX_NEG_ROTATION,
    )

    # After optimize_rotations returns rotation_field
    cc = undeformed_tet.cell_data["cell_center"]

    # Partition cells by body region using cc position
    foot_mask = cc[:, 2] < 5  # feet: bottom 5mm
    head_mask = cc[:, 2] > 40  # head: top portion (adjust based on mesh height)
    arm_mask = (
        (cc[:, 2] > 15) & (cc[:, 2] < 30) & (np.abs(cc[:, 0]) > 15)
    )  # arms: lateral at mid-height
    body_mask = (
        (cc[:, 2] > 10) & (cc[:, 2] < 30) & (np.abs(cc[:, 0]) < 10)
    )  # central body

    for name, mask in [
        ("feet", foot_mask),
        ("arms", arm_mask),
        ("body", body_mask),
        ("head", head_mask),
    ]:
        if mask.sum() == 0:
            continue
        r = rotation_field[mask]
        print(
            f"{name:6s} (n={mask.sum():5d}): "
            f"mean |r|={np.abs(r).mean():.3f} rad, "
            f"max |r|={np.abs(r).max():.3f} rad"
        )

    N = np.eye(4) - 1 / 4 * np.ones((4, 4))

    def calculate_deformation(tet, rotation_field, ITERATIONS, SAVE_GIF):
        """Deformation via a direct sparse solve of the shape-matching normal
        equations — replaces the previous iterative lsqr.

        Objective (S5.pdf §4.5): min_V' Σ_c ‖R_c N v_c − N v'_c‖²_F, which
        decomposes into 3 INDEPENDENT linear systems (one per spatial dim):

            min_vj ‖A v_j − b_j‖²   →   normal equations   AᵀA v_j = Aᵀ b_j

        A is built ONCE from topology + the centering operator N.  It has a 1-D
        null space (a global translation leaves every centered shape unchanged),
        so AᵀA has rank n_pts−1.  We pin one vertex to its current position,
        which makes the reduced system SPD; a single sparse LU factorisation is
        then reused across all three RHS columns.

        Why direct, not lsqr:  the previous lsqr (iter_lim=1000) was badly
        non-converged on large meshes.  On the 182k-cell tree the direct solve
        is the exact minimiser and ~4× fewer cells invert (2470 → 620) — though
        it does NOT eliminate inversions when the prescribed rotation field is
        very large (≈160°); that is a formulation limit, not a solver one (see
        OPTIMIZE.md finding 6).  This mirrors the rotation-smoothing stage
        (§4.4), which already uses spsolve.  ITERATIONS is retained for
        call-site compatibility but no longer used.

        Upgrade paths for very large models (n_pts ≳ 5·10⁵):
          • scikit-sparse CHOLMOD: supernodal Cholesky of the SPD AᵀA — faster
            than splu and reuses the symbolic factorisation.
          • PyPardiso (Intel MKL PARDISO): fastest multi-threaded CPU solver.
        """
        with bench.stage("deformation"):
            print("calculate_deformations")
            curr_t = time.time()

            n_cells = tet.number_of_cells
            n_pts = tet.number_of_points
            cells = tet.field_data["cells"]  # (n_cells, 4)

            rotation_matrices = calculate_rotation_matrices(tet, rotation_field)

            # Pre-compute per-cell rotation targets
            # old_verts_transformed[c, j, i] = target for cell c, dim j, local vertex i
            old_verts = tet.field_data["cell_vertices"][cells]  # (n_cells, 4, 3)
            old_verts_transformed = np.einsum(
                "ijk,ikl->ijl", rotation_matrices, (N @ old_verts).transpose(0, 2, 1)
            )  # (n_cells, 3, 4)

            # Build sparse A once — depends only on topology + N.
            # A[c*4 + i,  cells[c, k]]  =  N[i, k]
            # N = I4 − 0.25·ones4  →  N[i,k] = 0.75 if i==k else −0.25
            c_idx = np.arange(n_cells)
            _rows, _cols, _vals = [], [], []
            for i in range(4):
                for k in range(4):
                    _rows.append(c_idx * 4 + i)
                    _cols.append(cells[:, k])
                    _vals.append(
                        np.full(n_cells, 0.75 if i == k else -0.25, dtype=np.float64)
                    )
            A_deform = sparse.coo_matrix(
                (np.concatenate(_vals), (np.concatenate(_rows), np.concatenate(_cols))),
                shape=(n_cells * 4, n_pts),
                dtype=np.float64,
            ).tocsr()

            # RHS: B[c*4 + i, j] = old_verts_transformed[c, j, i]
            B_target = np.ascontiguousarray(
                old_verts_transformed.transpose(0, 2, 1).reshape(n_cells * 4, 3)
            )

            # Normal equations.  AᵀA is symmetric PSD with a 1-D (translation)
            # null space; pinning one vertex removes it → SPD.
            AtA = (A_deform.T @ A_deform).tocsc()
            AtB = np.asarray(A_deform.T @ B_target)  # (n_pts, 3) dense

            # Pin the lowest-z vertex to its current position.  The absolute
            # location is arbitrary (the mesh is re-centred downstream), so any
            # vertex works; lowest-z is a stable, deterministic choice.
            anchor = int(np.argmin(tet.points[:, 2]))
            free_idx = np.delete(np.arange(n_pts), anchor)
            anchor_pos = tet.points[anchor]  # (3,)

            AtA_csr = AtA.tocsr()
            AtA_ff = AtA_csr[free_idx][:, free_idx].tocsc()
            # Pinned column → RHS:  (AᵀA v)_free = AtB_free − AtA[free, anchor]·anchor_pos
            anchor_col = np.asarray(AtA[:, anchor].todense()).ravel()[free_idx]

            # One LU factorisation reused across all three coordinate solves.
            lu = sparse.linalg.splu(AtA_ff)
            new_verts = np.empty((n_pts, 3), dtype=np.float64)
            new_verts[anchor] = anchor_pos
            for j in range(3):
                rhs = AtB[free_idx, j] - anchor_col * anchor_pos[j]
                new_verts[free_idx, j] = lu.solve(rhs)

            # Diagnostic: a correct solve must not flip any cell.  F_c = ∇(new)
            # evaluated on the *undeformed* config (tet.points here); det<0 ⇒
            # inverted tet, det away from 1 ⇒ local volume change.  The previous
            # lsqr left hundreds of inverted tets on large meshes; this should be 0.
            _G = igl.grad(np.asarray(tet.points, float), cells.astype(np.int64))
            _gg = _G @ new_verts
            _F = np.stack(
                [_gg[:n_cells], _gg[n_cells : 2 * n_cells], _gg[2 * n_cells :]], axis=1
            )
            _det = np.linalg.det(_F)
            print(
                f"  deformation: inverted tets={int((_det < 0).sum())}, "
                f"vol-ratio min/median/max="
                f"{_det.min():.2f}/{np.median(_det):.2f}/{_det.max():.2f}"
            )

            print(time.time() - curr_t)
        bench.reference(
            "deformation",
            s4=lambda: s4ref.solve_deformation(
                points=tet.points.copy(),
                cells=cells,  # already (n_cells, 4)
                cell_centers=tet.cell_data["cell_center"],
                rotation_field=rotation_field,
                max_nfev=1000,
                ftol=1e-6,
            ),
            s5_result=new_verts,
            correctness=bench.compare_vector_fields,
        )
        return new_verts

    ITERATIONS = 1000
    SAVE_GIF = True
    new_vertices = calculate_deformation(
        undeformed_tet, rotation_field, ITERATIONS, SAVE_GIF
    )
    deformed_tet = pv.UnstructuredGrid(
        undeformed_tet.cells,
        np.full(undeformed_tet.number_of_cells, pv.CellType.TETRA),
        new_vertices,
    )

    # Check if displacement is dominated by rigid-body motion
    disp = deformed_tet.points - input_tet.points
    print(f"Mean displacement: {disp.mean(axis=0)}")  # should be ~0 if pure deformation
    print(
        f"Displacement magnitude: mean={np.linalg.norm(disp, axis=1).mean():.3f}, "
        f"max={np.linalg.norm(disp, axis=1).max():.3f}"
    )

    # Check symmetry: for each point (x, y, z), find the point closest to (-x, y, z) and compare displacements
    pts = input_tet.points
    mirror_pts = pts.copy()
    mirror_pts[:, 0] *= -1  # mirror across Y-Z plane
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    _, mirror_idx = tree.query(mirror_pts)
    # displacement at mirrored pair should have x-component sign-flipped, y and z same
    disp_mirrored = disp[mirror_idx].copy()
    disp_mirrored[:, 0] *= -1
    asymmetry = np.linalg.norm(disp - disp_mirrored, axis=1)
    print(f"Symmetry error: mean={asymmetry.mean():.3f}, max={asymmetry.max():.3f}")
    print(
        f"Displacement magnitude for comparison: mean={np.linalg.norm(disp, axis=1).mean():.3f}"
    )

    viz.deformation_pair(undeformed_tet, deformed_tet)
    viz.deformation_displacement(undeformed_tet, deformed_tet)

    for key in undeformed_tet.field_data.keys():
        deformed_tet.field_data[key] = undeformed_tet.field_data[key]
    for key in undeformed_tet.cell_data.keys():
        deformed_tet.cell_data[key] = undeformed_tet.cell_data[key]
    deformed_tet = update_tet_attributes(deformed_tet)

    x_min, x_max, y_min, y_max, z_min, z_max = deformed_tet.bounds
    offsets_applied = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
    deformed_tet.points -= offsets_applied

    deformed_tet.extract_surface().save(f"{output_path}/{model_name}_deformed_tet.stl")

    with open(f"{output_path}/deformed_{model_name}.pkl", "wb") as f:
        pickle.dump(deformed_tet, f)

    print("Slicing with Cura...")
    abs_extruder = os.path.abspath(extruder_path)
    abs_printer = os.path.abspath(printer_path)
    abs_model = os.path.abspath(f"{output_path}/{model_name}_deformed_tet.stl")
    abs_output = os.path.abspath(f"{output_path}/{model_name}_deformed_tet.gcode")

    # setting redundancy
    with open(config_path, "r") as jfh:
        slicer_config = json.load(jfh)

    args = []

    # Support your existing format: settings.global.all
    settings = slicer_config["settings"]["global"]["all"]
    for setting_name, data in settings.items():
        if isinstance(data, dict):
            val = (
                data.get("value")
                if data.get("value") is not None
                else data.get("default_value")
            )
        else:
            val = data
        if val is not None:
            args.extend(["-s", f"{setting_name}={val}"])

    # NOTE: `-j` registers a CuraEngine *definition* file (fdmprinter/fdmextruder
    # format).  config_path (core.def.json) is a Cura GUI settings export, not a
    # definition — its values are already applied below via `-s` flags, and
    # passing it with `-j` aborts the 5.0.0 engine ("setting with no value given").
    command = [
        cura_path,
        "slice",
        "-j",
        abs_printer,
        "-j",
        abs_extruder,
    ]
    command.extend(args)
    command.extend(["-l", abs_model, "-o", abs_output])

    # Run CuraEngine and *validate* it actually produced G-code.  Cura streams
    # its progress to stderr; capture it to a log so a version/settings failure
    # surfaces here with a clear message instead of crashing later at G-code
    # parse time on an empty/missing file.  (See OPTIMIZE.md §2 for the Debian
    # 5.0.0-engine vs bundled 5.11-definition mismatch this guards against.)
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"CuraEngine not found at '{cura_path}'. Install the cura-engine "
            f"package (Debian: `apt install cura-engine`) or pass --cura <path>."
        ) from exc

    cura_log = os.path.join(output_path, f"{model_name}_cura.log")
    with open(cura_log, "w") as _lf:
        _lf.write(proc.stdout or "")
        _lf.write(proc.stderr or "")

    produced = os.path.isfile(abs_output) and os.path.getsize(abs_output) > 0
    if proc.returncode != 0 or not produced:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-15:])
        raise RuntimeError(
            "CuraEngine failed to slice the deformed mesh "
            f"(exit code {proc.returncode}; output G-code "
            f"{'missing/empty' if not produced else 'written'}).\n"
            "This is usually a Cura version/definition mismatch — the engine here "
            "is 5.0.0 while the bundled config/*.def.json are exported from 5.11 "
            f"(see OPTIMIZE.md §2). Full CuraEngine log: {cura_log}\n"
            f"--- CuraEngine output (last lines) ---\n{tail}"
        )

    deformed_tet = pickle.load(open(f"{output_path}/deformed_{model_name}.pkl", "rb"))

    deformed_tet, _, _ = calculate_tet_attributes(deformed_tet)
    from pygcode import Line

    SEG_SIZE = 0.6
    MAX_ROTATION = 30
    MIN_ROTATION = -130
    NOZZLE_OFFSET = nozzle_offset

    vertex_transformations = deformed_tet.points - input_tet.points
    tangential_vectors = np.cross(
        np.array([0, 0, 1]), input_tet.cell_data["cell_center"][:, :2]
    )
    tangential_vectors /= np.linalg.norm(tangential_vectors, axis=1)[:, None]
    tangential_vectors[np.isnan(tangential_vectors).any(axis=1)] = [1, 0, 0]

    # OPT: Replace Python loop with np.bincount — O(n) numpy instead of O(n) Python.
    _cells_flat = input_tet.field_data["cells"].ravel()
    num_cells_per_vertex = np.bincount(
        _cells_flat, minlength=input_tet.number_of_points
    ).astype(float)

    # OPT: Vectorized vertex rotation computation — batch SVD over all cells at once.
    # Replaces a Python for-loop with per-cell SVD, covariance, and scatter-add.
    _cells_arr = deformed_tet.field_data["cells"]  # (n_cells, 4)
    _n_cells_d = deformed_tet.number_of_cells

    _new_verts = deformed_tet.field_data["cell_vertices"][_cells_arr]  # (n_cells, 4, 3)
    _old_verts = input_tet.field_data["cell_vertices"][_cells_arr]  # (n_cells, 4, 3)
    _new_cc = deformed_tet.cell_data["cell_center"]  # (n_cells, 3)
    _old_cc = input_tet.cell_data["cell_center"]  # (n_cells, 3)

    # Centre vertices around their cell centroid
    _new_verts -= _new_cc[:, None, :]
    _old_verts -= _old_cc[:, None, :]

    # Build per-cell plane_x_vector (normalised radial direction in XY, z=0)
    _xy_norm = np.linalg.norm(_old_cc[:, :2], axis=1, keepdims=True) + 1e-8
    _plane_x = np.concatenate(
        [_old_cc[:, :2] / _xy_norm, np.zeros((_n_cells_d, 1))], axis=1
    )  # (n_cells, 3)
    # plane_y = [0, 0, 1] for all cells

    # Project vertices onto (plane_x, plane_y=Z-axis) — shape (n_cells, 4, 2)
    _new_px = np.einsum("ni,nki->nk", _plane_x, _new_verts)  # dot with plane_x
    _new_py = _new_verts[:, :, 2]  # dot with [0,0,1]
    _new_proj = np.stack([_new_px, _new_py], axis=-1)

    _old_px = np.einsum("ni,nki->nk", _plane_x, _old_verts)
    _old_py = _old_verts[:, :, 2]
    _old_proj = np.stack([_old_px, _old_py], axis=-1)

    # Batch covariance (n_cells, 2, 2) and SVD → rotation matrices
    _cov = np.einsum("nki,nkj->nij", _new_proj, _old_proj)
    _U, _, _Vt = np.linalg.svd(_cov)
    _rot_mats = np.einsum("nij,njk->nik", _U, _Vt)  # (n_cells, 2, 2)

    # Extract signed rotation angles from 2×2 rotation matrices
    _cos_val = np.clip(_rot_mats[:, 0, 0], -1.0, 1.0)
    cell_rotations = -np.arccos(_cos_val)
    cell_rotations[_rot_mats[:, 1, 0] < 0] *= -1
    cell_rotations = np.clip(
        cell_rotations, np.deg2rad(MIN_ROTATION), np.deg2rad(MAX_ROTATION)
    )

    # Scatter-add per-vertex average rotation using np.add.at
    vertex_rotations = np.zeros(deformed_tet.number_of_points)
    _contrib = (
        cell_rotations[:, None] / num_cells_per_vertex[_cells_arr]
    )  # (n_cells, 4)
    np.add.at(vertex_rotations, _cells_arr, _contrib)
    viz.vertex_rotations(deformed_tet, vertex_rotations)

    tet_rotation_matrices = calculate_rotation_matrices(input_tet, cell_rotations)

    # OPT: Vectorized z_squish_scales — batch tetrahedron volume via np.linalg.det.
    # Replaces a Python loop with per-cell volume calculations.
    def _batch_tet_volume(verts4):
        """verts4: (n, 4, 3) → unsigned tet volumes (n,)"""
        d = verts4[:, 1:] - verts4[:, :1]  # (n, 3, 3)
        return np.abs(np.linalg.det(d)) / 6.0

    _warp_v = deformed_tet.field_data["cell_vertices"][_cells_arr]  # (n_cells, 4, 3)
    _unwarp_v = input_tet.field_data["cell_vertices"][_cells_arr]  # (n_cells, 4, 3)
    _vol_unwarp = _batch_tet_volume(_unwarp_v)
    _vol_warp = _batch_tet_volume(_warp_v)
    z_squish_scales = _vol_unwarp / (_vol_warp + 1e-15)
    viz.volume_scaling(deformed_tet, z_squish_scales)

    pos = np.array([0.0, 0.0, 20.0])
    feed = 5000
    gcode_points = []
    # Track CuraEngine's ;LAYER:/;TYPE: comments so the Unlayered writer can emit
    # ;LAYER_CHANGE / ;TYPE annotations; ignored by the polar (rtheta) writer.
    cur_layer = None
    cur_type = None
    with open(f"{output_path}/{model_name}_deformed_tet.gcode", "r") as fh:
        for line_text in fh.readlines():
            line = Line(line_text)

            if line.comment is not None:
                txt = (line.comment.text or "").strip()
                if txt.startswith("LAYER:"):
                    try:
                        cur_layer = int(txt[len("LAYER:") :])
                    except ValueError:
                        pass
                elif txt.startswith("TYPE:"):
                    cur_type = txt[len("TYPE:") :].strip()

            if not line.block.gcodes:
                continue

            for gcode in sorted(line.block.gcodes):
                if gcode.word == "G01" or gcode.word == "G00":
                    prev_pos = pos.copy()

                    if gcode.X is not None:
                        pos[0] = gcode.X
                    if gcode.Y is not None:
                        pos[1] = gcode.Y
                    if gcode.Z is not None:
                        pos[2] = gcode.Z

                    inv_time_feed = None
                    for word in line.block.words:
                        if word.letter == "F":
                            feed = word.value

                    extrusion = None
                    for param in line.block.modal_params:
                        if param.letter == "E":
                            extrusion = param.value

                    delta_pos = pos - prev_pos
                    distance = np.linalg.norm(delta_pos)
                    if distance > 0:
                        num_segments = -(-distance // SEG_SIZE)
                        seg_distance = distance / num_segments

                        time_to_complete_move = (1 / feed) * seg_distance
                        if time_to_complete_move == 0:
                            inv_time_feed = None
                        else:
                            inv_time_feed = 1 / time_to_complete_move

                        for i in range(int(num_segments)):
                            gcode_points.append(
                                {
                                    "position": (
                                        prev_pos + delta_pos * (i + 1) / num_segments
                                    ),
                                    "command": gcode.word,
                                    "extrusion": extrusion / num_segments
                                    if extrusion is not None
                                    else None,
                                    "inv_time_feed": inv_time_feed,
                                    "move_length": seg_distance,
                                    "start_position": prev_pos,
                                    "end_position": pos,
                                    "unsegmented_move_length": distance,
                                    "after_retract": False,
                                    "feed": feed,
                                    "layer": cur_layer,
                                    "type": cur_type,
                                }
                            )
                    else:
                        time_to_complete_move = (1 / feed) * distance
                        if time_to_complete_move == 0:
                            inv_time_feed = None
                        else:
                            inv_time_feed = 1 / time_to_complete_move

                        gcode_points.append(
                            {
                                "position": pos.copy(),
                                "command": gcode.word,
                                "extrusion": extrusion,
                                "inv_time_feed": inv_time_feed,
                                "move_length": distance,
                                "unsegmented_move_length": distance,
                                "after_retract": False,
                                "feed": feed,
                                "layer": cur_layer,
                                "type": cur_type,
                            }
                        )

    _positions_all = np.array([p["position"] for p in gcode_points])  # (n_pts, 3)
    gcode_points_containing_cells = deformed_tet.find_containing_cell(_positions_all)
    gcode_points_closest_cells = deformed_tet.find_closest_cell(_positions_all)

    # OPT: Pre-compute (new_position, rotation) for every gcode point using batched linear solves.
    # The original called np.linalg.solve once per point inside the sequential loop (40s).
    # Here we solve all points at once with np.linalg.solve on a batched (n, 3, 3) system,
    # then the sequential loop only handles smoothing / travelling logic using pre-computed values.
    print("Pre-computing barycentric transforms (batch)...")
    _n_pts = len(gcode_points)
    _commands_all = [p["command"] for p in gcode_points]
    _containing = np.array(gcode_points_containing_cells, dtype=np.int32)
    _closest = np.array(gcode_points_closest_cells, dtype=np.int32)
    _is_g01 = np.array([c == "G01" for c in _commands_all])

    # Determine effective cell per point (mirrors barycentric_interpolate logic):
    #  G00 + containing == -1  → eff = -1  (will produce None result)
    #  G01 + containing == -1  → eff = closest cell
    #  otherwise               → eff = containing cell
    _eff_cells = _containing.copy()
    _fallback = _is_g01 & (_containing == -1)
    _eff_cells[_fallback] = _closest[_fallback]

    _valid = _eff_cells >= 0  # points we can solve
    _valid_idx = np.where(_valid)[0]

    _cells_data = deformed_tet.field_data["cells"]
    _verts_data = deformed_tet.field_data["cell_vertices"]

    _vc = _eff_cells[_valid_idx]
    _vert_indices = _cells_data[_vc]  # (n_valid, 4)
    _cell_verts = _verts_data[_vert_indices]  # (n_valid, 4, 3)

    # Build batched T matrices: each column of T[i] is (vert_j - vert_d) for j in {a,b,c}
    _tet_d = _cell_verts[:, 3]  # (n_valid, 3)
    _T = (_cell_verts[:, :3] - _cell_verts[:, 3:4]).transpose(
        0, 2, 1
    )  # (n_valid, 3, 3)
    _rhs = _positions_all[_valid_idx] - _tet_d  # (n_valid, 3)

    # Batch solve — numpy's gufunc signature is (m,m),(m,n)->(m,n), so b must be
    # (..., m, n), NOT (..., m).  Adding a trailing K=1 column makes n=1 explicit;
    # squeeze removes it afterward.  Without this, numpy reads the batch axis as
    # the core "m" dimension and raises a shape-mismatch ValueError.
    with bench.stage("barycentric"):
        _lambdas = np.zeros((len(_valid_idx), 3))
        _det = np.linalg.det(_T)
        _ok = np.abs(_det) > 1e-10
        if _ok.all():
            # Fast path: skip the mask, solve the whole batch in place
            _lambdas = np.linalg.solve(_T, _rhs[:, :, None]).squeeze(-1)
        elif _ok.any():
            _lambdas = np.zeros((len(_valid_idx), 3))
            _lambdas[_ok] = np.linalg.solve(_T[_ok], _rhs[_ok, :, None]).squeeze(-1)
        else:
            _lambdas = np.zeros((len(_valid_idx), 3))

        _bary = np.empty((len(_valid_idx), 4))
        _bary[:, :3] = _lambdas
        _bary[:, 3] = 1.0 - _lambdas.sum(axis=1)
        _bary_ok = _bary.sum(axis=1) <= 1.01  # mirrors `> 1.01` → None

        # Compute pre-transformed positions and rotations
        _vt = vertex_transformations[_vert_indices]  # (n_valid, 4, 3)
        _pre_pos = _positions_all[_valid_idx] - (_vt * _bary[:, :, None]).sum(axis=1)
        _pre_pos[~_bary_ok] = np.nan

        _vr = vertex_rotations[_vert_indices]  # (n_valid, 4)
        _pre_rot = (_vr * _bary).sum(axis=1)
        _pre_rot[~_bary_ok] = np.nan

        # Map results back to all-point arrays (NaN = None in original code)
        pre_new_positions = np.full((_n_pts, 3), np.nan)
        pre_rotations = np.full(_n_pts, np.nan)
        pre_new_positions[_valid_idx] = _pre_pos
        pre_rotations[_valid_idx] = _pre_rot

        # OPT: Pre-compute validity as a boolean array — avoids calling np.any(np.isnan(...))
        # (which allocates a small array) on every single iteration of the reforming loop.
        _pre_valid = ~np.any(np.isnan(pre_new_positions), axis=1)  # (n_pts,) bool

    bench.reference(
        "barycentric",
        s4=lambda: s4ref.compute_barycentric_point_transforms(
            cell_vertices=_cell_verts, query_points=_positions_all[_valid_idx]
        ),
        s5_result=_bary,
        correctness=bench.compare_vector_fields,
    )

    print("Reforming...")
    prev_time = time.time()
    new_gcode_points = []
    prev_new_position = None
    travelling_over_air = False
    travelling = False
    prev_position = None
    prev_rotation = 0
    prev_travelling = False
    prev_command = "G00"
    ROTATION_AVERAGING_ALPHA = 0.2
    RETRACTION_LENGTH = 1.0
    ROTATION_MAX_DELTA = np.deg2rad(1)
    MAX_EXTRUSION_MULTIPLIER = 10
    lost_vertices = []
    highest_printed_point = 0
    for cell_index, (gcode_point, containing_cell_index) in enumerate(
        zip(gcode_points, gcode_points_containing_cells)
    ):
        position = gcode_point["position"]
        command = gcode_point["command"]
        inv_time_feed = gcode_point["inv_time_feed"]
        extrusion = gcode_point["extrusion"]

        # OPT: Replace per-point barycentric solve with pre-computed lookup.
        _pnp = pre_new_positions[cell_index]
        _pr = pre_rotations[cell_index]
        _pos_valid = bool(_pre_valid[cell_index])  # O(1) boolean array lookup

        dont_smooth_rotation = False
        if _pos_valid:
            new_position = _pnp.copy()
            rotation = float(_pr)
        else:
            new_position = None
            rotation = None
        if new_position is None:
            if command == "G01":
                lost_vertices.append(position)
                continue
            elif (
                command == "G00"
                and not travelling_over_air
                and prev_new_position is not None
            ):
                new_position = np.array(
                    [prev_new_position[0], prev_new_position[1], highest_printed_point]
                )
                rotation = max(min(prev_rotation, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth_rotation = True
                travelling_over_air = True
            elif travelling_over_air:
                continue
            else:
                continue
        else:
            if travelling_over_air:
                new_position[2] = highest_printed_point
                rotation = max(min(rotation, np.deg2rad(45)), np.deg2rad(-45))
                dont_smooth_rotation = True
            travelling_over_air = False

        extrusion_multiplier = 1
        if (
            extrusion is not None
            and extrusion != RETRACTION_LENGTH
            and extrusion != -RETRACTION_LENGTH
        ):
            extrusion_multiplier = (
                extrusion_multiplier * z_squish_scales[containing_cell_index]
            )
            extrusion = extrusion * min(extrusion_multiplier, MAX_EXTRUSION_MULTIPLIER)
        elif extrusion == -RETRACTION_LENGTH:
            travelling = True
        elif extrusion == RETRACTION_LENGTH:
            travelling = False
        if prev_rotation is not None and not dont_smooth_rotation:
            rotation = (
                ROTATION_AVERAGING_ALPHA * rotation
                + (1 - ROTATION_AVERAGING_ALPHA) * prev_rotation
            )

        if (
            prev_rotation is not None
            and prev_new_position is not None
            and np.abs(rotation - prev_rotation) > ROTATION_MAX_DELTA
        ):
            delta_rotation = rotation - prev_rotation
            num_interpolations = int(np.abs(delta_rotation) / ROTATION_MAX_DELTA) + 1
            delta_pos = new_position - prev_new_position
            for i in range(num_interpolations):
                new_gcode_points.append(
                    {
                        "position": prev_new_position
                        + (delta_pos * ((i + 1) / num_interpolations)),
                        "original_position": position,
                        "rotation": prev_rotation
                        + (delta_rotation * ((i + 1) / num_interpolations)),
                        "command": prev_command,
                        "extrusion": extrusion / num_interpolations
                        if extrusion is not None
                        else None,
                        "inv_time_feed": inv_time_feed * num_interpolations
                        if inv_time_feed is not None
                        else None,
                        "extrusion_multiplier": extrusion_multiplier,
                        "feed": gcode_point["feed"],
                        "travelling": prev_travelling,
                        "layer": gcode_point.get("layer"),
                        "type": gcode_point.get("type"),
                    }
                )
        else:
            new_gcode_points.append(
                {
                    "position": new_position,
                    "original_position": position,
                    "rotation": rotation,
                    "command": command,
                    "extrusion": extrusion,
                    "inv_time_feed": inv_time_feed,
                    "extrusion_multiplier": extrusion_multiplier,
                    "feed": gcode_point["feed"],
                    "travelling": travelling,
                    "layer": gcode_point.get("layer"),
                    "type": gcode_point.get("type"),
                }
            )

        prev_rotation = rotation
        prev_new_position = new_position.copy()
        prev_travelling = travelling
        prev_command = command

        if (
            command == "G01"
            and extrusion is not None
            and extrusion > 0
            and (highest_printed_point != 0 or new_position[2] < 1)
        ):
            highest_printed_point = max(highest_printed_point, new_position[2])

    print(f"Lost {len(lost_vertices)} vertices")
    print(f"Reforming done: {round(time.time() - prev_time, 2)}s")
    viz.gcode_toolpaths(
        new_gcode_points, background_mesh=deformed_tet, extrusion_only=True
    )
    viz.gcode_layer_build(new_gcode_points, background_mesh=deformed_tet)
    prev_time = time.time()

    out_gcode_path = f"{output_path}/{model_name}.gcode"

    if gcode_dialect == "unlayered":
        # Pull bead/thermal defaults from the Cura settings so the annotations
        # and flow match the slice; fall back to the example's PLA profile.
        def _setting(key, default):
            data = settings.get(key)
            if isinstance(data, dict):
                v = data.get("value")
                return v if v is not None else data.get("default_value", default)
            return data if data is not None else default

        start_gcode = None
        if start_gcode_path:
            with open(start_gcode_path) as sfh:
                start_gcode = sfh.read()
        end_gcode = None
        if end_gcode_path:
            with open(end_gcode_path) as efh:
                end_gcode = efh.read()

        write_unlayered_gcode(
            out_gcode_path,
            new_gcode_points,
            line_width=float(_setting("wall_line_width", _setting("line_width", 0.4))),
            layer_height=float(_setting("layer_height", 0.2)),
            nozzle_temp=int(_setting("material_print_temperature", 240)),
            bed_temp=int(_setting("material_bed_temperature", 55)),
            start_gcode=start_gcode,
            end_gcode=end_gcode,
        )
        print(f"Total time: {time.time() - total_t}")
        bench.finish()
        return

    prev_r = 0
    prev_theta = 0
    prev_z = 20
    theta_accum = 0

    with open(out_gcode_path, "w") as fh:
        print(f"Saving to output_gcode/{model_name}.gcode")
        fh.write("G94 ; mm/min feed  \n")
        fh.write("G28 ; home \n")
        fh.write("M83 ; relative extrusion \n")
        fh.write("G1 E10 ; prime extruder \n")
        fh.write("G94 ; mm/min feed \n")
        fh.write("G90 ; absolute positioning \n")
        fh.write(f"G0 C{prev_theta} X{prev_r} Z{prev_z} B0 ; go to start \n")
        fh.write("G93 ; inverse time feed \n")

        for i, point in enumerate(new_gcode_points):
            position = point["position"]
            rotation = point["rotation"]

            if np.all(np.isnan(position)):
                continue

            if position[2] < 0:
                continue

            z_hop = 0
            if point["travelling"]:
                z_hop = 1

            r = np.linalg.norm(position[:2])
            theta = np.arctan2(position[1], position[0])
            z = position[2]

            r += -np.sin(rotation) * (NOZZLE_OFFSET + z_hop)
            z += (np.cos(rotation) - 1) * (NOZZLE_OFFSET + z_hop) + z_hop

            delta_theta = theta - prev_theta
            if delta_theta > np.pi:
                delta_theta -= 2 * np.pi
            if delta_theta < -np.pi:
                delta_theta += 2 * np.pi

            theta_accum += delta_theta

            string = f"{point['command']} C{np.rad2deg(theta_accum):.5f} X{r:.5f} Z{z:.5f} B{np.rad2deg(rotation):.5f}"

            if point["extrusion"] is not None:
                string += f" E{point['extrusion']:.4f}"

            no_feed_value = False
            if point["inv_time_feed"] is not None:
                string += f" F{(point['inv_time_feed']):.4f}"
            else:
                string += f" F20000"
                fh.write(f"G94\n")
                no_feed_value = True

            fh.write(string + "\n")

            if no_feed_value:
                fh.write(f"G93\n")

            prev_r = r
            prev_theta = theta
            prev_z = z
    print(f"Total time: {time.time() - total_t}")
    bench.finish()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="S5.py",
        description=("Optimized Non-planar slicing"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────────────────────
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "model",
        metavar="MODEL.stl",
        help="Input surface mesh (STL, OBJ, PLY — anything open3d accepts).",
    )
    io.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT.gcode",
        default="output",
        help="Output G-code path.",
    )
    io.add_argument(
        "-c",
        "--config",
        metavar="CONFIG.json",
        default="config/core.def.json",
        help=("Custom Cura JSON settings."),
    )
    io.add_argument(
        "-u",
        "--cura",
        default="/usr/bin/CuraEngine",  # Debian `cura-engine` package; Windows: C:/Program Files/UltiMaker Cura 5.x/CuraEngine.exe
        help=("CuraEngine path."),
    )

    # On Debian the `cura` package ships engine-matched definitions under
    # /usr/share/cura/resources/definitions.  The bundled config/*.def.json are
    # exported from a newer Cura (5.11) and break the 5.0.0 engine packaged for
    # Debian (e.g. missing `wireframe_enabled` defaults), so default to the
    # system definitions and fall back to the bundled copies if absent.
    _CURA_DEFS = "/usr/share/cura/resources/definitions"

    def _def_path(name):
        sys_path = os.path.join(_CURA_DEFS, name)
        return sys_path if os.path.isfile(sys_path) else f"config/{name}"

    io.add_argument(
        "--extruder_path",
        default=_def_path("fdmextruder.def.json"),
        help=("Default extruder definitions for CuraEngine."),
    )

    io.add_argument(
        "--printer_path",
        default=_def_path("fdmprinter.def.json"),
        help=("Default printer definitions for CuraEngine."),
    )

    # ── Mesh placement ────────────────────────────────────────────────────────
    mesh_grp = parser.add_argument_group("mesh placement")
    mesh_grp.add_argument(
        "--reorient",
        metavar="X,Y,Z",
        default=None,
        help=(
            "Reorient the part before slicing by an intrinsic Euler rotation, "
            'X,Y,Z in degrees (e.g. "90,0,180"). Applied first, before scale, '
            "auto-centering and offset. Default: none."
        ),
    )
    mesh_grp.add_argument(
        "--scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help="Uniform scale factor applied to the mesh before slicing. Default: 1.0.",
    )
    mesh_grp.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help=(
            "Additional XYZ offset (mm) applied after auto-centering. "
            "Auto-centering places the bounding-box center at (0,0) and the "
            "bottom face at Z=0; use --offset to fine-tune placement. "
            "Default: 0 0 0."
        ),
    )

    # ── Rotation field ────────────────────────────────────────────────────────
    rot_grp = parser.add_argument_group("rotation field")
    rot_grp.add_argument(
        "--max-overhang",
        type=float,
        default=30.0,
        metavar="DEG",
        help=(
            "Maximum printable overhang angle from vertical (degrees). "
            "Faces whose normal exceeds 90+MAX_OVERHANG degrees from ẑ "
            "are treated as overhangs requiring tilt correction. Default: 30."
        ),
    )
    rot_grp.add_argument(
        "--rotation-multiplier",
        type=float,
        default=2.0,
        metavar="K",
        help=(
            "Scale factor applied to the raw overhang-angle magnitude before "
            "it enters the QP as r_init. Increases tilt aggressiveness. Default: 2."
        ),
    )
    rot_grp.add_argument(
        "--neighbor-loss-weight",
        type=float,
        default=30.0,
        help=(
            "Laplacian smoothing weight W for the rotation field: balances "
            "neighbour-agreement against the per-cell overhang target. Smoothing "
            "length scales ~sqrt(W), so use order-of-magnitude changes. Default: 30."
        ),
    )

    # ── Machine / G-code ─────────────────────────────────────────────────────
    mach_grp = parser.add_argument_group("machine / G-code")
    mach_grp.add_argument(
        "--nozzle-offset",
        type=float,
        default=41.5,
        metavar="MM",
        help=(
            "Distance (mm) from the B-axis pivot to the nozzle tip. "
            "Used to correct the radial and Z position when the nozzle tilts. "
            "Default: 42."
        ),
    )
    mach_grp.add_argument(
        "--rotation-smoothing",
        type=float,
        default=0.25,
        metavar="ALPHA",
        help=(
            "EMA alpha for B-axis smoothing between consecutive G-code points "
            "(0 = fully smoothed toward previous, 1 = no smoothing). Default: 0.25."
        ),
    )
    mach_grp.add_argument(
        "--gcode-dialect",
        choices=["rtheta", "unlayered"],
        default="rtheta",
        help=(
            "Output G-code dialect. 'rtheta' (default): polar Core-R-Theta "
            "(C/X/Z/B + inverse-time feed). 'unlayered': Unlayered-Slicer / RRF "
            "dialect — Cartesian G1 X Y Z B C E, relative extrusion, mm/min "
            "feedrates, and ;LAYER_CHANGE/;TYPE/;WIDTH annotations (RTCP machine)."
        ),
    )
    mach_grp.add_argument(
        "--start-gcode",
        metavar="FILE",
        default=None,
        help=(
            "Override the start G-code (machine-specific homing/probe/purge) for "
            "the unlayered dialect with the contents of FILE. Default: built-in."
        ),
    )
    mach_grp.add_argument(
        "--end-gcode",
        metavar="FILE",
        default=None,
        help="Override the end G-code for the unlayered dialect with FILE. Default: built-in.",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print extra diagnostic information during each pipeline stage.",
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not os.path.isfile(args.model):
        parser.error(f"Model file not found: {args.model}")
    if args.scale <= 0:
        parser.error("--scale must be positive.")
    if not (0.0 < args.rotation_smoothing <= 1.0):
        parser.error("--rotation-smoothing must be in (0, 1].")

    # ── Echo configuration ────────────────────────────────────────────────────
    if args.verbose:
        print("── S5 args──────────────────────")
        print(f"  model            : {args.model}")
        print(f"  output           : {args.output}")
        print(f"  config           : {args.config or '(none)'}")
        print(f"  reorient         : {args.reorient or '(none)'}")
        print(f"  scale            : {args.scale}")
        print(f"  offset           : {args.offset}")
        print(f"  max_overhang     : {args.max_overhang}°")
        print(f"  rotation_mult    : {args.rotation_multiplier}")
        print(f"  neighbor_loss_weight: {args.neighbor_loss_weight}")
        print(f"  nozzle_offset    : {args.nozzle_offset} mm")
        print(f"  rot_smoothing    : {args.rotation_smoothing}")
        print("─────────────────────────────────────────────────────────")

    # ── Run ───────────────────────────────────────────────────────────────────
    slice(
        config_path=args.config,
        cura_path=args.cura,
        model_path=args.model,
        extruder_path=args.extruder_path,
        printer_path=args.printer_path,
        output_path=args.output,
        scale=args.scale,
        offset=args.offset,
        max_overhang=args.max_overhang,
        rotation_multiplier=args.rotation_multiplier,
        neighbor_loss_weight=args.neighbor_loss_weight,
        nozzle_offset=args.nozzle_offset,
        reorient=args.reorient,
        gcode_dialect=args.gcode_dialect,
        start_gcode_path=args.start_gcode,
        end_gcode_path=args.end_gcode,
    )
