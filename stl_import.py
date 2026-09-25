"""Tetrahedralize an uploaded surface mesh (STL/OBJ/PLY) into a TetMesh.

The S³ pipeline operates on tetrahedral volumes, but users usually have surface
meshes. This converts an in-memory surface file to a :class:`TetMesh`, mirroring
the tiered, geometry-preserving fallback S5.py uses:

  1. TetGen on the cleaned surface — fast and quality-controlled, but brittle:
     self-intersections and near-degenerate facets make it fail or hard-crash.
  2. TetGen on an automatically repaired surface — trimesh topology cleanup plus
     hole filling, then pymeshfix (Attene's MeshFix) to remove
     self-intersections and return a watertight manifold.
  3. FloatTetWild — envelope-based and robust to broken input by design (its own
     winding-number inside/outside test). Slower, approximate boundary.

No tier decimates: full surface resolution is preserved throughout. Both
backends can hard-crash (segfault) rather than raise, so each attempt runs in a
forked child process: a crash kills only the child and surfaces as a normal
error instead of taking down the Streamlit server.

The repair tiers need optional extras (`requirements-repair.txt`); without them
the import simply falls back to tier 1 and reports what was missing.
"""
import io
import os
import pickle
import tempfile
import multiprocessing as mp

import numpy as np

from .mesh import TetMesh

SUFFIXES = ('stl', 'obj', 'ply', 'off')

# Quality knobs match S5.py's TetGen call (order=1 linear tets, 20° minimum
# dihedral, radius-edge ratio 1.5). Each later tier loosens constraints so
# borderline surfaces still tetrahedralize without decimating geometry.
_TIERS = (
    dict(order=1, mindihedral=20, minratio=1.5),
    dict(order=1, mindihedral=20, minratio=1.5, epsilon=1e-4),
    dict(order=1, epsilon=1e-4),
)

# FloatTetWild: epsilon = surface envelope and edge_length_r = target edge, both
# relative to the bounding-box diagonal. 0.05 preserves enough surface detail
# for the deformation field at a usable tet count.
_FTETWILD = dict(epsilon=1e-3, edge_length_r=0.05)


def read_surface(payload, suffix):
    """Load surface vertices/faces from file bytes, with cheap topology cleanup."""
    import trimesh

    kind = suffix.lower().lstrip('.')
    mesh = trimesh.load(io.BytesIO(payload), file_type=kind, force='mesh')
    if getattr(mesh, 'is_empty', True) or mesh.faces is None or not len(mesh.faces):
        raise ValueError('The uploaded file has no triangular surface to slice.')
    # Weld coincident vertices and drop degenerate/duplicate faces so TetGen's
    # boundary recovery has a clean PLC to start from.
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError('The surface collapsed to nothing after cleanup.')
    return np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)


def repair_surface(points, faces, notes=None):
    """Return a watertight, manifold, self-intersection-free copy of a surface.

    Two stages, as in S5.py: trimesh closes holes and makes the winding
    consistent, then pymeshfix removes self-intersections. Every connected
    component is kept (`remove_smallest_components=False`) so multi-part models
    are not silently gutted, and the closest ones are joined into a single PLC.
    Missing pymeshfix degrades to the trimesh-only result rather than failing.
    """
    import trimesh

    record = [] if notes is None else notes
    mesh = trimesh.Trimesh(np.asarray(points, float), np.asarray(faces, np.int64),
                           process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError('The surface collapsed to nothing during repair.')
    before = int(len(mesh.faces))
    if not mesh.is_watertight:
        # Best effort: trimesh's hole filling and winding fix need networkx, and
        # MeshFix below closes holes anyway, so a missing extra is not fatal.
        try:
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            record.append(f'Hole filling: {len(mesh.faces)-before} triangles added; '
                          f'watertight={mesh.is_watertight}.')
        except ImportError as error:
            record.append(f'trimesh hole filling unavailable ({error}); '
                          'leaving the holes to MeshFix.')
    v = np.asarray(mesh.vertices, float)
    f = np.asarray(mesh.faces, np.int64)
    try:
        import pymeshfix
    except ImportError:
        record.append('pymeshfix is not installed; skipped self-intersection repair '
                      '(pip install -r requirements-repair.txt).')
        return v, f
    fix = pymeshfix.MeshFix(v, f)
    fix.repair(joincomp=True, remove_smallest_components=False)
    v2 = np.asarray(fix.points, float)
    f2 = np.asarray(fix.faces, np.int64)
    if not len(f2):
        raise ValueError('MeshFix removed the entire surface.')
    record.append(f'MeshFix repair: {len(faces)} → {len(f2)} triangles.')
    return v2, f2


def _worker(backend, points, faces, out_path):
    """Tetrahedralize in a child process; pickle node/elem (or the error)."""
    # The backends and their dependencies log to native stdout/stderr; silence
    # them so only the pickled result crosses back to the parent.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    errors = []
    try:
        if backend == 'tetgen':
            import tetgen

            tri = np.ascontiguousarray(faces, dtype=np.int32)
            for kwargs in _TIERS:
                try:
                    tg = tetgen.TetGen(np.ascontiguousarray(points, float), tri)
                    tg.tetrahedralize(**kwargs)
                    if len(tg.elem):
                        node = np.asarray(tg.node, float)
                        elem = np.asarray(tg.elem, np.int64)
                        break
                    errors.append('empty tetrahedralization')
                except Exception as error:  # noqa: BLE001 - report every tier's failure
                    errors.append(str(error).strip())
            else:
                raise RuntimeError(' | '.join(errors))
        else:  # 'ftetwild'
            import wildmeshing as wm

            tet = wm.Tetrahedralizer(**_FTETWILD)
            tet.set_mesh(np.ascontiguousarray(points, np.float64),
                         np.ascontiguousarray(faces, np.int32))
            tet.tetrahedralize()
            # Winding-number inside/outside (floodfill=False): flood fill seeds
            # from outside and leaks through holes on high-genus models.
            node, elem, _ = tet.get_tet_mesh(floodfill=False)
            node = np.asarray(node, float)
            elem = np.asarray(elem, np.int64)
            if not len(elem):
                raise RuntimeError('FloatTetWild produced an empty mesh')
        with open(out_path, 'wb') as fh:
            pickle.dump({'ok': True, 'node': node, 'elem': elem}, fh)
    except Exception as error:  # noqa: BLE001 - the parent decides the next tier
        with open(out_path, 'wb') as fh:
            pickle.dump({'ok': False, 'error': str(error).strip() or repr(error)}, fh)


def _run(backend, points, faces):
    """Run one backend in an isolated child; return (node, elem) or raise."""
    fd, out = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    os.remove(out)  # the child (re)creates it only on a clean result
    proc = mp.get_context('fork').Process(target=_worker,
                                          args=(backend, points, faces, out))
    proc.start()
    proc.join()
    if not os.path.exists(out):
        raise RuntimeError(f'{backend} crashed (exit code {proc.exitcode}) — '
                           'unrecoverable self-intersection or degenerate facet')
    try:
        with open(out, 'rb') as fh:
            result = pickle.load(fh)
    finally:
        os.remove(out)
    if not result['ok']:
        raise RuntimeError(result['error'])
    node, elem = result['node'], result['elem']
    # Reject a degenerate/disconnected result (a few stray tets sharing no
    # vertices — a "soup"). A conforming tet mesh shares vertices heavily
    # (points ≈ cells/5); a soup has points ≈ 4·cells.
    if len(elem) < 4 or len(node) > 3 * len(elem):
        raise RuntimeError(f'{backend} returned a degenerate mesh '
                           f'({len(elem)} cells, {len(node)} points)')
    # TetMesh rejects unreferenced vertices; FloatTetWild can leave some behind.
    used, inverse = np.unique(elem, return_inverse=True)
    return node[used], inverse.reshape(elem.shape).astype(np.int64)


def _covers(node, points, tol=.02):
    """True when a tet mesh still spans the input surface's bounding box.

    Repair can silently delete geometry: MeshFix resolves a self-intersecting
    union (two overlapping parts) by dropping one of them. Comparing bounds
    against the input catches that, so the caller can fall through to a tier
    that keeps the whole model instead of slicing a mutilated one.
    """
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    slack = tol * float(np.linalg.norm(high - low))
    return bool(np.all(np.min(node, axis=0) <= low + slack)
                and np.all(np.max(node, axis=0) >= high - slack))


def tetrahedralize(payload, suffix='stl', repair=True, notes=None):
    """Convert surface-file bytes to a :class:`TetMesh`, repairing if needed.

    `repair=False` restricts the run to TetGen on the as-loaded surface — the
    behaviour before automatic repair existed. `notes`, if given, is a list that
    collects human-readable progress lines (which tier ran, what repair did, why
    a tier failed) for the caller to display.
    """
    record = [] if notes is None else notes
    points, faces = read_surface(payload, suffix)
    fixed = {}

    def repaired():
        if 'm' not in fixed:
            fixed['m'] = repair_surface(points, faces, record)
        return fixed['m']

    attempts = [('TetGen (as-is)', 'tetgen', lambda: (points, faces))]
    if repair:
        attempts += [('TetGen (repaired)', 'tetgen', repaired),
                     ('FloatTetWild (robust)', 'ftetwild', lambda: (points, faces))]
    for label, backend, geometry in attempts:
        try:
            node, elem = _run(backend, *geometry())
            if not _covers(node, points):
                raise ValueError('the result no longer spans the input bounding '
                                 'box — part of the model was dropped')
            mesh = TetMesh(node, elem)
        except (RuntimeError, ValueError, ImportError, MemoryError) as error:
            record.append(f'{label} failed: {str(error).strip()}')
            continue
        record.append(f'Volume mesh from {label}: {len(mesh.cells)} tetrahedra.')
        return mesh
    raise RuntimeError('Could not tetrahedralize the surface'
                       + (', even after automatic repair' if repair else '')
                       + '. It is likely empty, zero-volume or broken beyond '
                       'recovery — inspect it in CAD before slicing. Details: '
                       + ' | '.join(record))
