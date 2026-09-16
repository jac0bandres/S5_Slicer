"""Tetrahedralize an uploaded surface mesh (STL/OBJ/PLY) into a TetMesh.

The S³ pipeline operates on tetrahedral volumes, but users usually have surface
meshes. This converts an in-memory surface file to a :class:`TetMesh` with
TetGen, mirroring the established S5.py path (trimesh cleanup -> TetGen with
quality/epsilon fallbacks). TetGen can hard-crash (segfault) on self-
intersecting or degenerate CAD surfaces rather than raise, so it runs in a
forked child process: a crash kills only the child and surfaces as a normal
error instead of taking down the Streamlit server.
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


def _worker(points, faces, out_path):
    """Tetrahedralize in a child process; pickle node/elem (or the error)."""
    # TetGen and its dependencies log to native stdout/stderr; silence them so
    # only the pickled result crosses back to the parent.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    import tetgen

    tri = np.ascontiguousarray(faces, dtype=np.int32)
    errors = []
    for kwargs in _TIERS:
        try:
            tg = tetgen.TetGen(np.ascontiguousarray(points, float), tri)
            tg.tetrahedralize(**kwargs)
            if len(tg.elem):
                with open(out_path, 'wb') as fh:
                    pickle.dump({'ok': True, 'node': np.asarray(tg.node, float),
                                 'elem': np.asarray(tg.elem, np.int64)}, fh)
                return
            errors.append('empty tetrahedralization')
        except Exception as error:  # noqa: BLE001 - report every tier's failure
            errors.append(str(error).strip())
    with open(out_path, 'wb') as fh:
        pickle.dump({'ok': False, 'error': ' | '.join(errors)}, fh)


def tetrahedralize(payload, suffix='stl'):
    """Convert surface-file bytes to a :class:`TetMesh` via an isolated TetGen run."""
    points, faces = read_surface(payload, suffix)
    fd, out = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    os.remove(out)  # the child (re)creates it only on a clean result
    proc = mp.get_context('fork').Process(target=_worker, args=(points, faces, out))
    proc.start()
    proc.join()
    if not os.path.exists(out):
        raise RuntimeError(f'TetGen crashed (exit code {proc.exitcode}) — the surface '
                           'likely has self-intersections or degenerate facets. Repair '
                           'it in CAD (or with pymeshfix/MeshLab) and re-upload.')
    try:
        with open(out, 'rb') as fh:
            result = pickle.load(fh)
    finally:
        os.remove(out)
    if not result['ok']:
        raise RuntimeError(f'TetGen could not tetrahedralize the surface: {result["error"]}')
    return TetMesh(result['node'], result['elem'])
