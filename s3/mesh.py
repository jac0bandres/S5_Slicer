"""Array representation of QMeshPatch's tetrahedra (official S³ ordering)."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

# QMeshPatch::inputTETFile, not an arbitrary marching-tet face ordering.
FACE_ORDER = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 0], [3, 1, 0]])


def unit(v):
    v = np.asarray(v, dtype=float)
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(lengths < 1e-14) or not np.isfinite(lengths).all():
        raise ValueError("Cannot normalize a degenerate/nonfinite vector")
    return v / lengths


@dataclass
class TetMesh:
    points: np.ndarray
    cells: np.ndarray

    def __post_init__(self):
        self.points = np.array(self.points, dtype=float, copy=True)
        self.cells = np.array(self.cells, dtype=np.int64, copy=True)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Expected points with shape (n, 3)")
        if self.cells.ndim != 2 or self.cells.shape[1] != 4 or not len(self.cells):
            raise ValueError("Expected nonempty cells with shape (m, 4)")
        if not np.isfinite(self.points).all():
            raise ValueError("Nonfinite vertices")
        if self.cells.min() < 0 or self.cells.max() >= len(self.points):
            raise ValueError("Invalid vertex index")
        if len(np.unique(self.cells)) != len(self.points):
            raise ValueError("Unreferenced vertices are not supported")
        verts = self.points[self.cells]
        edges = verts[:, 1:] - verts[:, :1]
        self.signed_volumes = np.linalg.det(edges) / 6
        if np.any(np.abs(self.signed_volumes) < 1e-14):
            raise ValueError("Degenerate tetrahedron")
        self.volumes = np.abs(self.signed_volumes)
        # Gradient basis: grad(phi_i), indexed [cell, local vertex, xyz].
        affine = np.concatenate([np.ones((*verts.shape[:2], 1)), verts], axis=2)
        self.grad_basis = np.linalg.inv(affine)[:, 1:, :].transpose(0, 2, 1)
        faces, owners, lookup = [], [], {}
        self.cell_faces = np.empty((len(self.cells), 4), dtype=int)
        for ci, cell in enumerate(self.cells):
            for fi, ids in enumerate(FACE_ORDER):
                tri = cell[ids]
                key = tuple(sorted(tri))
                if key not in lookup:
                    lookup[key] = len(faces)
                    faces.append(tri)
                    owners.append([ci, -1])
                else:
                    if owners[lookup[key]][1] != -1:
                        raise ValueError("Nonmanifold face shared by more than two tetrahedra")
                    owners[lookup[key]][1] = ci
                self.cell_faces[ci, fi] = lookup[key]
        self.faces = np.asarray(faces)
        self.face_cells = np.asarray(owners)
        self.boundary = self.face_cells[:, 1] == -1
        self.neighbor_pairs = self.face_cells[~self.boundary]
        self.neighbors = [[] for _ in self.cells]
        for ci, fs in enumerate(self.cell_faces):
            for fi in fs:
                a, b = self.face_cells[fi]
                if b >= 0:
                    self.neighbors[ci].append(int(b if a == ci else a))
        # Identify gauges independently for disconnected vertex components.
        a = np.repeat(self.cells[:, 0], 3)
        b = self.cells[:, 1:].ravel()
        graph = sparse.coo_matrix((np.ones(len(a)*2),
                                  (np.r_[a,b], np.r_[b,a])),
                                 shape=(len(self.points), len(self.points))).tocsr()
        _, labels = connected_components(graph)
        self.vertex_components=labels
        self.anchors = np.unique(labels, return_index=True)[1]

    def normals(self, points=None):
        """Raw face winding normals, as stored by QMeshPatch (typically inward)."""
        v = (self.points if points is None else points)[self.faces]
        return unit(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]))

    def gradient(self, scalar):
        return np.einsum('cij,ci->cj', self.grad_basis, np.asarray(scalar)[self.cells])

    def gradient_operator(self):
        m, n = len(self.cells), len(self.points)
        rows = np.broadcast_to(np.arange(3*m).reshape(m, 1, 3), (m, 4, 3))
        cols = np.broadcast_to(self.cells[:, :, None], (m, 4, 3))
        return sparse.coo_matrix((self.grad_basis.ravel(), (rows.ravel(), cols.ravel())),
                                 shape=(3*m, n)).tocsr()


def read_tet(path, *, cpp_float32=False):
    """Read official zero-indexed .tet data; optionally emulate C++ float import."""
    with Path(path).open() as f:
        nv, label = f.readline().split()
        if label != 'vertices':
            raise ValueError("Missing vertices header")
        nt, label = f.readline().split()
        if label != 'tets':
            raise ValueError("Missing tets header")
        points = np.array([list(map(float, f.readline().split())) for _ in range(int(nv))],
                          dtype=np.float32 if cpp_float32 else float).astype(float)
        records = np.array([list(map(int, f.readline().split())) for _ in range(int(nt))])
        if records.shape != (int(nt), 5) or np.any(records[:, 0] != 4):
            raise ValueError("Invalid tetrahedron record")
    return TetMesh(points, records[:, 1:])


def write_tet(path, points, cells):
    with Path(path).open('w') as f:
        f.write(f'{len(points)} vertices\n{len(cells)} tets\n')
        np.savetxt(f, points, fmt='%.17g')
        np.savetxt(f, np.column_stack([np.full(len(cells), 4), cells]), fmt='%d')
