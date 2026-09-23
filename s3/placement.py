"""Uniform model scaling and rigid placement before S3 deformation."""
from dataclasses import dataclass
import numpy as np
from .mesh import TetMesh


@dataclass(frozen=True)
class Placement:
    scale: float = 1.
    rotation_x: float = 0.
    rotation_y: float = 0.
    rotation_z: float = 0.
    center_xy: bool = False
    drop_to_bed: bool = True
    offset_x: float = 0.
    offset_y: float = 0.

    def __post_init__(self):
        if not np.isfinite([self.scale, self.rotation_x, self.rotation_y,
                            self.rotation_z, self.offset_x, self.offset_y]).all() or self.scale <= 0:
            raise ValueError('Placement values must be finite and scale must be positive')

    @property
    def rotation(self):
        """Right-handed rotations about world X, then Y, then Z, in degrees."""
        x, y, z = np.deg2rad([self.rotation_x, self.rotation_y, self.rotation_z])
        cx, cy, cz = np.cos([x, y, z])
        sx, sy, sz = np.sin([x, y, z])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return rz @ ry @ rx

    def apply(self, mesh):
        """Scale/rotate about bounding-box center, then center/drop/offset.

        Returns a new mesh. Cell and face indexing is retained, so objective
        selections remain valid. Stress directions must use rotate_vectors.
        """
        center = (mesh.points.min(axis=0) + mesh.points.max(axis=0)) / 2
        points = ((mesh.points-center)*self.scale) @ self.rotation.T + center
        if self.center_xy:
            points[:, :2] -= (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        if self.drop_to_bed:
            points[:, 2] -= points[:, 2].min()
        points[:, :2] += [self.offset_x, self.offset_y]
        return TetMesh(points, mesh.cells.copy())

    def rotate_vectors(self, vectors):
        """Rotate directions without translation or scaling their magnitudes."""
        return np.asarray(vectors) @ self.rotation.T
