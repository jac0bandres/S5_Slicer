"""Tetrahedral scalar isosurfaces; no planar slicer dependency."""
from pathlib import Path
import numpy as np
from .mesh import unit


def isosurface(mesh, scalar, level, *, return_cell_ids=False):
    """Extract a level with upstream epsilon handling and shorter quad diagonal.

    Vertex/face order may differ from QMesh's linked lists. Geometry is compared
    independently of that order. Input scalar is mutated as in the C++ routine.
    """
    near = np.abs(scalar-level) < 1e-5
    scalar[near] = level + np.where(scalar[near] > level, 1e-5, -1e-5)
    values = scalar[mesh.cells]
    active = np.flatnonzero((values.min(axis=1) < level) & (values.max(axis=1) > level))
    edge_vertices, points, triangles, owners = {}, [], [], []
    gradients = mesh.gradient(scalar)
    for ci in active:
        cell = mesh.cells[ci]
        ids = []
        for i in range(4):
            for j in range(i+1,4):
                a,b = int(cell[i]),int(cell[j])
                if (scalar[a]-level)*(scalar[b]-level) >= 0:
                    continue
                key = tuple(sorted((a,b)))
                if key not in edge_vertices:
                    t = (level-scalar[a])/(scalar[b]-scalar[a])
                    edge_vertices[key] = len(points)
                    points.append((1-t)*mesh.points[a]+t*mesh.points[b])
                ids.append(edge_vertices[key])
        polygon = np.array([points[i] for i in ids])
        center = polygon.mean(axis=0)
        u = unit(polygon[0]-center)
        v = np.cross(unit(gradients[ci]),u)
        angles = np.arctan2((polygon-center)@v,(polygon-center)@u)
        ids = np.asarray(ids)[np.argsort(angles)].tolist()
        if len(ids) == 3:
            triangles.append(ids)
            owners.append(ci)
        elif len(ids) == 4:
            owners.extend([ci,ci])
            q = np.array([points[i] for i in ids])
            if np.linalg.norm(q[0]-q[2]) <= np.linalg.norm(q[1]-q[3]):
                triangles.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
            else:
                triangles.extend([[ids[0],ids[1],ids[3]],[ids[1],ids[2],ids[3]]])
        else:
            raise ValueError('Unexpected tetrahedral level intersection')
    result = (np.asarray(points).reshape(-1,3),np.asarray(triangles,dtype=int).reshape(-1,3))
    return (*result,np.asarray(owners,dtype=int)) if return_cell_ids else result


def write_layers(mesh, scalar, count, directory):
    if count < 1:
        raise ValueError('Layer count must be positive')
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    field = np.array(scalar,copy=True)
    lo,span=float(field.min()),float(np.ptp(field))
    if span<=0: raise ValueError('Scalar field must have a nonzero range')
    manifest = []
    for i in range(count):
        level = lo+(i+0.5)*span/count
        points, faces = isosurface(mesh,field,level)
        path = directory/f'{i:04d}.obj'
        with path.open('w') as f:
            f.write(f'# S3 scalar level {level:.17g}\n')
            np.savetxt(f,points,fmt='v %.17g %.17g %.17g')
            np.savetxt(f,faces+1,fmt='f %d %d %d')
        manifest.append(dict(file=path.name,level=level,vertices=len(points),faces=len(faces)))
    return manifest
