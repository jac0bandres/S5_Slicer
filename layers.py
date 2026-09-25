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


def first_bed_layer(mesh, scalar, height):
    """Choose an isosurface whose entire geometry lies in 0 <= Z <= height.

    For a linear tetrahedral field, the minimum field value on the portion
    Z >= height occurs at a vertex or an edge crossing that plane. Staying
    below that value prevents any first-layer component above the height cap.
    Validate the extracted triangles too, including isosurface perturbations.
    """
    from .surface import Surface
    field=np.asarray(scalar,dtype=float)
    if not np.isfinite(height) or height<=0:
        raise ValueError('First-layer height must be finite and positive')
    if field.shape!=(len(mesh.points),) or not np.isfinite(field).all() or np.ptp(field)<=0:
        raise ValueError('Invalid layer scalar field')
    z=mesh.points[:,2]
    if z.min()<0 or z.min()>1e-9:
        raise ValueError('First layer requires the model on the bed at Z=0; enable Drop model to bed')
    edges=np.unique(np.sort(np.concatenate([mesh.cells[:,[i,j]] for i in range(4)
                                            for j in range(i+1,4)]),axis=1),axis=0)
    a,b=edges.T
    crossing=(z[a]<height)&(z[b]>=height) | (z[b]<height)&(z[a]>=height)
    a,b=a[crossing],b[crossing]
    values=field[a]+(height-z[a])/(z[b]-z[a])*(field[b]-field[a])
    candidates=np.r_[field[z>=height],values]
    lo=float(field.min())
    cap=float(candidates.min()) if len(candidates) else float(field.max())
    if cap<=lo:
        raise ValueError('Scalar field starts above the first-layer height limit; rerun with a fixed build plate')
    level=cap if cap<float(field.max()) else lo+(cap-lo)*.5
    for _ in range(40):
        points,faces,cells=isosurface(mesh,field.copy(),level,return_cell_ids=True)
        if len(faces) and points[:,2].min()>=0 and points[:,2].max()<=height+1e-10:
            layer=Surface(points,faces,level,cell_ids=cells)
            return layer,dict(height_limit=float(height),minimum_z=float(points[:,2].min()),
                              maximum_z=float(points[:,2].max()),bed_spacing_passed=True)
        level=lo+(level-lo)*.5
    raise ValueError('Cannot extract a first layer within the physical bed-height limit')


def fixed_layers(mesh,scalar,count,first_layer_height=.2):
    """Fixed-count diagnostic layers with a physically bounded first surface."""
    if not isinstance(count,int) or count<1:raise ValueError('Layer count must be positive')
    from .surface import Surface
    first,bed_report=first_bed_layer(mesh,scalar,first_layer_height)
    scalar=np.asarray(scalar)
    last=float(scalar.min()+(count-.5)*np.ptp(scalar)/count)
    levels=np.linspace(first.level,max(first.level,last),count)
    layers=[first]
    for level in levels[1:]:
        if level<=layers[-1].level:continue
        points,faces,cells=isosurface(mesh,scalar.copy(),float(level),return_cell_ids=True)
        if len(faces):layers.append(Surface(points,faces,float(level),cell_ids=cells))
    return layers,bed_report


def write_layers(mesh, scalar, count, directory, *, first_layer_height=.2):
    if count < 1:
        raise ValueError('Layer count must be positive')
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if first_layer_height is None:
        from .surface import Surface
        layers=[]
        for level in np.min(scalar)+(np.arange(count)+.5)*np.ptp(scalar)/count:
            points,faces=isosurface(mesh,np.array(scalar,copy=True),level)
            if len(faces):layers.append(Surface(points,faces,level))
    else:
        layers,_=fixed_layers(mesh,scalar,count,first_layer_height)
    manifest = []
    for i,layer in enumerate(layers):
        level,points,faces=layer.level,layer.points,layer.faces
        path = directory/f'{i:04d}.obj'
        with path.open('w') as f:
            f.write(f'# S3 scalar level {level:.17g}\n')
            np.savetxt(f,points,fmt='v %.17g %.17g %.17g')
            np.savetxt(f,faces+1,fmt='f %d %d %d')
        manifest.append(dict(file=path.name,level=level,vertices=len(points),faces=len(faces)))
    return manifest
