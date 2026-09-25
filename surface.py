"""Triangle-surface distances used to validate adaptive layer spacing."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def point_triangle_squared(point,triangles):
    """Exact closest-point squared distances, including segment/vertex regions."""
    t=np.asarray(triangles,dtype=float)
    a=t[:,0];ab=t[:,1]-a;ac=t[:,2]-a;ap=np.asarray(point)-a
    aa=np.einsum('ij,ij->i',ab,ab);cc=np.einsum('ij,ij->i',ac,ac)
    cross=np.einsum('ij,ij->i',ab,ac)
    pa=np.einsum('ij,ij->i',ap,ab);pc=np.einsum('ij,ij->i',ap,ac)
    determinant=aa*cc-cross*cross
    valid=determinant>1e-24
    u=np.divide(cc*pa-cross*pc,determinant,out=np.zeros(len(t)),where=valid)
    v=np.divide(aa*pc-cross*pa,determinant,out=np.zeros(len(t)),where=valid)
    inside=valid & (u>=0)&(v>=0)&(u+v<=1)
    projection=a+u[:,None]*ab+v[:,None]*ac
    best=np.where(inside,np.sum((point-projection)**2,axis=1),np.inf)
    for i,j in [(0,1),(1,2),(2,0)]:
        edge=t[:,j]-t[:,i]
        length2=np.sum(edge*edge,axis=1)
        fraction=np.divide(np.sum((point-t[:,i])*edge,axis=1),length2,
                           out=np.zeros(len(t)),where=length2>0)
        closest=t[:,i]+np.clip(fraction,0,1)[:,None]*edge
        best=np.minimum(best,np.sum((point-closest)**2,axis=1))
    return best


def triangle_triangle_squared(triangle,targets):
    """Triangle-pair minimum: vertex/face, edge/edge, and edge/face crossings."""
    source=np.broadcast_to(np.asarray(triangle,dtype=float),np.asarray(targets).shape)
    targets=np.asarray(targets,dtype=float)
    best=np.full(len(targets),np.inf)
    for i in range(3):
        best=np.minimum(best,point_triangle_squared(source[:,i],targets))
        best=np.minimum(best,point_triangle_squared(targets[:,i],source))
    for i in range(3):
        a=source[:,i];u=source[:,(i+1)%3]-a
        aa=np.einsum('ij,ij->i',u,u)
        for j in range(3):
            b=targets[:,j];v=targets[:,(j+1)%3]-b;r=a-b
            ee=np.einsum('ij,ij->i',v,v);uv=np.einsum('ij,ij->i',u,v)
            ur=np.einsum('ij,ij->i',u,r);vr=np.einsum('ij,ij->i',v,r)
            denominator=aa*ee-uv*uv
            s=np.clip(np.divide(uv*vr-ur*ee,denominator,out=np.zeros(len(best)),
                                where=denominator>1e-14*aa*ee),0,1)
            t=(uv*s+vr)/ee
            s=np.where(t<0,np.clip(-ur/aa,0,1),s)
            s=np.where(t>1,np.clip((uv-ur)/aa,0,1),s)
            t=np.clip(t,0,1)
            best=np.minimum(best,np.sum((r+s[:,None]*u-t[:,None]*v)**2,axis=1))
    # An edge can pierce a triangle interior without touching any triangle edge.
    for edges,faces in ((source,targets),(targets,source)):
        normal=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
        for i in range(3):
            a=edges[:,i];b=edges[:,(i+1)%3]
            d0=np.einsum('ij,ij->i',a-faces[:,0],normal)
            d1=np.einsum('ij,ij->i',b-faces[:,0],normal)
            valid=(d0*d1<=0)&(d0!=d1)
            t=np.divide(d0,d0-d1,out=np.zeros(len(best)),where=valid)
            p=a+t[:,None]*(b-a)
            # Barycentric coordinates detect an interior intersection directly.
            ab=faces[:,1]-faces[:,0];ac=faces[:,2]-faces[:,0];ap=p-faces[:,0]
            aa=np.einsum('ij,ij->i',ab,ab);cc=np.einsum('ij,ij->i',ac,ac)
            cross=np.einsum('ij,ij->i',ab,ac)
            pa=np.einsum('ij,ij->i',ap,ab);pc=np.einsum('ij,ij->i',ap,ac)
            determinant=aa*cc-cross*cross
            u=(cc*pa-cross*pc)/determinant;v=(aa*pc-cross*pa)/determinant
            best=np.where(valid&(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12),0.,best)
    return np.maximum(best,0.)


@dataclass
class Surface:
    points: np.ndarray
    faces: np.ndarray
    level: float=0.
    inserted: bool=False
    cell_ids: np.ndarray | None=None

    def __post_init__(self):
        self.points=np.asarray(self.points,dtype=float).reshape(-1,3)
        self.faces=np.asarray(self.faces,dtype=int).reshape(-1,3)
        if not len(self.faces):raise ValueError('Surface has no triangles')
        if self.cell_ids is not None:
            self.cell_ids=np.asarray(self.cell_ids,dtype=int)
            if self.cell_ids.shape!=(len(self.faces),):raise ValueError('Invalid surface cell provenance')
        if not np.isfinite(self.points).all():raise ValueError('Nonfinite surface')
        self.triangles=self.points[self.faces]
        self.centers=self.triangles.mean(axis=1)
        self.radii=np.linalg.norm(self.triangles-self.centers[:,None],axis=2).max(axis=1)
        cross=np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0])
        lengths=np.linalg.norm(cross,axis=1)
        if np.any(lengths<1e-14):raise ValueError('Degenerate surface triangle')
        self.normals=cross/lengths[:,None]
        self.tree=cKDTree(self.centers)

    def nearest(self,point):
        """Exact point-to-triangle-mesh query with conservative sphere pruning."""
        _,first=self.tree.query(point)
        best=float(point_triangle_squared(point,self.triangles[[first]])[0])
        candidates=self.tree.query_ball_point(point,np.sqrt(best)+self.radii.max()+1e-12)
        candidates=np.asarray(candidates,dtype=int)
        bounds=np.linalg.norm(self.centers[candidates]-point,axis=1)-self.radii[candidates]
        candidates=candidates[bounds<=np.sqrt(best)+1e-12]
        distances=point_triangle_squared(point,self.triangles[candidates])
        chosen=int(np.argmin(distances))
        return float(np.sqrt(max(0.,distances[chosen]))),int(candidates[chosen])

    def distances(self,points):
        return np.array([self.nearest(p)[0] for p in points])

    def triangle_minimum(self,triangle):
        center=triangle.mean(axis=0);radius=np.linalg.norm(triangle-center,axis=1).max()
        upper,_=self.nearest(center)
        candidates=np.asarray(self.tree.query_ball_point(center,upper+radius+self.radii.max()+1e-12),dtype=int)
        bounds=np.linalg.norm(self.centers[candidates]-center,axis=1)-self.radii[candidates]-radius
        candidates=candidates[bounds<=upper+1e-12]
        return float(np.sqrt(triangle_triangle_squared(triangle,self.triangles[candidates]).min()))


def triangle_distance_bounds(triangle,target):
    """Bounds for min/max distance from a whole triangle to a surface.

    Lower bound follows the 1-Lipschitz distance function. The upper bound
    uses convexity of distance to one fixed target triangle, so it applies to
    every point of the source triangle, not just to its vertices.
    """
    center=triangle.mean(axis=0)
    radius=np.linalg.norm(triangle-center,axis=1).max()
    center_distance,nearest=target.nearest(center)
    vertex_distances=target.distances(triangle)
    fixed=target.triangles[[nearest]]
    upper=max(np.sqrt(point_triangle_squared(p,fixed)[0]) for p in triangle)
    min_upper=min(center_distance,float(vertex_distances.min()))
    # A triangle lies in its plane and bounding sphere. Their distance lower
    # bounds combine, and taking the minimum over target triangles bounds the
    # union. This is especially useful for parallel planar layers.
    candidates=np.asarray(target.tree.query_ball_point(center,min_upper+radius+target.radii.max()+1e-12),dtype=int)
    normals=target.normals[candidates]
    signed=np.einsum('mij,mj->mi',triangle[None]-target.triangles[candidates,:1],normals)
    plane=np.where((signed.min(axis=1)>0)|(signed.max(axis=1)<0),np.abs(signed).min(axis=1),0.)
    sphere=np.linalg.norm(target.centers[candidates]-center,axis=1)-target.radii[candidates]-radius
    lower=max(0.,center_distance-radius,float(np.maximum(plane,sphere).min()))
    return (lower,min_upper,
            max(center_distance,float(vertex_distances.max())),float(upper))


def subdivide(triangle):
    a,b,c=triangle;ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
    return [np.array([a,ab,ca]),np.array([ab,b,bc]),np.array([ca,bc,c]),np.array([ab,bc,ca])]


def conform_surface(surface):
    """Split hanging edges left by adaptive trimming without changing geometry.

Every existing vertex in an edge interior becomes part of that face boundary.
Faces needing such splits are triangulated around their centroid, preserving
orientation and source-cell provenance. This avoids false toolpath boundaries.
"""
    points=surface.points.tolist();faces=[];owners=[]
    tree=cKDTree(surface.points)
    tolerance=1e-10*max(1.,float(np.ptp(surface.points,axis=0).max()))
    cache={}
    for fi,face in enumerate(surface.faces):
        polygon=[];split=False
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))))
            if key not in cache:
                p,q=surface.points[list(key)];edge=q-p;length=np.linalg.norm(edge)
                candidates=np.asarray(tree.query_ball_point((p+q)/2,length/2+tolerance),dtype=int)
                t=(surface.points[candidates]-p)@edge/(length*length)
                error=np.linalg.norm(surface.points[candidates]-(p+t[:,None]*edge),axis=1)
                mask=(error<tolerance)&(t>tolerance/length)&(t<1-tolerance/length)
                cache[key]=candidates[mask][np.argsort(t[mask])].tolist()
            middle=cache[key] if a<b else cache[key][::-1]
            polygon.extend([int(a),*middle]);split|=bool(middle)
        if not split:
            faces.append(face.tolist());owners.append(fi)
        else:
            center=len(points);points.append(surface.triangles[fi].mean(axis=0).tolist())
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                faces.append([a,b,center]);owners.append(fi)
    cells=None if surface.cell_ids is None else surface.cell_ids[owners]
    return Surface(points,faces,surface.level,surface.inserted,cells)


def separation_bounds(source,target,*,tolerance=.01,max_refinements=10000,
                      minimum_threshold=None,maximum_threshold=None):
    """Certified intervals for directed minimum/maximum surface distances.

    Adaptive subdivision tightens Lipschitz bounds. If the budget is exhausted,
    returned intervals remain valid but ``resolved`` is false. Callers must not
    turn an unresolved interval into a claim of satisfying thickness limits.
    Optional thresholds stop as soon as the requested spacing decisions are
    established by bounds. ``resolved`` still refers to interval precision;
    ``decision_resolved`` reports the separate threshold decision.
    """
    if tolerance<=0 or max_refinements<0:raise ValueError('Invalid bound parameters')
    cells=[(tri,triangle_distance_bounds(tri,target)) for tri in source.triangles]
    # Near d_min, generic Lipschitz bounds can require thousands of subdivisions.
    # Exact triangle-pair minima decide this part without refining the surface.
    if minimum_threshold is not None:
        exact=min(target.triangle_minimum(tri) for tri in source.triangles)
        margin=1e-12*max(1.,float(np.ptp(source.points,axis=0).max()),float(np.ptp(target.points,axis=0).max()))
        minimum_interval=(max(0.,exact-margin),exact+margin)
    refinements=0
    while True:
        bounds=np.array([b for _,b in cells])
        minimum_lower=float(bounds[:,0].min());minimum_upper=float(bounds[:,1].min())
        if minimum_threshold is not None:minimum_lower,minimum_upper=minimum_interval
        maximum_lower=float(bounds[:,2].max());maximum_upper=float(bounds[:,3].max())
        min_precise=minimum_upper-minimum_lower<=tolerance
        max_precise=maximum_upper-maximum_lower<=tolerance
        decisions=minimum_threshold is not None or maximum_threshold is not None
        min_decided=minimum_threshold is None or minimum_lower>=minimum_threshold or minimum_upper<minimum_threshold
        max_decided=maximum_threshold is None or maximum_upper<=maximum_threshold or maximum_lower>maximum_threshold
        min_done=(min_decided or min_precise) if decisions else min_precise
        max_done=max_decided if decisions else max_precise
        if (min_done and max_done) or refinements>=max_refinements:
            return dict(min_lower=minimum_lower,min_upper=minimum_upper,
                        max_lower=maximum_lower,max_upper=maximum_upper,
                        resolved=bool(min_precise and max_precise),
                        decision_resolved=bool(min_decided and max_decided) if decisions else None,
                        refinements=refinements)
        if not min_done:
            index=int(np.argmin(bounds[:,0]))
        else:index=int(np.argmax(bounds[:,3]))
        tri,_=cells.pop(index)
        cells.extend((child,triangle_distance_bounds(child,target)) for child in subdivide(tri))
        refinements+=1
