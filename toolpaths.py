"""Surface toolpaths following S³ §5.1: boundary contours and stress interiors.

Distances use a surface FEM heat solve; stress is a sign-ambiguous tangent line
field fitted by a scalar potential. These are numerical approximations, not
certified bead coverage. Separate curves remain separate deposition strokes.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from .surface import Surface


@dataclass(frozen=True)
class ToolpathConfig:
    spacing: float=.4
    waypoint_distance: float=.4
    boundary_count: int=2
    mode: str='contour'
    max_faces: int=150000
    max_paths: int=10000

    def __post_init__(self):
        if not np.isfinite([self.spacing,self.waypoint_distance]).all() or min(self.spacing,self.waypoint_distance)<=0:
            raise ValueError('Path spacing and waypoint distance must be finite and positive')
        if self.mode not in ('contour','hybrid'):raise ValueError('Unknown toolpath mode')
        for name in ('boundary_count','max_faces','max_paths'):
            value=getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class Toolpath:
    points: np.ndarray
    normals: np.ndarray
    kind: str
    closed: bool
    level: float

    @property
    def length(self):
        return float(np.linalg.norm(np.diff(self.points,axis=0),axis=1).sum())


def topology(surface):
    raw=np.concatenate([surface.faces[:,[0,1]],surface.faces[:,[1,2]],surface.faces[:,[2,0]]])
    edges,inverse,counts=np.unique(np.sort(raw,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2):raise ValueError('Toolpath surface has nonmanifold edges')
    return edges,counts,inverse.reshape(3,-1).T


def refine_surface(surface,spacing,max_faces):
    """Conforming midpoint subdivision preserves the piecewise planar surface.

    Two initial passes introduce interior DOFs on coarse all-boundary triangles;
    subsequent passes split only long edges with compatible neighbor splits.
No smoothing or displacement of the original layer is performed.
"""
    current=surface
    passes=0
    while True:
        edges,_,face_edges=topology(current)
        lengths=np.linalg.norm(current.points[edges[:,0]]-current.points[edges[:,1]],axis=1)
        selected=(lengths>spacing) if passes>=2 else np.ones(len(edges),dtype=bool)
        if not selected.any():return current
        marked=selected[face_edges]
        if int((1+marked.sum(axis=1)).sum())>max_faces:
            raise ValueError(f'Layer refinement exceeds {max_faces} faces; increase path spacing or max_faces')
        passes+=1
        mid_ids=np.full(len(edges),-1,dtype=int)
        mid_ids[selected]=np.arange(selected.sum())+len(current.points)
        points=np.vstack([current.points,current.points[edges[selected]].mean(axis=1)])
        pieces=[];parents=[]
        for mask in range(8):
            pick=np.flatnonzero((marked@np.array([1,2,4]))==mask)
            if not len(pick):continue
            vertices=current.faces[pick];mids=mid_ids[face_edges[pick]]
            count=int(mask).bit_count()
            if count==0:children=vertices[:,None,:]
            elif count==3:
                a,b,c=vertices.T;ab,bc,ca=mids.T
                children=np.stack([np.column_stack(x) for x in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))],axis=1)
            else:
                # Rotate to split edge AB (one edge), or AB and BC (two).
                start=next(i for i in range(3) if ((mask>>i)&1) and (count==1 or ((mask>>((i+1)%3))&1)))
                a,b,c=np.roll(vertices,-start,axis=1).T
                ab,bc,ca=np.roll(mids,-start,axis=1).T
                triplets=((a,ab,c),(ab,b,c)) if count==1 else ((b,bc,ab),(a,ab,c),(ab,bc,c))
                children=np.stack([np.column_stack(x) for x in triplets],axis=1)
            pieces.append(children.reshape(-1,3));parents.extend(np.repeat(pick,count+1))
        faces=np.vstack(pieces)
        cells=None if current.cell_ids is None else current.cell_ids[parents]
        current=Surface(points,faces,current.level,current.inserted,cells)


class SurfaceFEM:
    def __init__(self,surface):
        self.surface=surface
        p=surface.triangles
        cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        twice_area=np.linalg.norm(cross,axis=1)
        self.area=twice_area/2
        self.basis=np.stack([np.cross(surface.normals,p[:,2]-p[:,1]),
                             np.cross(surface.normals,p[:,0]-p[:,2]),
                             np.cross(surface.normals,p[:,1]-p[:,0])],axis=1)/twice_area[:,None,None]
        n=len(surface.points)
        local=self.area[:,None,None]*(self.basis@self.basis.transpose(0,2,1))
        rows=np.broadcast_to(surface.faces[:,:,None],local.shape)
        cols=np.broadcast_to(surface.faces[:,None,:],local.shape)
        self.stiffness=sparse.coo_matrix((local.ravel(),(rows.ravel(),cols.ravel())),shape=(n,n)).tocsr()
        self.mass=np.zeros(n)
        np.add.at(self.mass,surface.faces.ravel(),np.repeat(self.area/3,3))
        edges,counts,self.face_edges=topology(surface)
        self.edges=edges
        self.boundary=np.unique(edges[counts==1])
        graph=sparse.coo_matrix((np.ones(2*len(edges)),
                                (edges.T.ravel(),edges[:,::-1].T.ravel())),shape=(n,n)).tocsr()
        self.components,self.labels=connected_components(graph)
        self.anchors=np.unique(self.labels,return_index=True)[1]
        if len(np.unique(self.labels[self.boundary]))!=self.components:
            raise ValueError('Each layer component needs a boundary for contour toolpaths')
        self.vertex_normals=np.zeros_like(surface.points)
        np.add.at(self.vertex_normals,surface.faces.ravel(),np.repeat(cross,3,axis=0))
        lengths=np.linalg.norm(self.vertex_normals,axis=1)
        if np.any(lengths<1e-14):raise ValueError('Inconsistent layer normals')
        self.vertex_normals/=lengths[:,None]

    def gradient(self,values):
        return np.einsum('fij,fi->fj',self.basis,np.asarray(values)[self.surface.faces])

    def fit(self,vectors,pins):
        rhs=np.zeros(len(self.mass))
        local=self.area[:,None]*np.einsum('fij,fj->fi',self.basis,vectors)
        np.add.at(rhs,self.surface.faces.ravel(),local.ravel())
        free=np.ones(len(rhs),dtype=bool);free[pins]=False
        result=np.zeros(len(rhs))
        if free.any():result[free]=spsolve(self.stiffness[free][:,free].tocsc(),rhs[free])
        if not np.isfinite(result).all():raise ValueError('Nonfinite surface potential')
        return result

    def boundary_distance(self):
        lengths=np.linalg.norm(self.surface.points[self.edges[:,0]]-self.surface.points[self.edges[:,1]],axis=1)
        time=lengths.mean()**2
        # Boundary heat is held at one; interior starts cold.
        heat=np.ones(len(self.mass));free=np.ones(len(heat),dtype=bool);free[self.boundary]=False
        system=sparse.diags(self.mass)+time*self.stiffness
        heat[free]=spsolve(system[free][:,free].tocsc(),-np.asarray(system[free][:,self.boundary].sum(axis=1)).ravel())
        gradient=-self.gradient(heat)
        norms=np.linalg.norm(gradient,axis=1)
        direction=np.divide(gradient,norms[:,None],out=np.zeros_like(gradient),where=norms[:,None]>1e-14)
        distance=self.fit(direction,self.boundary)
        # Discrete FEM can overshoot on poor elements; record before clipping.
        negative=int((distance < -1e-8).sum())
        return np.maximum(distance,0),negative

    def stress_potential(self,stress,critical,boundary_distance):
        surface=self.surface
        tangent=stress-surface.normals*np.einsum('ij,ij->i',stress,surface.normals)[:,None]
        lengths=np.linalg.norm(tangent,axis=1)
        valid=lengths>1e-10
        if np.any(critical & ~valid):
            raise ValueError('Principal stress is normal to a critical layer region; no tangent toolpath direction')
        tangent=np.divide(tangent,lengths[:,None],out=np.zeros_like(tangent),where=valid[:,None])
        vectors=np.cross(surface.normals,tangent)
        fallback=self.gradient(boundary_distance)
        norms=np.linalg.norm(fallback,axis=1)
        fallback=np.divide(fallback,norms[:,None],out=np.zeros_like(fallback),where=norms[:,None]>1e-12)
        vectors[~valid]=fallback[~valid]
        # Lift line-field signs over face adjacency. Residual below exposes
        # incompatibility left by cycles/singularities rather than hiding it.
        owners=[[] for _ in self.edges]
        for face,edges in enumerate(self.face_edges):
            for edge in edges:owners[edge].append(face)
        seen=np.zeros(len(surface.faces),dtype=bool)
        for root in range(len(seen)):
            if seen[root]:continue
            seen[root]=True;stack=[root]
            while stack:
                face=stack.pop()
                for edge in self.face_edges[face]:
                    for other in owners[edge]:
                        if seen[other]:continue
                        if np.dot(vectors[face],vectors[other])<0:vectors[other]*=-1
                        seen[other]=True;stack.append(other)
        field=self.fit(vectors,self.anchors)
        gradient=self.gradient(field)
        predicted=np.cross(surface.normals,gradient)
        norms=np.linalg.norm(predicted,axis=1)
        cosine=np.divide(np.abs(np.einsum('ij,ij->i',predicted,tangent)),norms,
                         out=np.zeros(len(norms)),where=norms>1e-12)
        angles=np.rad2deg(np.arccos(np.clip(cosine,0,1)))
        report=dict(stress_angle_mean_degrees=float(np.average(angles[critical],weights=self.area[critical])),
                    stress_angle_max_degrees=float(angles[critical].max()),
                    stress_fit_rms=float(np.sqrt(np.average(np.sum((gradient-vectors)**2,axis=1),weights=self.area))))
        return field,report


def _resample(points,normals,step):
    """Keep every triangle crossing, so chords never shortcut a curved layer."""
    output=[];directions=[]
    for a,b,na,nb in zip(points[:-1],points[1:],normals[:-1],normals[1:]):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/step)))
        t=np.arange(count)/count
        output.extend(a[None]+t[:,None]*(b-a))
        directions.extend(na[None]+t[:,None]*(nb-na))
    output.append(points[-1]);directions.append(normals[-1])
    directions=np.asarray(directions)
    lengths=np.linalg.norm(directions,axis=1)
    if np.any(lengths<1e-12):raise ValueError('Opposing surface normals on a toolpath segment')
    directions/=lengths[:,None]
    return np.asarray(output),directions


def isolines(fem,field,level,kind,step,*,clip_field=None,clip_min=0.):
    """March triangles and join only actual shared edges, preserving holes/islands."""
    surface=fem.surface
    values=np.asarray(field,dtype=float).copy()
    epsilon=1e-10*max(1.,float(np.ptp(values)))
    near=np.abs(values-level)<epsilon
    values[near]=level+epsilon
    tri_values=values[surface.faces]
    active=np.flatnonzero((tri_values.min(axis=1)<level)&(tri_values.max(axis=1)>level))
    nodes={};positions=[];normals=[];segments=[]
    def node(key,p,n):
        if key not in nodes:
            nodes[key]=len(positions);positions.append(p);normals.append(n/np.linalg.norm(n))
        return nodes[key]
    for face in active:
        hits=[]
        ids=surface.faces[face]
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=sorted((int(ids[i]),int(ids[j])))
            if (values[a]-level)*(values[b]-level)>=0:continue
            t=(level-values[a])/(values[b]-values[a])
            p=(1-t)*surface.points[a]+t*surface.points[b]
            n=(1-t)*fem.vertex_normals[a]+t*fem.vertex_normals[b]
            clip=0. if clip_field is None else (1-t)*clip_field[a]+t*clip_field[b]
            hits.append(((a,b),p,n,clip))
        if len(hits)!=2:continue
        if clip_field is not None:
            keep=[h[3]>=clip_min for h in hits]
            if not any(keep):continue
            if not all(keep):
                bad=0 if not keep[0] else 1;good=1-bad
                a,b=hits[bad],hits[good]
                t=(clip_min-a[3])/(b[3]-a[3])
                hits[bad]=(('clip',int(face)),(1-t)*a[1]+t*b[1],(1-t)*a[2]+t*b[2],clip_min)
        if np.linalg.norm(hits[0][1]-hits[1][1])<1e-12:continue
        segments.append(tuple(node(*h[:3]) for h in hits))
    adjacency=[[] for _ in positions]
    for edge,(a,b) in enumerate(segments):adjacency[a].append(edge);adjacency[b].append(edge)
    used=np.zeros(len(segments),dtype=bool);paths=[]
    def walk(start,edge):
        chain=[start];current=start
        while not used[edge]:
            used[edge]=True
            a,b=segments[edge];current=b if current==a else a
            chain.append(current)
            if current==start or len(adjacency[current])!=2:break
            choices=[e for e in adjacency[current] if not used[e]]
            if not choices:break
            edge=choices[0]
        pts=np.asarray(positions)[chain];ns=np.asarray(normals)[chain]
        pts,ns=_resample(pts,ns,step)
        paths.append(Toolpath(pts,ns,kind,chain[0]==chain[-1],float(level)))
    for start,edges in enumerate(adjacency):
        if len(edges)!=2:
            for edge in edges:
                if not used[edge]:walk(start,edge)
    for edge,(start,_) in enumerate(segments):
        if not used[edge]:walk(start,edge)
    return paths


def generate_toolpaths(layer,config=None,*,stress=None,stress_mask=None):
    """Generate sampled deposition curves in original material coordinates.

stress and stress_mask are indexed by tetrahedron, using layer.cell_ids for
exact transfer. A hybrid layer must intersect at least one critical cell;
other layers automatically use boundary contours throughout.
"""
    cfg=config or ToolpathConfig()
    refined=refine_surface(layer,cfg.spacing,cfg.max_faces)
    fem=SurfaceFEM(refined)
    distance,negative=fem.boundary_distance()
    hybrid=False;alignment={}
    if cfg.mode=='hybrid':
        if stress is None or stress_mask is None or refined.cell_ids is None:
            raise ValueError('Hybrid paths require per-tetrahedron stress, stress_mask, and layer provenance')
        stress=np.asarray(stress,dtype=float);stress_mask=np.asarray(stress_mask,dtype=bool)
        if stress.ndim!=2 or stress.shape[1]!=3 or stress_mask.shape!=(len(stress),) or not np.isfinite(stress).all():
            raise ValueError('Invalid toolpath stress arrays')
        if refined.cell_ids.min()<0 or refined.cell_ids.max()>=len(stress):raise ValueError('Stress does not cover layer cells')
        critical=stress_mask[refined.cell_ids]
        hybrid=bool(critical.any())
    paths=[]
    if float(distance.max())/cfg.spacing>cfg.max_paths:raise ValueError('Too many contour levels; increase path spacing')
    levels=np.arange(cfg.spacing/2,float(distance.max()),cfg.spacing)
    if hybrid:levels=levels[:cfg.boundary_count]
    for level in levels:
        paths.extend(isolines(fem,distance,level,'contour',cfg.waypoint_distance))
        if len(paths)>cfg.max_paths:raise ValueError('Toolpath count exceeds max_paths')
    if hybrid and distance.max()>cfg.boundary_count*cfg.spacing:
        field,alignment=fem.stress_potential(stress[refined.cell_ids],critical,distance)
        # A separate gauge per connected component must not enlarge its range.
        for label in range(fem.components):
            vertices=fem.labels==label
            field[vertices]-=field[vertices].min()
        if float(field.max())/cfg.spacing>cfg.max_paths:raise ValueError('Too many stress levels; increase path spacing')
        levels=np.arange(cfg.spacing/2,float(field.max()),cfg.spacing)
        for level in levels:
            paths.extend(isolines(fem,field,level,'stress',cfg.waypoint_distance,
                                  clip_field=distance,clip_min=cfg.boundary_count*cfg.spacing))
            if len(paths)>cfg.max_paths:raise ValueError('Toolpath count exceeds max_paths')
    report=dict(mode='hybrid' if hybrid else 'contour',config=asdict(cfg),
                curves=len(paths),closed_curves=sum(p.closed for p in paths),
                waypoints=sum(len(p.points) for p in paths),length=sum(p.length for p in paths),
                refined_faces=len(refined.faces),components=fem.components,
                boundary_distance_max=float(distance.max()),negative_distance_vertices=negative,
                components_below_first_offset=sum(float(distance[fem.labels==label].max())<=cfg.spacing/2
                                                  for label in range(fem.components)),
                coverage_certified=False,**alignment)
    return paths,report
