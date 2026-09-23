"""Paper-defined deformation and height-field pipeline, independent of C++ quirks."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components,dijkstra
from .mesh import TetMesh, unit
from .numerics import from_two_vectors_batch
from .paper import project_printing_direction, blend_quaternions, solve_paper_scales, QuaternionBlendWorkspace


@dataclass(frozen=True)
class PaperConfig:
    alpha: float = 30.
    beta: float = 10.
    gamma: float = 10.
    weight_sf: float = 1.
    weight_sr: float = 0.
    weight_sq: float = 0.
    rigidity: float = 1.
    scale_compatibility: float = 6.
    inner_iterations: int = 7
    max_outer_iterations: int = 20
    relative_tolerance: float = 0.05
    objective_tolerance: float = 1e-8
    final_conformal_weight: float = 10.
    concavity_control: bool = True
    nozzle_cone_angle: float = 60.
    concavity_exponent: float = 1.
    exclude_build_plate: bool = True
    fix_build_plate: bool = True
    base_band: float = 0.
    enforce_build_order: bool = True
    scale_solver: str = 'auto'
    scale_tolerance: float = 1e-9
    scale_max_iterations: int = 5000

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name=='scale_solver':continue
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite parameter {name}')
        for name in ['alpha','beta','gamma']:
            if not 0<=getattr(self,name)<90: raise ValueError(f'Invalid {name}')
        for name in ['weight_sf','weight_sr','weight_sq']:
            if not 0<=getattr(self,name)<=1: raise ValueError(f'Invalid {name}')
        if self.weight_sf+self.weight_sr+self.weight_sq<=0:
            raise ValueError('At least one objective weight must be positive')
        if self.rigidity<=0 or self.scale_compatibility<0:
            raise ValueError('Invalid deformation energy weights')
        if self.inner_iterations<1 or self.max_outer_iterations<1:
            raise ValueError('Iteration counts must be positive')
        if any(not isinstance(v,int) or isinstance(v,bool)
               for v in [self.inner_iterations,self.max_outer_iterations]):
            raise ValueError('Iteration counts must be integers')
        if not 0<self.relative_tolerance<1 or self.objective_tolerance<0:
            raise ValueError('Invalid stopping tolerances')
        if self.final_conformal_weight<=0 or self.concavity_exponent<=0:
            raise ValueError('Invalid adaptive weights')
        if not 0<self.nozzle_cone_angle<180:
            raise ValueError('Nozzle cone angle must be in (0,180)')
        if not 0<=self.base_band<1:
            raise ValueError('Build-plate band must be in [0,1)')
        if self.scale_solver not in ('auto','direct','iterative'):
            raise ValueError('Invalid Eq. 12 solver')
        if not 0<self.scale_tolerance<1:
            raise ValueError('Invalid Eq. 12 tolerance')
        if (not isinstance(self.scale_max_iterations,int) or isinstance(self.scale_max_iterations,bool)
                or self.scale_max_iterations<1):
            raise ValueError('Eq. 12 iteration limit must be a positive integer')


def floating_minima(mesh,scalar,base_vertices,edges=None):
    """Off-base minima of the PL field, including connected equal-value plateaus.

Every such minimum seeds a layer component away from the intended starting
surface. Absence is a necessary build-order check, not a support/collision proof.
"""
    if edges is None:
        edges=np.unique(np.sort(np.concatenate([mesh.cells[:,[i,j]] for i in range(4)
                                               for j in range(i+1,4)]),axis=1),axis=0)
    tolerance=1e-9*max(1.,float(np.ptp(scalar)))
    difference=scalar[edges[:,0]]-scalar[edges[:,1]]
    equal=edges[np.abs(difference)<=tolerance]
    graph=sparse.coo_matrix((np.ones(2*len(equal)),(equal.T.ravel(),equal[:,::-1].T.ravel())),
                           shape=(len(scalar),len(scalar))).tocsr()
    count,labels=connected_components(graph)
    has_lower=np.zeros(count,dtype=bool)
    has_lower[labels[edges[difference>tolerance,0]]]=True
    has_lower[labels[edges[difference<-tolerance,1]]]=True
    has_lower[labels[base_vertices]]=True
    roots=np.unique(labels,return_index=True)[1]
    return roots[~has_lower].tolist()


class Objectives:
    """Original-material-space Eq. 7 constraints.

    Surface selections are global mesh face indices. Multiple selected faces on
    a cell are intersected explicitly; an empty intersection raises, rather than
    silently choosing one face. The paper's single-face feasibility argument does
    not guarantee feasibility for this extension to multi-boundary-face cells.
    """
    def __init__(self,mesh,cfg,*,stress=None,stress_mask=None,sq_faces=None,sf_faces=None):
        self.mesh,self.cfg=mesh,cfg
        owner=mesh.face_cells[:,0]
        centers=mesh.points[mesh.cells].mean(axis=1)
        facecenters=mesh.points[mesh.faces].mean(axis=1)
        raw=mesh.normals()
        signs=np.sign(np.einsum('ij,ij->i',raw,facecenters-centers[owner]))
        self.signs=signs
        self.normals=raw*signs[:,None]
        self.sf=np.zeros(len(mesh.faces),dtype=bool)
        if cfg.weight_sf>0:
            if sf_faces is None:
                self.sf=mesh.boundary.copy()
                if cfg.exclude_build_plate:
                    tol=1e-8*max(1.,np.ptp(mesh.points,axis=0).max())
                    on_plate=np.all(np.abs(mesh.points[mesh.faces,2]-mesh.points[:,2].min())<=tol,axis=1)
                    self.sf &= ~on_plate
            else:
                self.sf[self._face_ids(sf_faces)]=True
        self.sq=np.zeros(len(mesh.faces),dtype=bool)
        if cfg.weight_sq>0:
            if sq_faces is None:
                self.sq=mesh.boundary.copy()
            else:
                self.sq[self._face_ids(sq_faces)]=True
        self.stress=None
        self.stress_mask=np.zeros(len(mesh.cells),dtype=bool)
        if cfg.weight_sr>0:
            if stress is None or stress_mask is None:
                raise ValueError('SR requires principal stress directions and an explicit critical-cell mask')
            stress=np.asarray(stress,dtype=float)
            mask=np.asarray(stress_mask,dtype=bool)
            if stress.shape!=(len(mesh.cells),3) or mask.shape!=(len(mesh.cells),):
                raise ValueError('Stress inputs have incorrect shape')
            self.stress=stress.copy()
            self.stress[mask]=unit(stress[mask])
            self.stress_mask=mask
        self.sf_by_cell=[fs[self.sf[fs]] for fs in mesh.cell_faces]
        self.sq_by_cell=[fs[self.sq[fs]] for fs in mesh.cell_faces]
        self.active=np.flatnonzero(self.sf[mesh.cell_faces].any(axis=1)|
                                   self.sq[mesh.cell_faces].any(axis=1)|self.stress_mask)

    def _face_ids(self,indices):
        ids=np.asarray(indices,dtype=int)
        if ids.ndim!=1 or np.any(ids<0) or np.any(ids>=len(self.mesh.faces)):
            raise ValueError('Invalid face selection')
        if not self.mesh.boundary[ids].all():
            raise ValueError('Fabrication surface selections must be boundary faces')
        return ids

    def project(self,ci,direction,preferred=None):
        cfg=self.cfg
        try:
            return project_printing_direction(direction,sf_normals=self.normals[self.sf_by_cell[ci]],
                     sq_normals=self.normals[self.sq_by_cell[ci]],
                     stress=self.stress[ci] if self.stress_mask[ci] else None,
                     alpha=cfg.alpha,beta=cfg.beta,gamma=cfg.gamma,preferred=preferred)
        except ValueError as exc:
            raise ValueError(f'Cell {ci}: {exc}') from exc

    def keep_weights(self,points):
        # Eq. 9 switches use actual deformed boundary normals, not R*n.
        cfg=self.cfg
        normals=self.mesh.normals(points)*self.signs[:,None]
        dots=normals[:,2]
        sf=self.sf & (dots < -np.sin(np.deg2rad(cfg.alpha))-1e-10)
        sq=self.sq & (np.abs(dots)>np.sin(np.deg2rad(cfg.gamma))+1e-10) & (np.abs(dots)<1-1e-10)
        weights=np.maximum.reduce([cfg.weight_sf*sf[self.mesh.cell_faces].any(axis=1),
                                   cfg.weight_sq*sq[self.mesh.cell_faces].any(axis=1),
                                   cfg.weight_sr*self.stress_mask])
        return weights

    def metric(self,scalar,progress=None):
        # Eq. 13: unweighted sum of spherical geodesic distances in radians.
        directions=unit(self.mesh.gradient(scalar))
        distances=[]
        last_update=time.perf_counter()
        for completed,ci in enumerate(self.active,1):
            closest,_=self.project(ci,directions[ci])
            # atan2 is stable near zero, unlike arccos(dot).
            distances.append(np.arctan2(np.linalg.norm(np.cross(directions[ci],closest)),
                                       np.clip(directions[ci]@closest,-1,1)))
            if progress and (completed==len(self.active) or time.perf_counter()-last_update>=.5):
                progress(f'{completed:,}/{len(self.active):,} active cells')
                last_update=time.perf_counter()
        return float(np.sum(distances)),float(max(distances,default=0.))


def concavity_weights(mesh,scalar,nozzle_angle,exponent):
    """Eqs. 5, 10, 11 on the shared-face-center isovalue.

    The intersection-edge direction is n_L cross the outward shared-face normal;
    this puts the L polygon on the left when viewed along the layer normal.
    """
    pairs=mesh.neighbor_pairs
    normals=unit(mesh.gradient(scalar))
    fs=np.flatnonzero(~mesh.boundary)
    raw=mesh.normals()[fs]
    centers=mesh.points[mesh.cells].mean(axis=1)
    toward=centers[pairs[:,1]]-centers[pairs[:,0]]
    raw*=np.sign(np.einsum('ij,ij->i',raw,toward))[:,None]
    left,right=normals[pairs[:,0]],normals[pairs[:,1]]
    h=np.cross(left,raw)
    lengths=np.linalg.norm(h,axis=1)
    valid=lengths>1e-12
    h[valid]/=lengths[valid,None]
    signed_sine=np.einsum('ij,ij->i',np.cross(left,right),h)
    omega=np.sign(signed_sine)*np.arccos(np.clip(np.einsum('ij,ij->i',left,right),-1,1))
    theta=np.deg2rad(nozzle_angle)
    if theta<np.pi/2:
        collisions=valid & (signed_sine>=-np.sin(theta)) & (signed_sine < -1e-12)
    else:
        collisions=valid & (signed_sine<=-np.sin(theta))
    weights=np.ones(len(pairs))
    if collisions.any():
        concave=valid & (omega<0)
        weights[concave]+=np.abs(omega[concave])**exponent
    return weights,int(collisions.sum())


def run_paper(mesh,config=None,*,stress=None,stress_mask=None,sq_faces=None,sf_faces=None,callback=None,
              progress_callback=None):
    """Paper stages 2.3(1–6), including Eq. 13 stopping and raw height transfer.

    Uses Z-up as printed in the paper. No implicit Y/Z conversion is performed.
    The iteration cap is a numerical guard and is reported as such, never as
    convergence. Element validity is monitored, not guaranteed by this method.
    ``callback`` receives completed outer-iteration metrics. Optional
    ``progress_callback`` receives stage messages and elapsed seconds, including
    throttled cell counts during projection and objective evaluation.
    """
    cfg=config or PaperConfig()
    run_started=time.perf_counter()
    context=''
    def emit(message):
        if progress_callback:
            progress_callback(dict(message=context+message,elapsed_seconds=time.perf_counter()-run_started))
    emit('Preparing build-plate constraints')
    base_height=float(mesh.points[:,2].min())
    base_tolerance=1e-8*max(1.,float(np.ptp(mesh.points,axis=0).max()))
    if cfg.base_band>0:
        # Pin a thin slab of the lowest surface instead of requiring a perfectly
        # flat contact face, so arbitrarily oriented uploads (STLs) still anchor
        # to the plate. The band is a fraction of the model's z-extent.
        span=max(float(np.ptp(mesh.points[:,2])),base_tolerance)
        boundary_vertices=np.unique(mesh.faces[mesh.boundary])
        base_vertices=boundary_vertices[mesh.points[boundary_vertices,2]<=base_height+cfg.base_band*span]
    else:
        base_faces=mesh.faces[mesh.boundary]
        base_faces=base_faces[np.all(np.abs(mesh.points[base_faces,2]-base_height)<=base_tolerance,axis=1)]
        base_vertices=np.unique(base_faces)
    base_cells=np.flatnonzero(np.isin(mesh.cells,base_vertices).sum(axis=1)>=3)
    if cfg.fix_build_plate and not len(base_vertices):
        raise ValueError('No flat build-plate contact faces found. Orient the input onto its intended base, '
                         'raise the build-plate pin band to anchor the lowest surface, or disable fixed build plate '
                         'for an unconstrained experiment.')
    emit('Preparing fabrication objectives')
    problem=Objectives(mesh,cfg,stress=stress,stress_mask=stress_mask,sq_faces=sq_faces,sf_faces=sf_faces)
    emit(f'Checking mesh connectivity · {len(problem.active):,} active cells')
    points=mesh.points.copy(); points[:,2]-=points[:,2].min()
    edges=np.unique(np.sort(np.concatenate([mesh.cells[:,[i,j]] for i in range(4)
                                           for j in range(i+1,4)]),axis=1),axis=0)
    if cfg.fix_build_plate and cfg.enforce_build_order and floating_minima(mesh,points[:,2],base_vertices,edges):
        raise ValueError('The input has a disconnected or elevated starting region that is not connected downward '
                         'to the build plate. Orient the part so every region drains to the plate, or turn off '
                         'build-order enforcement to experiment on organic models (some regions may then need support).')
    guide=None
    if cfg.fix_build_plate:
        emit('Computing build-plate distance guide')
        lengths=np.linalg.norm(mesh.points[edges[:,0]]-mesh.points[edges[:,1]],axis=1)
        graph=sparse.coo_matrix((np.r_[lengths,lengths],(edges.T.ravel(),edges[:,::-1].T.ravel())),
                               shape=(len(points),len(points))).tocsr()
        distance=dijkstra(graph,indices=base_vertices,min_only=True)
        guide=mesh.gradient(distance)
    emit('Preparing deformation frames')
    p0=mesh.points[mesh.cells]; p0-=p0.mean(axis=1,keepdims=True)
    pinv=np.linalg.pinv(p0)
    history=[]; rotations=np.tile(np.eye(3),(len(mesh.cells),1,1))
    blend_workspace=QuaternionBlendWorkspace()
    scales=np.ones((len(mesh.cells),3))
    emit('Evaluating initial objective (Eq. 13)')
    previous,_=problem.metric(points[:,2],
        (lambda counts: emit('Evaluating initial objective (Eq. 13) · '+counts)) if progress_callback else None)
    emit(f'Initial objective Π {previous:.5g}')
    initial_metric=previous
    reason='iteration_limit'
    if previous<=cfg.objective_tolerance:
        reason='objective_tolerance'
    else:
        for outer in range(cfg.max_outer_iterations):
            started=time.perf_counter()
            context=f'Outer {outer+1}/{cfg.max_outer_iterations} · '
            emit('Fitting rotations and updating weights')
            current=points[mesh.cells];current-=current.mean(axis=1,keepdims=True)
            f=(pinv@current).transpose(0,2,1)
            u,_,vt=np.linalg.svd(f)
            correction=np.ones((len(mesh.cells),3));correction[:,2]=np.linalg.det(u@vt)
            rotations=(u*correction[:,None,:])@vt
            keep=problem.keep_weights(points)
            if outer and cfg.concavity_control:
                edge_weights,collisions=concavity_weights(mesh,points[:,2],cfg.nozzle_cone_angle,cfg.concavity_exponent)
            else:
                edge_weights=np.ones(len(mesh.neighbor_pairs));collisions=0
            for inner in range(cfg.inner_iterations):
                context=f'Outer {outer+1}/{cfg.max_outer_iterations} · Inner {inner+1}/{cfg.inner_iterations} · '
                emit(f'Projecting fabrication constraints (Eq. 7) · 0/{len(problem.active):,} active cells')
                last_update=time.perf_counter()
                targets=rotations.copy();conformal=np.zeros(len(mesh.cells),dtype=bool)
                projected=np.empty((len(problem.active),3))
                for completed,ci in enumerate(problem.active,1):
                    d=rotations[ci,2,:]
                    closest,conformal[ci]=problem.project(ci,d,preferred=guide[ci] if guide is not None else None)
                    projected[completed-1]=closest
                    if progress_callback and (completed==len(problem.active) or time.perf_counter()-last_update>=.5):
                        emit(f'Projecting fabrication constraints (Eq. 7) · {completed:,}/{len(problem.active):,} active cells')
                        last_update=time.perf_counter()
                if len(problem.active):
                    active_rotations=rotations[problem.active]
                    world=np.einsum('cij,cj->ci',active_rotations,projected)
                    targets[problem.active]=from_two_vectors_batch(world,np.array([0.,0.,1.]))@active_rotations
                inner_keep=keep.copy()
                # §3.3.1's final local/global step strengthens SQ-C targets.
                if inner==cfg.inner_iterations-1:
                    inner_keep[conformal]=np.maximum(inner_keep[conformal],cfg.final_conformal_weight)
                emit('Blending rotations (Eq. 8) · solving sparse system; reusing factorization when unchanged')
                rotations=blend_quaternions(mesh,rotations,targets,inner_keep,edge_weights,
                    fixed_identity=base_cells if cfg.fix_build_plate else None,workspace=blend_workspace)
            context=f'Outer {outer+1}/{cfg.max_outer_iterations} · '
            emit('Solving deformation and scales (Eq. 12)')
            initial_points=points.copy()
            initial_points[:,2]+=base_height if cfg.fix_build_plate else 0.
            proposed,proposed_scales=solve_paper_scales(mesh,rotations,cfg.rigidity,cfg.scale_compatibility,
                                           fixed_vertices=base_vertices if cfg.fix_build_plate else None,
                                           progress=(lambda message: emit('Eq. 12 · '+message)) if progress_callback else None,
                                           solver=cfg.scale_solver,initial=np.r_[initial_points.ravel(),scales.ravel()],
                                           tolerance=cfg.scale_tolerance,max_iterations=cfg.scale_max_iterations)
            proposed[:,2]-=base_height if cfg.fix_build_plate else proposed[:,2].min()
            step=1.
            if cfg.fix_build_plate:
                for attempt in range(25):
                    emit(f'Checking build feasibility · attempt {attempt+1}/25 · step {step:.5g}')
                    candidate=points+step*(proposed-points)
                    determinant=np.linalg.det(np.einsum('cij,cik->ckj',mesh.grad_basis,candidate[mesh.cells]))
                    if (determinant.min()>1e-6 and candidate[:,2].min()>=-base_tolerance
                            and (not cfg.enforce_build_order
                                 or not floating_minima(mesh,candidate[:,2],base_vertices,edges))):
                        break
                    step*=.5
                else:
                    reason='build_feasibility_limit'
                    break
                points=candidate
                scales=scales+step*(proposed_scales-scales)
            else:points,scales=proposed,proposed_scales
            emit('Evaluating objective (Eq. 13)')
            metric,worst=problem.metric(points[:,2],
                (lambda counts: emit('Evaluating objective (Eq. 13) · '+counts)) if progress_callback else None)
            f=np.einsum('cij,cik->ckj',mesh.grad_basis,points[mesh.cells])
            det=np.linalg.det(f)
            relative=abs(previous-metric)/max(previous,cfg.objective_tolerance,1e-15)
            row=dict(outer=outer+1,pi=metric,worst_angle_degrees=float(np.rad2deg(worst)),
                     relative_change=relative,inverted_cells=int((det<0).sum()),
                     min_determinant=float(det.min()),max_determinant=float(det.max()),
                     local_collision_pairs=collisions,step_fraction=step,seconds=time.perf_counter()-started)
            history.append(row)
            if callback:callback(row)
            if metric<=cfg.objective_tolerance:
                reason='objective_tolerance';break
            if relative<cfg.relative_tolerance:
                reason='relative_stagnation';break
            previous=metric
    context=''
    emit('Finalizing height field and checking build connectivity')
    scalar=points[:,2].copy()
    build=dict(fixed=cfg.fix_build_plate,base_vertices=base_vertices.tolist(),
               base_scalar_span=float(np.ptp(scalar[base_vertices])) if len(base_vertices) else None,
               base_scalar_max=float(scalar[base_vertices].max()) if len(base_vertices) else None,
               below_plate_vertices=int((scalar < -base_tolerance).sum()),
               floating_minimum_vertices=floating_minima(mesh,scalar,base_vertices,edges),
               minimum_scalar=float(scalar.min()))
    emit(f'Finished · {reason} · {len(history)} outer iterations')
    return dict(points=mesh.points.copy(),cells=mesh.cells.copy(),deformed=points,
                scalar=scalar,rotations=rotations,scales=scales,history=history,
                initial_pi=initial_metric,stop_reason=reason,config=asdict(cfg),build_plate=build)
