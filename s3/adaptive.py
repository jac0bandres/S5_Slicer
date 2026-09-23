"""S³ §4.2: separated initial layers, midpoint insertion, partial-layer trimming.

Distance decisions use conservative triangle-surface bounds. This implementation
reports numerical resolution limits explicitly; it does not infer a continuous
thickness guarantee from vertex samples alone.
"""
from dataclasses import dataclass
import numpy as np
from .layers import isosurface,first_bed_layer
from .surface import Surface,separation_bounds,triangle_distance_bounds,subdivide,conform_surface


@dataclass(frozen=True)
class AdaptiveConfig:
    minimum: float=.16
    maximum: float=.4
    tolerance: float=.005
    max_layers: int=2000
    max_refinements: int=2000
    trim_depth: int=6
    first_layer_height: float | None=None

    def __post_init__(self):
        if not (0<self.tolerance<self.minimum<self.maximum):
            raise ValueError('Require 0 < tolerance < minimum < maximum')
        if self.first_layer_height is not None and (not np.isfinite(self.first_layer_height) or self.first_layer_height<=0):
            raise ValueError('First-layer height must be finite and positive')
        if self.max_layers<2 or self.max_refinements<0 or self.trim_depth<0:
            raise ValueError('Invalid adaptive budgets')


def _surface(mesh,scalar,level,inserted=False):
    points,faces,owners=isosurface(mesh,scalar.copy(),level,return_cell_ids=True)
    return Surface(points,faces,level,inserted,owners) if len(faces) else None


def _bounds(top,bottom,cfg,criterion='both'):
    return separation_bounds(top,bottom,tolerance=cfg.tolerance,
                             max_refinements=cfg.max_refinements,
                             minimum_threshold=cfg.minimum-cfg.tolerance if criterion in ('minimum','both') else None,
                             maximum_threshold=cfg.maximum+cfg.tolerance if criterion in ('maximum','both') else None)


def _union(layers):
    points=[];faces=[];offset=0
    for layer in layers:
        points.append(layer.points);faces.append(layer.faces+offset);offset+=len(layer.points)
    return Surface(np.vstack(points),np.vstack(faces))


def trim_inserted(layer,previous,cfg):
    """Keep only polygon pieces whose minimum distance satisfies the margin.

    Mixed pieces are subdivided. At the depth limit unresolved pieces are removed
    conservatively and counted; their removal is not a coverage certificate.
    """
    target=_union(previous)
    kept=[];owners=[];removed=0;unresolved=0
    stack=[(tri,0,i) for i,tri in enumerate(layer.triangles)]
    while stack:
        tri,depth,owner=stack.pop()
        lo,_,_,hi=triangle_distance_bounds(tri,target)
        if lo>=cfg.minimum-cfg.tolerance:
            kept.append(tri)
            owners.append(owner)
        elif hi<cfg.minimum-cfg.tolerance:
            removed+=1
        elif depth>=cfg.trim_depth:
            removed+=1;unresolved+=1
        else:
            stack.extend((child,depth+1,owner) for child in subdivide(tri))
    if not kept:return None,dict(removed_pieces=removed,unresolved_pieces=unresolved)
    raw=np.asarray(kept).reshape(-1,3)
    points,inverse=np.unique(raw,axis=0,return_inverse=True)
    cells=None if layer.cell_ids is None else layer.cell_ids[owners]
    return conform_surface(Surface(points,inverse.reshape(-1,3),layer.level,True,cells)),dict(removed_pieces=removed,unresolved_pieces=unresolved)


def adaptive_layers(mesh,scalar,config=None,callback=None):
    cfg=config or AdaptiveConfig()
    field=np.asarray(scalar,dtype=float)
    if field.shape!=(len(mesh.points),) or not np.isfinite(field).all():
        raise ValueError('Invalid scalar field')
    # Shift scalar origin; first-layer selection uses physical bed distance.
    field=field-field.min()
    upper=float(field.max())
    first,bed_report=first_bed_layer(mesh,field,cfg.first_layer_height if cfg.first_layer_height is not None else cfg.minimum)
    layers=[first];history=[]
    gradient_scale=float(np.linalg.norm(mesh.gradient(field),axis=1).max())
    delta=max(cfg.minimum*gradient_scale,upper*1e-8)
    epsilon=max(1e-8,upper*1e-8)
    # Step 1: a level is retained only when its minimum distance lower bound
    # clears d_min. Scalar increments are a search heuristic, not a thickness test.
    while len(layers)<cfg.max_layers:
        level=layers[-1].level+delta
        accepted=None
        while level<upper-epsilon:
            candidate=_surface(mesh,field,level)
            if candidate is None:
                level+=delta;continue
            bounds=_bounds(candidate,layers[-1],cfg,'minimum')
            if bounds['min_lower']>=cfg.minimum-cfg.tolerance:
                accepted=candidate;break
            level+=delta*.25
        if accepted is None:break
        layers.append(accepted)
        if callback:callback(dict(stage='initial_layers',count=len(layers),level=level))
    if len(layers)>=cfg.max_layers:
        raise RuntimeError('Initial layer search reached max_layers')
    # Step 2: midpoint refinement until every directed top-to-bottom maximum
    # has a conservative upper bound at most d_max (within stated tolerance).
    i=0
    while i<len(layers)-1:
        bottom,top=layers[i:i+2]
        bounds=_bounds(top,bottom,cfg,'maximum')
        if bounds['max_upper']<=cfg.maximum+cfg.tolerance:
            i+=1;continue
        level=(bottom.level+top.level)/2
        if len(layers)>=cfg.max_layers or top.level-bottom.level<epsilon:
            raise RuntimeError('Cannot certify maximum layer spacing within refinement budget')
        middle=_surface(mesh,field,level,True)
        if middle is None:raise RuntimeError('Empty intermediate isosurface during refinement')
        layers.insert(i+1,middle)
        if callback:callback(dict(stage='insert_layer',count=len(layers),level=level))
    # Step 3: only newly inserted layers are trimmed; initial layers are retained.
    trimmed=[layers[0]]
    for layer in layers[1:]:
        level,inserted=layer.level,layer.inserted
        info={}
        if layer.inserted:
            layer,info=trim_inserted(layer,trimmed,cfg)
        if layer is not None:
            trimmed.append(layer)
        history.append(dict(level=level,inserted=inserted,removed_layer=layer is None,**info))
    # Validate the delivered surfaces after trimming, against previously deposited
    # surfaces. This does not certify coverage of the entire input volume.
    checks=[]
    for i,layer in enumerate(trimmed[1:],1):
        bounds=_bounds(layer,_union(trimmed[:i]),cfg)
        checks.append(dict(level=layer.level,**bounds,
                           minimum_ok=bounds['min_lower']>=cfg.minimum-cfg.tolerance,
                           maximum_ok=bounds['max_upper']<=cfg.maximum+cfg.tolerance))
        if callback:callback(dict(stage='spacing_checks',count=i,total=len(trimmed)-1))
    return trimmed,dict(initial_and_inserted_count=len(layers),delivered_count=len(trimmed),
                        trim_history=history,distance_checks=checks,
                        spacing_passed=all(c['minimum_ok'] and c['maximum_ok'] for c in checks),
                        coverage_certified=False,
                        first_layer=bed_report,first_layer_level=first.level,last_layer_level=trimmed[-1].level,
                        remaining_scalar_range=upper-trimmed[-1].level)
