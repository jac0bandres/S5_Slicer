"""Shared layer/toolpath generation and reviewable geometry exports for UI/CLI."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .adaptive import AdaptiveConfig,adaptive_layers
from .layers import isosurface
from .mesh import TetMesh
from .surface import Surface
from .toolpaths import ToolpathConfig,generate_toolpaths


def build_visualization(result,*,count=12,adaptive=None,toolpath_config=None,
                        stress=None,stress_mask=None,callback=None):
    mesh=TetMesh(result['points'],result['cells'])
    if adaptive is not None:
        layers,layer_report=adaptive_layers(mesh,result['scalar'],adaptive,callback)
        layer_report.update(mode='adaptive',config=asdict(adaptive))
    else:
        if not isinstance(count,int) or count<1:raise ValueError('Layer count must be positive')
        scalar=result['scalar'];span=float(np.ptp(scalar))
        if not np.isfinite(scalar).all() or span<=0:raise ValueError('Invalid layer scalar field')
        layers=[]
        for level in scalar.min()+(np.arange(count)+.5)*span/count:
            points,faces,cells=isosurface(mesh,scalar.copy(),float(level),return_cell_ids=True)
            if len(faces):layers.append(Surface(points,faces,float(level),cell_ids=cells))
        layer_report=dict(mode='fixed',requested_count=count,delivered_count=len(layers),coverage_certified=False)
    if not layers:raise ValueError('No curved layers intersect the model at the requested levels')
    cfg=toolpath_config or ToolpathConfig()
    paths=[];reports=[]
    for index,layer in enumerate(layers):
        try:
            curves,report=generate_toolpaths(layer,cfg,stress=stress,stress_mask=stress_mask)
        except ValueError as error:
            raise ValueError(f'Layer {index+1} (scalar {layer.level:.6g}): {error}') from error
        paths.append(curves);reports.append(dict(index=index,level=layer.level,**report))
        if callback:callback(dict(stage='toolpaths',count=index+1,total=len(layers),curves=len(curves)))
    report=dict(layers=layer_report,toolpath_config=asdict(cfg),toolpaths=reports,
                build_plate=result.get('build_plate'),
                total_curves=sum(r['curves'] for r in reports),
                total_waypoints=sum(r['waypoints'] for r in reports),
                total_length=sum(r['length'] for r in reports),
                empty_layer_indices=[i for i,p in enumerate(paths) if not p],
                coordinates='original material space',travel_planning=False,coverage_certified=False)
    return layers,paths,report


def write_visualization(directory,result,layers,paths,report,*,html=True):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    layer_dir=root/'layers';layer_dir.mkdir(exist_ok=True)
    path_dir=root/'toolpaths';path_dir.mkdir(exist_ok=True)
    manifest=[]
    for index,(layer,curves) in enumerate(zip(layers,paths)):
        name=f'{index:04d}'
        with (layer_dir/f'{name}.obj').open('w') as f:
            f.write(f'# S3 scalar level {layer.level:.17g}\n')
            np.savetxt(f,layer.points,fmt='v %.17g %.17g %.17g')
            np.savetxt(f,layer.faces+1,fmt='f %d %d %d')
        records=[];offset=1
        with (path_dir/f'{name}.obj').open('w') as f:
            for ci,curve in enumerate(curves):
                f.write(f'g {curve.kind}_{ci}\n')
                np.savetxt(f,curve.points,fmt='v %.17g %.17g %.17g')
                f.write('l '+' '.join(str(i) for i in range(offset,offset+len(curve.points)))+'\n')
                offset+=len(curve.points)
                records.append(dict(kind=curve.kind,closed=curve.closed,field_level=curve.level,
                                    points=curve.points.tolist(),normals=curve.normals.tolist(),length=curve.length))
        (path_dir/f'{name}.json').write_text(json.dumps(dict(layer=index,scalar_level=layer.level,paths=records))+'\n')
        manifest.append(dict(file=f'{name}.obj',level=layer.level,inserted=layer.inserted,
                             vertices=len(layer.points),faces=len(layer.faces),curves=len(curves)))
    (root/'visualization.json').write_text(json.dumps(report,indent=2)+'\n')
    if html:
        from .visualization import write_html
        write_html(root/'preview.html',result,layers,paths,report)
    return manifest
