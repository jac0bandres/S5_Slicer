"""Run paper-defined S³ deformation and scalar-layer extraction on a .tet mesh."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy
from .mesh import read_tet, write_tet, TetMesh
from .deform import SupportFreeConfig, support_free
from .pipeline import PaperConfig, run_paper
from .layers import write_layers
from .adaptive import AdaptiveConfig
from .toolpaths import ToolpathConfig
from .visual_pipeline import build_visualization,write_visualization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh',type=Path)
    parser.add_argument('-o','--output',type=Path,required=True)
    parser.add_argument('--implementation',choices=['paper','cpp-reference'],default='paper')
    parser.add_argument('--config',type=Path,help='JSON configuration overrides for the selected implementation')
    parser.add_argument('--objectives',type=Path,help='NPZ with stress, stress_mask, sq_faces and/or sf_faces')
    parser.add_argument('--up-axis',choices=['y','z'],default='z',help='Input up axis; paper uses Z, official meshes use Y')
    parser.add_argument('--layers',type=int,default=0,help='Export this many scalar isosurfaces as OBJ')
    parser.add_argument('--toolpaths',action='store_true',help='Generate surface deposition strokes (defaults to 12 layers)')
    parser.add_argument('--layer-mode',choices=['fixed','adaptive'],default='fixed')
    parser.add_argument('--min-thickness',type=float,default=.16)
    parser.add_argument('--max-thickness',type=float,default=.4)
    parser.add_argument('--path-mode',choices=['contour','hybrid'],default='contour')
    parser.add_argument('--path-spacing',type=float,default=.4)
    parser.add_argument('--waypoint-distance',type=float,default=.4)
    parser.add_argument('--preview-html',action='store_true',help='Export offline interactive preview (requires plotly)')
    args = parser.parse_args()
    if args.layers<0: parser.error('Layer count must be nonnegative')
    if (args.preview_html or args.layer_mode=='adaptive') and not args.toolpaths:
        parser.error('--preview-html and adaptive layer mode require --toolpaths')
    if args.toolpaths and args.implementation!='paper':
        parser.error('Surface toolpaths currently use the Z-up paper pipeline')
    path_cfg=ToolpathConfig(spacing=args.path_spacing,waypoint_distance=args.waypoint_distance,mode=args.path_mode)
    adaptive=AdaptiveConfig(minimum=args.min_thickness,maximum=args.max_thickness,
                            tolerance=min(.005,args.min_thickness/10)) if args.layer_mode=='adaptive' else None
    if args.preview_html:
        try:import plotly
        except ImportError:parser.error('Install s3/requirements-ui.txt for HTML visualization')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty (preserves prior experiments)')
    config_type=PaperConfig if args.implementation=='paper' else SupportFreeConfig
    cfg = config_type(**(json.loads(args.config.read_text()) if args.config else {}))
    mesh=read_tet(args.mesh,cpp_float32=args.implementation=='cpp-reference')
    objectives={}
    if args.objectives:
        with np.load(args.objectives,allow_pickle=False) as data:
            unknown=set(data.files)-{'stress','stress_mask','sq_faces','sf_faces'}
            if unknown: parser.error(f'Unknown objective arrays: {sorted(unknown)}')
            objectives={k:data[k] for k in data.files}
    target_up='z' if args.implementation=='paper' else 'y'
    if args.up_axis!=target_up:
        transform=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=float)
        if target_up=='y': transform=transform.T
        mesh=TetMesh(mesh.points@transform.T,mesh.cells)
        if 'stress' in objectives:objectives['stress']=objectives['stress']@transform.T
    start = time.perf_counter()
    runner=run_paper if args.implementation=='paper' else support_free
    if args.implementation=='cpp-reference' and objectives:
        parser.error('C++ comparison path currently supports SF only')
    result = runner(mesh,cfg,**objectives,callback=lambda row: print(json.dumps(row),flush=True))
    visual=None
    if args.toolpaths:
        visual=build_visualization(result,count=args.layers or 12,adaptive=adaptive,toolpath_config=path_cfg,
            stress=objectives.get('stress'),stress_mask=objectives.get('stress_mask'),
            callback=lambda row:print(json.dumps(row),flush=True))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'result.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    write_tet(args.output/'deformed.tet',result['deformed'],result['cells'])
    manifest = []
    if visual is not None:
        manifest=write_visualization(args.output,result,*visual,html=args.preview_html)
    elif args.layers:
        mesh = TetMesh(result['points'],result['cells'])
        manifest = write_layers(mesh,result['scalar'],args.layers,args.output/'layers')
    report = dict(status='deformation/layers/toolpaths for visual inspection' if visual else 'deformation/scalar stages',
                  print_ready=False,visualization=visual[2] if visual else None,
                  build_plate=result.get('build_plate'),
                  implementation=args.implementation,input_up_axis=args.up_axis,output_up_axis=target_up,
                  source_commit='a77ef04115b1383a58f228a7f67ce4434c9d09f3',
                  source_path=str(args.mesh.resolve()),sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                  objectives_path=str(args.objectives.resolve()) if args.objectives else None,
                  objectives_sha256=hashlib.sha256(args.objectives.read_bytes()).hexdigest() if args.objectives else None,
                  input_precision='float64' if args.implementation=='paper' else 'float32',
                  config=asdict(cfg),history=result['history'],layers=manifest,
                  stop_reason=result.get('stop_reason','fixed_cpp_iterations'),initial_pi=result.get('initial_pi'),
                  seconds=time.perf_counter()-start,
                  versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__))
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Wrote {args.output}; elapsed {report["seconds"]:.1f}s')


if __name__ == '__main__':
    main()
