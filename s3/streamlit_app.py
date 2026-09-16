"""Launch from the repository: streamlit run s3/streamlit_app.py."""
from pathlib import Path
import sys
import io
import json
import hashlib
import tempfile
import time
import zipfile
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from s3.mesh import TetMesh, read_tet, write_tet
from s3.pipeline import PaperConfig, run_paper
from s3.layers import isosurface, write_layers
from s3.adaptive import AdaptiveConfig
from s3.toolpaths import ToolpathConfig
from s3.visual_pipeline import build_visualization,write_visualization
from s3.visualization import deformation_figure,toolpath_figure
from s3.stl_import import tetrahedralize, SUFFIXES

ROOT = Path(__file__).resolve().parent


def figure(points, faces, scalar=None, name='Surface', opacity=1.):
    trace = go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                      i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                      intensity=scalar, colorscale='Viridis', color='#38bdf8',
                      opacity=opacity, name=name, showscale=scalar is not None,
                      colorbar=dict(title='Scalar height'))
    fig = go.Figure(trace)
    fig.update_layout(scene=dict(aspectmode='data', xaxis_title='X (mm)',
                                yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
                      margin=dict(l=0, r=0, t=0, b=0), height=550,
                      uirevision=name)
    return fig


def apply_up_axis(mesh, up_axis):
    """Rotate a Y-up mesh into the Z-up frame used throughout the workbench."""
    if up_axis == 'Y':
        return TetMesh(mesh.points @ np.array([[1,0,0],[0,0,1],[0,-1,0]]), mesh.cells)
    return mesh


@st.cache_data(show_spinner=False, max_entries=8)
def load_mesh(payload, up_axis, kind='tet'):
    if kind == 'tet':
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.tet'
            path.write_bytes(payload)
            mesh = read_tet(path)
    else:
        mesh = tetrahedralize(payload, kind)
    return apply_up_axis(mesh, up_axis)


def bundle(result, report, count, visual=None):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        np.savez_compressed(root/'result.npz', **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        write_tet(root/'deformed.tet', result['deformed'], result['cells'])
        if visual is None:
            manifest = write_layers(TetMesh(result['points'], result['cells']), result['scalar'], count, root/'layers')
        else:
            layers,paths,visual_report=visual
            manifest=write_visualization(root,result,layers,paths,visual_report)
            report=dict(report,visualization=visual_report)
        (root/'report.json').write_text(json.dumps(dict(report, layers=manifest), indent=2))
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file(): archive.write(path, path.relative_to(root))
        return output.getvalue()


def main():
    st.set_page_config(page_title='S³ Research Workbench', page_icon='🧊', layout='wide')
    st.title('S³ Research Workbench')
    st.caption('Paper-based deformation • adaptive curved layers • surface toolpaths')
    st.info('Inspect deformation and sampled deposition strokes here. Fixed-count layers are previews; adaptive spacing checks and path diagnostics are available below.')
    with st.sidebar:
        st.header('Mesh')
        examples = {'Cantilever · quick test': ROOT/'tests/fixtures/cantilever.tet'}
        official = ROOT.parents[1]/'S3_DeformFDM/DataSet/TET_MODEL'
        examples.update({f'Official · {p.stem}': p for p in sorted(official.glob('*.tet'))})
        source = st.selectbox('Source', list(examples)+['Upload .tet','Upload STL / surface'])
        upload = surface = None
        if source=='Upload .tet':
            upload = st.file_uploader('Tetrahedral mesh', type=['tet'])
        elif source=='Upload STL / surface':
            surface = st.file_uploader('Surface mesh (tetrahedralized on upload)', type=list(SUFFIXES),
                help='STL/OBJ/PLY/OFF surfaces are volume-meshed with TetGen. Watertight, self-intersection-free surfaces work best.')
        axis = st.selectbox('Input up axis', ['Z','Y'], index=1 if source.startswith('Official') else 0,
                            help='Official S³ datasets use Y-up. All displayed results use Z-up.')
        st.header('Objectives')
        sf = st.slider('Support-free weight', 0., 1., 1., .05)
        sr = st.slider('Strength weight', 0., 1., 0., .05)
        sq = st.slider('Surface-quality weight', 0., 1., 0., .05)
        objective_upload = st.file_uploader('Objective fields (.npz)', type=['npz'],
                            help='Strength requires stress and stress_mask. Optional sf_faces and sq_faces select global boundary-face indices.')
        with st.expander('Angles and deformation'):
            alpha = st.number_input('Support angle α (degrees)', 0., 89., 30.)
            beta = st.number_input('Strength angle β (degrees)', 0., 89., 10.)
            gamma = st.number_input('Surface angle γ (degrees)', 0., 89., 10.)
            rigidity = st.number_input('Rigidity', .001, value=1.)
            compatibility = st.number_input('Scale compatibility', 0., value=6.)
            concavity = st.checkbox('Concavity weighting', True)
            plate = st.checkbox('Exclude build-plate faces from SF', True)
            fix_plate = st.checkbox('Fix build-plate contact surface', True,
                help='Hold the original flat base in place so the optimizer cannot tilt the entire part off its intended starting surface.')
            base_band = st.number_input('Build-plate pin band (fraction of height)', 0., .5,
                .02 if source=='Upload STL / surface' else 0., .005, format='%.3f',
                help='0 requires a perfectly flat contact face. Raise it to pin a thin slab of the lowest surface '
                     'instead — useful for STL/surface uploads that do not rest on a flat base.')
            enforce_order = st.checkbox('Enforce downward build-order connectivity',
                source!='Upload STL / surface',
                help='On: reject inputs with pockets/regions that do not drain to the plate (a build-validity '
                     'guarantee). Off: allow experimenting on organic models — some regions may need support.')
        st.header('Run budget')
        outer = st.number_input('Maximum outer iterations', 1, 100, 5)
        inner = st.number_input('Inner iterations', 1, 50, 7)
        tolerance = st.number_input('Relative stopping tolerance', .0001, .99, .05, format='%.4f')
        st.caption('Relative stagnation is not objective satisfaction. Large meshes can take several minutes per run.')
        run = st.button('Run paper pipeline', type='primary', width='stretch')
    try:
        if surface is not None:
            payload, kind = surface.getvalue(), Path(surface.name).suffix.lstrip('.').lower()
        elif upload is not None:
            payload, kind = upload.getvalue(), 'tet'
        elif source in examples:
            payload, kind = examples[source].read_bytes(), 'tet'
        else:
            payload = kind = None
        if payload is None:
            st.info('Upload a .tet mesh or an STL/OBJ/PLY/OFF surface to begin. '
                    'Surfaces are tetrahedralized with TetGen on upload.')
            return
        with st.spinner('Tetrahedralizing surface…' if kind!='tet' else 'Reading mesh…'):
            mesh = load_mesh(payload, axis, kind)
        fields = {}
        field_bytes = objective_upload.getvalue() if objective_upload else b''
        if field_bytes:
            with np.load(io.BytesIO(field_bytes), allow_pickle=False) as data:
                if set(data.files)-{'stress','stress_mask','sf_faces','sq_faces'}:
                    raise ValueError('Unknown objective arrays; use stress, stress_mask, sf_faces, sq_faces.')
                fields = {k:data[k] for k in data.files}
            if axis=='Y' and 'stress' in fields:
                fields['stress'] = fields['stress'] @ np.array([[1,0,0],[0,0,1],[0,-1,0]])
        cfg = PaperConfig(alpha=alpha,beta=beta,gamma=gamma,weight_sf=sf,weight_sr=sr,weight_sq=sq,
                          rigidity=rigidity,scale_compatibility=compatibility,concavity_control=concavity,
                          exclude_build_plate=plate,max_outer_iterations=int(outer),inner_iterations=int(inner),
                          fix_build_plate=fix_plate,base_band=base_band,enforce_build_order=enforce_order,
                          relative_tolerance=tolerance)
    except Exception as error:
        st.error(f'Input error: {error}')
        return
    fingerprint = hashlib.sha256(payload+field_bytes+axis.encode()+json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()
    a,b = st.columns(2)
    a.metric('Vertices', f'{len(mesh.points):,}')
    b.metric('Tetrahedra', f'{len(mesh.cells):,}')
    if run:
        if sr>0 and not {'stress','stress_mask'} <= fields.keys():
            st.error('Strength optimization needs both stress and stress_mask in the objective NPZ.')
        else:
            started = time.perf_counter()
            with st.status('Solving paper equations…', expanded=True) as status:
                progress = st.empty()
                try:
                    result = run_paper(mesh,cfg,**fields,callback=lambda row: progress.write(
                        f"Outer {row['outer']}/{outer} · Π {row['pi']:.5g} · worst violation {row['worst_angle_degrees']:.2f}° · inverted cells {row['inverted_cells']}"))
                    report = dict(implementation='paper',config=asdict(cfg),input_up_axis=axis,output_up_axis='Z',
                                  mesh_sha256=hashlib.sha256(payload).hexdigest(),
                                  objectives_sha256=hashlib.sha256(field_bytes).hexdigest() if field_bytes else None,
                                  history=result['history'],initial_pi=result['initial_pi'],stop_reason=result['stop_reason'],
                                  build_plate=result['build_plate'],
                                  seconds=time.perf_counter()-started,print_ready=False)
                    st.session_state['experiment'] = (fingerprint,result,report)
                    st.session_state['experiment_fields'] = fields
                    st.session_state.pop('export',None)
                    st.session_state.pop('visual',None)
                    status.update(label='Run finished',state='complete',expanded=False)
                except Exception as error:
                    status.update(label='Run failed',state='error')
                    st.error(str(error))
                    st.caption('Some combinations of boundary-face objectives are infeasible; the solver reports them instead of relaxing constraints silently.')
    saved = st.session_state.get('experiment')
    if not saved:
        st.plotly_chart(figure(mesh.points,mesh.faces[mesh.boundary],name='Input'),width='stretch')
        return
    saved_id,result,report = saved
    if saved_id != fingerprint:
        st.warning('Controls or input changed. The views below still show the last successful run; run again to apply changes.')
    original = TetMesh(result['points'],result['cells'])
    faces = original.faces[original.boundary]
    st.subheader('Last successful run')
    st.write(f"Stop: **{report['stop_reason']}** · {report['seconds']:.1f} seconds · {len(report['history'])} outer iterations")
    if not report.get('build_plate',{}).get('fixed',False):
        st.warning('This saved run does not fix the build-plate contact surface. Global tilting can create layers that do not start from the intended base. Run again with the contact surface fixed.')
    elif report['build_plate']['below_plate_vertices']:
        st.error('The deformation places material below the fixed starting plane. This result does not define a valid build from that plate.')
    tabs = st.tabs(['Geometry','Curved layers','Toolpaths','Diagnostics','Download'])
    with tabs[0]:
        left,right = st.columns(2)
        with left:
            st.markdown('**Original geometry / scalar field**')
            st.plotly_chart(figure(result['points'],faces,result['scalar'],name='Original'),width='stretch')
        with right:
            st.markdown('**Deformed geometry**')
            st.plotly_chart(figure(result['deformed'],faces,name='Deformed'),width='stretch')
        with st.expander('Explore the deformation'):
            amount=st.slider('Deformation amount',0.,1.,1.,.05)
            st.plotly_chart(deformation_figure(result,amount),width='stretch')
            st.caption('Interpolation between the original and final geometry, with the original shown as a translucent reference.')
    with tabs[1]:
        layer_mode=st.radio('Layer generation',['Fixed count','Adaptive thickness'],horizontal=True)
        if layer_mode=='Fixed count':
            st.info('Fixed count is a sparse visual preview, not a deposition sequence. Its first surface may be well above the plate. Choose Adaptive thickness to inspect layers at physical thicknesses.')
        count = st.slider('Diagnostic layer count', 1, 100, 12)
        adaptive=None
        if layer_mode=='Adaptive thickness':
            c1,c2=st.columns(2)
            minimum=c1.number_input('Minimum layer thickness (mm)',min_value=.01,value=.16,step=.01)
            maximum=c2.number_input('Maximum layer thickness (mm)',min_value=.02,value=.4,step=.01)
            st.caption('Adaptive generation measures distances on curved surfaces and can take longer. The report includes trimming and any unresolved spacing checks.')
            if minimum>=maximum:
                st.error('Maximum thickness must exceed minimum thickness.')
            else:adaptive=AdaptiveConfig(minimum=minimum,maximum=maximum,tolerance=min(.005,minimum/10))
        index = st.slider('Layer index', 1, count, 1) if count>1 else 1
        scalar = result['scalar']
        level = float(scalar.min()+(index-.5)*np.ptp(scalar)/count)
        points,triangles = isosurface(original,scalar.copy(),level)
        fig = figure(result['points'],faces,name='Layer context',opacity=.12)
        if len(triangles):
            fig.add_trace(figure(points,triangles,name='Layer').data[0])
        st.plotly_chart(fig,width='stretch')
        st.caption(f'Layer {index}/{count} · scalar value {level:.5g} · {len(triangles):,} triangles. Equal scalar increments need not yield equal physical thickness.')
    with tabs[2]:
        c1,c2,c3=st.columns(3)
        path_mode=c1.selectbox('Toolpath pattern',['Boundary contours','Stress hybrid'])
        spacing=c2.number_input('Path spacing (mm)',min_value=.05,value=.4,step=.05)
        sample=c3.number_input('Waypoint distance (mm)',min_value=.05,value=.4,step=.05)
        with st.expander('Toolpath settings'):
            rings=st.number_input('Hybrid boundary contours',min_value=1,max_value=10,value=2)
            max_faces=st.number_input('Maximum refined triangles per layer',min_value=1000,value=150000,step=10000)
        path_cfg=ToolpathConfig(spacing=spacing,waypoint_distance=sample,mode='hybrid' if path_mode=='Stress hybrid' else 'contour',
                                boundary_count=int(rings),max_faces=int(max_faces))
        visual_id=hashlib.sha256((saved_id+json.dumps(dict(layer_mode=layer_mode,count=count,
            adaptive=asdict(adaptive) if adaptive else None,paths=asdict(path_cfg)),sort_keys=True)).encode()).hexdigest()
        st.caption('Paths are computed directly on the original curved layers. Stress hybrid uses the stress fields attached to the saved deformation run.')
        if st.button('Generate layers and toolpaths',type='primary'):
            if layer_mode=='Adaptive thickness' and adaptive is None:
                st.error('Correct the layer thickness range before generating.')
            else:
                with st.status('Generating curved layers and toolpaths…',expanded=True) as status:
                    progress=st.empty()
                    try:
                        saved_fields=st.session_state.get('experiment_fields',{})
                        visual=build_visualization(result,count=count,adaptive=adaptive,toolpath_config=path_cfg,
                            stress=saved_fields.get('stress'),stress_mask=saved_fields.get('stress_mask'),
                            callback=lambda row:progress.write(row))
                        st.session_state['visual']=(visual_id,saved_id,*visual)
                        st.session_state.pop('export',None)
                        status.update(label='Layers and toolpaths ready',state='complete',expanded=False)
                    except Exception as error:
                        status.update(label='Path generation failed',state='error')
                        st.error(str(error))
        stored_visual=st.session_state.get('visual')
        if stored_visual and stored_visual[1]==saved_id:
            _,_,layers,paths,visual_report=stored_visual
            if stored_visual[0]!=visual_id:
                st.warning('Layer or path settings changed. The view still shows the last generated toolpaths.')
            c1,c2,c3=st.columns(3)
            c1.metric('Curved layers',len(layers));c2.metric('Deposition strokes',visual_report['total_curves'])
            c3.metric('Path length (mm)',f"{visual_report['total_length']:.1f}")
            selected=st.slider('Toolpath layer',1,len(layers),1) if len(layers)>1 else 1
            display=st.radio('Show paths',['Selected layer','Through selected layer','All layers'],horizontal=True)
            c1,c2=st.columns(2)
            show_surface=c1.checkbox('Show layer surface',True)
            show_normals=c2.checkbox('Show surface normals',False)
            st.plotly_chart(toolpath_figure(result,layers,paths,index=None if display=='All layers' else selected-1,
                cumulative=display=='Through selected layer',show_surface=show_surface,show_normals=show_normals),width='stretch')
            st.caption(f"Layer {selected}: scalar {layers[selected-1].level:.5g} · {'inserted partial layer' if layers[selected-1].inserted else 'full layer'}. Orange: contours; pink: stress interiors. Separate strokes have no implied travel connection.")
            if visual_report['empty_layer_indices']:
                st.warning('Some layers have no paths at this spacing: '+', '.join(str(i+1) for i in visual_report['empty_layer_indices']))
            if 'spacing_passed' in visual_report['layers'] and not visual_report['layers']['spacing_passed']:
                st.warning('The delivered adaptive layers do not pass every spacing bound. Inspect the layer report in Diagnostics.')
        else:
            st.write('Generate toolpaths to inspect individual layers, accumulated deposition, and surface normals.')
    with tabs[3]:
        history = report['history']
        if history:
            last = history[-1]
            c1,c2,c3 = st.columns(3)
            c1.metric('Final Π',f"{last['pi']:.6g}")
            c2.metric('Worst violation',f"{last['worst_angle_degrees']:.3f}°")
            c3.metric('Inverted cells',last['inverted_cells'])
            st.line_chart({'Π':[report['initial_pi']]+[row['pi'] for row in history]},x_label='Outer iteration')
            st.dataframe(history,width='stretch')
        else:
            st.write(f"Initial Π: {report['initial_pi']:.6g}; no outer iterations required.")
        st.json(report['config'],expanded=False)
        if report.get('build_plate'):
            st.write('Build-plate and layer-start checks')
            st.json(report['build_plate'],expanded=False)
        stored_visual=st.session_state.get('visual')
        if stored_visual and stored_visual[1]==saved_id:
            st.subheader('Layer and toolpath diagnostics')
            st.json(stored_visual[4]['layers'],expanded=False)
            st.dataframe(stored_visual[4]['toolpaths'],width='stretch')
            st.caption('Boundary distances and stress-field integration are numerical approximations. Equal field increments do not certify complete bead coverage.')
    with tabs[4]:
        stored_visual=st.session_state.get('visual')
        valid_visual=stored_visual if stored_visual and stored_visual[1]==saved_id else None
        export_id=saved_id+(valid_visual[0] if valid_visual else '')
        st.write('Export arrays, deformed mesh, layer OBJs, and the run report.'+
                 (' Includes toolpath OBJ/JSON files with waypoint normals and an offline interactive preview.html.' if valid_visual else ' Generate toolpaths to include path files and the interactive HTML preview.'))
        if st.button('Prepare ZIP'):
            with st.spinner('Building archive…'):
                st.session_state['export'] = (export_id,count,bundle(result,report,count,valid_visual[2:] if valid_visual else None))
        exported = st.session_state.get('export')
        if exported and exported[:2]==(export_id,count):
            st.download_button('Download experiment',exported[2],'s3_experiment.zip','application/zip')
        st.caption('Exports use Z-up coordinates. No G-code is produced.')


if __name__ == '__main__':
    main()
