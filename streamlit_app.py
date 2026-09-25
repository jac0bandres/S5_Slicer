"""Launch from the repository: streamlit run streamlit_app.py."""
from pathlib import Path
import sys
import io
import json
import hashlib
import tempfile
import time
import zipfile
from dataclasses import asdict, replace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from s3.mesh import TetMesh, read_tet, write_tet
from s3.placement import Placement
from s3.pipeline import PaperConfig, run_paper
from s3.cura import CuraConfig, slice_with_cura
from s3.gcode_preview import gcode_motion_preview
from s3.visualization import deformation_figure
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


def source_axis_directions(up_axis, placement):
    """Directions of the file's XYZ axes in the current placement preview."""
    axes = np.eye(3)
    if up_axis == 'Y':
        axes = axes @ np.array([[1,0,0],[0,0,1],[0,-1,0]])
    return axes @ placement.rotation.T


@st.cache_data(show_spinner='Preparing volume mesh…', max_entries=8)
def load_mesh(payload, kind='tet', repair=True):
    """Cache meshing by file content and repair settings, before orientation."""
    notes = []
    if kind == 'tet':
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.tet'
            path.write_bytes(payload)
            mesh = read_tet(path)
    else:
        mesh = tetrahedralize(payload, kind, repair=repair, notes=notes)
    return mesh, notes


@st.cache_data(show_spinner=False, max_entries=8)
def placed_mesh(payload, up_axis, kind, repair, placement):
    """Reuse the volume mesh across placement changes and UI reruns."""
    mesh, notes = load_mesh(payload, kind, repair)
    return placement.apply(apply_up_axis(mesh, up_axis)), notes


@st.cache_data(show_spinner=False, max_entries=8)
def tet_download(points, cells):
    """Serialize the undeformed mesh using the same format as .tet uploads."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'mesh.tet'
        write_tet(path, points, cells)
        return path.read_bytes()


def bundle(result, report):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        np.savez_compressed(root/'result.npz', **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        write_tet(root/'deformed.tet', result['deformed'], result['cells'])
        (root/'report.json').write_text(json.dumps(report, indent=2))
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file(): archive.write(path, path.relative_to(root))
        return output.getvalue()


def reset_placement():
    for name,value in dict(scale=100.,rotation_x=0.,rotation_y=0.,rotation_z=0.,
                           center_xy=False,drop_to_bed=True,offset_x=0.,offset_y=0.).items():
        st.session_state['placement_'+name]=value


def main():
    st.set_page_config(page_title='S³ Research Workbench', page_icon='🧊', layout='wide')
    gcode_sidebar_slot=st.sidebar.container()
    st.session_state['ready_gcode_download']=None
    st.session_state['gcode_download_name']='s3_print.gcode'
    st.session_state['gcode_download_status']='Run the paper pipeline, then click Slice with Cura in the Cura slicing tab.'
    st.title('S³ Research Workbench')
    st.caption('S3 deformation • Cura slicing • S4/S5 G-code reformation')
    st.info('Run the S3 deformation, then use Cura slicing to generate walls, infill and G-code.')
    with st.sidebar:
        st.header('Mesh')
        examples = {'Cantilever · quick test': ROOT/'tests/fixtures/cantilever.tet'}
        official = ROOT.parent/'S3_DeformFDM/DataSet/TET_MODEL'
        examples.update({f'Official · {p.stem}': p for p in sorted(official.glob('*.tet'))})
        source = st.selectbox('Source', list(examples)+['Upload .tet','Upload STL / surface'])
        upload = surface = None
        if source=='Upload .tet':
            upload = st.file_uploader('Tetrahedral mesh', type=['tet'])
        elif source=='Upload STL / surface':
            surface = st.file_uploader('Surface mesh (tetrahedralized on upload)', type=list(SUFFIXES),
                help='STL/OBJ/PLY/OFF surfaces are volume-meshed with TetGen. Holes, flipped normals and '
                     'self-intersections are repaired automatically.')
        repair_surfaces = st.checkbox('Automatic mesh repair', True, disabled=source!='Upload STL / surface',
            help='Close holes and remove self-intersections (MeshFix), then fall back to FloatTetWild when '
                 'TetGen still cannot mesh the surface. Off: TetGen on the uploaded surface only.')
        axis = st.selectbox('Input up axis', ['Z','Y'], index=1 if source.startswith('Official') else 0,
                            help='Official S³ datasets use Y-up. All displayed results use Z-up.')
        with st.expander('Scale, rotate and place',expanded=True):
            scale=st.number_input('Model scale (%)',min_value=.01,value=100.,step=10.,key='placement_scale')
            rotation_x=st.number_input('Rotate X (degrees)',value=0.,step=90.,key='placement_rotation_x')
            rotation_y=st.number_input('Rotate Y (degrees)',value=0.,step=90.,key='placement_rotation_y')
            rotation_z=st.number_input('Rotate Z (degrees)',value=0.,step=90.,key='placement_rotation_z')
            center_xy=st.checkbox('Center XY on rotation axis',False,key='placement_center_xy')
            drop_to_bed=st.checkbox('Drop model to bed',True,key='placement_drop_to_bed')
            offset_x=st.number_input('X placement offset (mm)',value=0.,step=5.,key='placement_offset_x')
            offset_y=st.number_input('Y placement offset (mm)',value=0.,step=5.,key='placement_offset_y')
            st.button('Reset placement',on_click=reset_placement)
            st.caption('Uniform scale; rotations about the model center, applied X → Y → Z after up-axis conversion. Offsets apply after centering. Rerun slicing after changes.')
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
                help='0 anchors the lowest contact patch, or the lowest boundary triangle if the bottom is curved or tilted. '
                     'Raise it to pin a thin slab of the lowest surface instead.')
        st.header('Run budget')
        scale_solver=st.selectbox('Eq. 12 solver',['Automatic','Iterative (CG)','Direct (LU)'],
            help='Automatic uses iterative CG for at least 10,000 free unknowns. CG avoids LU factorization and reports residual progress. Direct LU remains available for comparison.')
        outer = st.number_input('Maximum outer iterations', 1, 100, 5)
        inner = st.number_input('Inner iterations', 1, 50, 7)
        tolerance = st.number_input('Relative stopping tolerance', .0001, .99, .05, format='%.4f')
        st.caption('Relative stagnation is not objective satisfaction. Large meshes can take several minutes per run.')
        has_input = source in examples or upload is not None or surface is not None
        run = st.button('Run paper pipeline', type='primary', width='stretch', key='run_paper_pipeline',
                        disabled=not has_input)
        run_feedback = st.empty()
        if run:
            run_feedback.info('Starting pipeline — preparing the cached mesh and inputs…')
    try:
        if surface is not None:
            payload, kind = surface.getvalue(), Path(surface.name).suffix.lstrip('.').lower()
            input_name = surface.name
        elif upload is not None:
            payload, kind = upload.getvalue(), 'tet'
            input_name = upload.name
        elif source in examples:
            payload, kind = examples[source].read_bytes(), 'tet'
            input_name = examples[source].name
        else:
            payload = kind = None
        if payload is None:
            st.info('Upload a .tet mesh or an STL/OBJ/PLY/OFF surface to begin. '
                    'Surfaces are tetrahedralized with TetGen on upload.')
            return gcode_sidebar_slot
        placement=Placement(scale=scale/100,rotation_x=rotation_x,rotation_y=rotation_y,
            rotation_z=rotation_z,center_xy=center_xy,drop_to_bed=drop_to_bed,
            offset_x=offset_x,offset_y=offset_y)
        mesh, mesh_notes = placed_mesh(payload, axis, kind, repair_surfaces, placement)
        for note in mesh_notes:
            st.caption(note)
        fields = {}
        field_bytes = objective_upload.getvalue() if objective_upload else b''
        if field_bytes:
            with np.load(io.BytesIO(field_bytes), allow_pickle=False) as data:
                if set(data.files)-{'stress','stress_mask','sf_faces','sq_faces'}:
                    raise ValueError('Unknown objective arrays; use stress, stress_mask, sf_faces, sq_faces.')
                fields = {k:data[k] for k in data.files}
            if axis=='Y' and 'stress' in fields:
                fields['stress'] = fields['stress'] @ np.array([[1,0,0],[0,0,1],[0,-1,0]])
        if 'stress' in fields:
            fields['stress']=placement.rotate_vectors(fields['stress'])
        cfg = PaperConfig(alpha=alpha,beta=beta,gamma=gamma,weight_sf=sf,weight_sr=sr,weight_sq=sq,
                          rigidity=rigidity,scale_compatibility=compatibility,concavity_control=concavity,
                          exclude_build_plate=plate,max_outer_iterations=int(outer),inner_iterations=int(inner),
                          fix_build_plate=fix_plate,base_band=base_band,
                          relative_tolerance=tolerance,
                          scale_solver={'Automatic':'auto','Iterative (CG)':'iterative','Direct (LU)':'direct'}[scale_solver])
    except Exception as error:
        st.error(f'Input error: {error}')
        if run:
            run_feedback.error('Pipeline could not start. See the input error in the main panel.')
        return gcode_sidebar_slot
    fingerprint = hashlib.sha256(payload+field_bytes+axis.encode()+json.dumps(dict(config=asdict(cfg),placement=asdict(placement)),sort_keys=True).encode()).hexdigest()
    st.subheader('Live input preview')
    st.caption(f'Current input · {axis}-up file converted to Z-up · scale {scale:g}% · '
               f'rotation X {rotation_x:g}°, Y {rotation_y:g}°, Z {rotation_z:g}°')
    a,b = st.columns(2)
    a.metric('Vertices', f'{len(mesh.points):,}')
    b.metric('Tetrahedra', f'{len(mesh.cells):,}')
    size=np.ptp(mesh.points,axis=0)
    st.caption(f'Placed size: X {size[0]:.3f} × Y {size[1]:.3f} × Z {size[2]:.3f} mm')
    placement_fig=figure(mesh.points,mesh.faces[mesh.boundary],name='Current input')
    span=max(float(size.max()),1.)
    lo=np.minimum(mesh.points[:,:2].min(axis=0),0)-span*.1
    hi=np.maximum(mesh.points[:,:2].max(axis=0),0)+span*.1
    placement_fig.add_trace(go.Mesh3d(x=[lo[0],hi[0],hi[0],lo[0]],
        y=[lo[1],lo[1],hi[1],hi[1]],z=[0,0,0,0],i=[0,0],j=[1,2],k=[2,3],
        color='#94a3b8',opacity=.2,name='Z=0 reference',showlegend=True))
    guide_origin=mesh.points.min(axis=0)-span*.15
    for label,color,direction in zip(('X','Y','Z'),('#ef4444','#22c55e','#3b82f6'),
                                     source_axis_directions(axis,placement)):
        tip=guide_origin+direction*span*.18
        placement_fig.add_trace(go.Scatter3d(
            x=[guide_origin[0],tip[0]],y=[guide_origin[1],tip[1]],
            z=[guide_origin[2],tip[2]],mode='lines+text',text=['',f'File {label}'],
            textposition='top center',line=dict(color=color,width=5),
            name=f'File {label} axis'))
    st.plotly_chart(placement_fig,width='stretch')
    st.caption('This preview updates when input and placement controls change. The colored guides show the file’s XYZ axes after conversion and rotation.')
    st.download_button('Download tetrahedral mesh (.tet)',tet_download(mesh.points,mesh.cells),
                       file_name=Path(input_name).stem+'.tet',mime='text/plain',
                       key='input_tet_download',on_click='ignore')
    if run:
        if sr>0 and not {'stress','stress_mask'} <= fields.keys():
            st.error('Strength optimization needs both stress and stress_mask in the objective NPZ.')
            run_feedback.error('Pipeline needs strength fields. See the error in the main panel.')
        else:
            started = time.perf_counter()
            run_feedback.info('Pipeline running. Progress is shown at the top of the main panel.')
            with st.status('Solving paper equations…', expanded=True) as status:
                progress = st.empty()
                st.caption('Stage times update as work advances. Sparse solves can take several minutes on large meshes.')
                iteration_log = st.container()
                def show_progress(event):
                    progress.write(f"{event['elapsed_seconds']:.1f}s elapsed · {event['message']}")
                def show_iteration(row):
                    iteration_log.write(
                        f"Outer {row['outer']}/{outer} completed in {row['seconds']:.1f}s · "
                        f"Π {row['pi']:.5g} · worst violation {row['worst_angle_degrees']:.2f}° · "
                        f"relative change {row['relative_change']:.2%} · "
                        f"step {row['step_fraction']:.5g} · inverted cells {row['inverted_cells']}")
                try:
                    result = run_paper(mesh,cfg,**fields,callback=show_iteration,progress_callback=show_progress)
                    report = dict(implementation='paper',config=asdict(cfg),placement=asdict(placement),input_up_axis=axis,output_up_axis='Z',
                                  input_name=input_name,
                                  mesh_sha256=hashlib.sha256(payload).hexdigest(),
                                  objectives_sha256=hashlib.sha256(field_bytes).hexdigest() if field_bytes else None,
                                  history=result['history'],initial_pi=result['initial_pi'],stop_reason=result['stop_reason'],
                                  build_plate=result['build_plate'],
                                  seconds=time.perf_counter()-started,print_ready=False)
                    st.session_state['experiment'] = (fingerprint,result,report)
                    st.session_state['experiment_fields'] = fields
                    st.session_state.pop('export',None)
                    st.session_state.pop('visual',None)
                    st.session_state.pop('cura_export',None)
                    run_feedback.success('Pipeline finished. Results are below.')
                    status.update(label=f"Run finished in {report['seconds']:.1f}s · {result['stop_reason']}",
                                  state='complete',expanded=True)
                except Exception as error:
                    run_feedback.error('Pipeline failed. See the error in the main panel.')
                    status.update(label='Run failed',state='error')
                    st.error(str(error))
                    st.caption('Some combinations of boundary-face objectives are infeasible; the solver reports them instead of relaxing constraints silently.')
    saved = st.session_state.get('experiment')
    if not saved:
        return gcode_sidebar_slot
    st.session_state['gcode_download_status']='Click Slice with Cura in the Cura slicing tab.'
    saved_id,result,report = saved
    model_stem = Path(report.get('input_name', input_name if saved_id == fingerprint else 's3_print')).stem
    gcode_name = model_stem + '.gcode'
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
    tabs = st.tabs(['Geometry','Cura slicing','Diagnostics','Download'],
                   key='workbench_view',on_change='rerun')
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
        st.subheader('Cura slicing and S4/S5 reformation')
        st.write('Cura slices the deformed solid. Its walls, infill, top/bottom layers, travels and extrusion are mapped back into the original model through the tetrahedra.')
        c1,c2,c3=st.columns(3)
        cura_height=c1.number_input('Layer height (mm)',min_value=.01,value=.2,step=.01)
        cura_first=c2.number_input('Cura first-layer height (mm)',min_value=.01,value=.2,step=.01)
        cura_width=c3.number_input('Line width (mm)',min_value=.01,value=.4,step=.01)
        c1,c2,c3=st.columns(3)
        cura_infill=c1.number_input('Infill density (%)',min_value=0.,max_value=100.,value=20.,step=5.)
        cura_walls=c2.number_input('Wall count',min_value=1,value=2,step=1)
        cura_speed=c3.number_input('Cura print speed (mm/s)',min_value=.1,value=30.)
        c1,c2=st.columns(2)
        brim_width=c1.number_input('Physical bed brim width (mm)',min_value=0.,value=0.,step=.4,
            help='Adds a flat brim around the reformed first-layer outer walls after Cura slicing. Zero disables it.')
        brim_gap=c2.number_input('Brim gap (mm)',min_value=0.,value=0.,step=.05,
            help='Distance from the reformed first-layer footprint to the innermost brim line.')
        defaults=CuraConfig()
        with st.expander('Cura profile and machine settings'):
            engine=st.text_input('CuraEngine executable',defaults.engine)
            printer=st.text_input('Printer definition',defaults.printer_definition)
            extruder=st.text_input('Extruder definition',defaults.extruder_definition)
            profile=st.text_input('Cura settings profile',defaults.profile)
            profile_upload=st.file_uploader('Upload Cura settings JSON',type=['json'])
            diameter=st.number_input('Filament diameter (mm)',min_value=.1,value=1.75)
            nozzle=st.number_input('Nozzle temperature (°C)',min_value=0.,value=240.)
            bed=st.number_input('Bed temperature (°C)',min_value=0.,value=55.)
            segment=st.number_input('Reform segment length (mm)',min_value=.05,value=.6,step=.05)
            nozzle_offset=st.number_input('Nozzle offset (mm)',min_value=0.,value=41.5)
            z_hop=st.number_input('Compensated hop height (mm)',min_value=0.,value=defaults.z_hop,step=.1)
            travel_speed=st.number_input('Travel linear axis limit (mm/s)',min_value=.1,value=defaults.travel_speed)
            rotary_speed=st.number_input('Travel rotary axis limit (°/s)',min_value=.1,value=defaults.rotary_speed)
        cura_cfg=CuraConfig(engine=engine,printer_definition=printer,extruder_definition=extruder,profile=profile,
            layer_height=cura_height,first_layer_height=cura_first,line_width=cura_width,
            infill_density=cura_infill,wall_count=int(cura_walls),print_speed=cura_speed,
            filament_diameter=diameter,nozzle_temperature=nozzle,bed_temperature=bed,
            segment_length=segment,nozzle_offset=nozzle_offset,z_hop=z_hop,
            travel_speed=travel_speed,rotary_speed=rotary_speed,
            brim_width=brim_width,brim_gap=brim_gap)
        profile_data=profile_upload.getvalue() if profile_upload else b''
        profile_signature=profile_data
        if not profile_data and profile and Path(profile).is_file():
            profile_signature=Path(profile).read_bytes()
        cura_id=hashlib.sha256(saved_id.encode()+json.dumps(asdict(cura_cfg),sort_keys=True).encode()+profile_signature).hexdigest()
        if st.button('Slice with Cura',type='primary'):
            with st.status('Cura deform/reform…',expanded=True) as status:
                with tempfile.TemporaryDirectory() as folder:
                    directory=Path(folder)
                    try:
                        run_cfg=cura_cfg
                        if profile_data:
                            (directory/'profile.json').write_bytes(profile_data)
                            run_cfg=replace(cura_cfg,profile=str(directory/'profile.json'))
                        text,cura_report=slice_with_cura(result,directory,run_cfg,callback=st.write)
                        files={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
                        st.session_state['cura_export']=(cura_id,saved_id,text,cura_report,files)
                        st.session_state.pop('cura_failure_log',None)
                        status.update(label='Cura G-code ready to download',state='complete',expanded=False)
                    except Exception as exc:
                        st.session_state.pop('cura_export',None)
                        log=directory/'cura.log'
                        if log.exists():st.session_state['cura_failure_log']=log.read_text()
                        status.update(label='Cura slicing failed',state='error')
                        st.error(str(exc))
        cura_saved=st.session_state.get('cura_export')
        current_cura=cura_saved if cura_saved and cura_saved[0]==cura_id else None
        if cura_saved and not current_cura:
            st.info('Cura settings changed. Click Slice with Cura to update the G-code.')
        if current_cura:
            st.session_state['ready_gcode_download']=current_cura[2]
            st.session_state['gcode_download_name']=gcode_name
            st.session_state['gcode_download_status']='Ready: Cura toolpaths reformed to S5 X/Z/B/C.'
            c1,c2=st.columns(2)
            c1.metric('Cura layers',current_cura[3]['layers'])
            c2.metric('Reformed moves',f"{current_cura[3]['moves']:,}")
            st.download_button('Download G-code',current_cura[2],gcode_name,'text/plain',key='cura_tab_download',type='primary')
            if st.checkbox('Show Cura G-code motion preview',False):
                motion=gcode_motion_preview(current_cura[2])
                st.plotly_chart(motion,width='stretch')
                st.download_button('Download G-code preview HTML',motion.to_html(include_plotlyjs=True),'s3_cura_preview.html','text/html')
            st.json(current_cura[3],expanded=False)
        if st.session_state.get('cura_failure_log'):
            st.download_button('Download Cura error log',st.session_state['cura_failure_log'],'cura.log','text/plain')
    with tabs[2]:
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
    with tabs[3]:
        st.write('Export arrays, deformed mesh, and the run report.')
        if st.button('Prepare ZIP'):
            with st.spinner('Building archive…'):
                st.session_state['export'] = (saved_id,bundle(result,report))
        exported = st.session_state.get('export')
        if exported and len(exported)==2 and exported[0]==saved_id:
            st.download_button('Download experiment',exported[1],'s3_experiment.zip','application/zip')
        st.caption('Exports use Z-up coordinates.')
        st.subheader('Cura G-code')
        if current_cura:
            st.download_button('Download G-code',current_cura[2],gcode_name,'text/plain',key='cura_download_tab',type='primary')
            st.download_button('Download original Cura G-code',current_cura[4]['cura_deformed.gcode'],
                               model_stem+'_cura_deformed.gcode','text/plain')
            archive=io.BytesIO()
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for name,payload in current_cura[4].items():
                    download_name = gcode_name if name == 'print.gcode' else model_stem+'_cura_deformed.gcode' if name == 'cura_deformed.gcode' else name
                    z.writestr(download_name,payload)
            st.download_button('Download Cura project ZIP',archive.getvalue(),'s3_cura.zip','application/zip')
        else:
            st.info('Open Cura slicing and click Slice with Cura.')
    return gcode_sidebar_slot


def gcode_download_panel(slot):
    with slot:
        st.header('G-code download')
        payload=st.session_state.get('ready_gcode_download')
        st.caption(st.session_state.get('gcode_download_status','Click Slice with Cura to create G-code.'))
        st.download_button('Download G-code',payload or '',file_name=st.session_state.get('gcode_download_name','s3_print.gcode'),
                           mime='text/plain',disabled=payload is None,
                           key='sidebar_gcode_download',type='primary',width='stretch')


if __name__ == '__main__':
    gcode_download_panel(main())
