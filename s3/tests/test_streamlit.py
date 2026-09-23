"""UI integration: a real solve, stale controls, and a readable export."""
import io
import json
from pathlib import Path
import zipfile
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


def test_run_inspect_and_export():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=30).run()
    assert not app.exception
    sidebar_download=next(b for b in app.get('download_button') if b.proto.id.endswith('-sidebar_gcode_download'))
    assert sidebar_download.proto.disabled
    next(s for s in app.selectbox if s.label=='Eq. 12 solver').set_value('Iterative (CG)')
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
    assert not app.exception
    assert 'experiment' in app.session_state
    assert app.session_state['experiment'][2]['config']['scale_solver']=='iterative'
    assert any('elapsed · Finished' in item.value for item in app.markdown)
    assert any('Outer 1/5 completed in' in item.value for item in app.markdown)
    assert any(item.label.startswith('Run finished in') for item in app.status)
    assert next(r for r in app.radio if r.label=='Layer generation').value=='Adaptive thickness'
    assert not any(s.label=='Diagnostic layer count' for s in app.slider)
    next(b for b in app.button if b.label=='Prepare ZIP').click().run()
    assert not app.exception
    payload=app.session_state['export'][2]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        report=json.loads(archive.read('report.json'))
        assert report['implementation']=='paper'
        assert report['layers']==[]
        assert not any(name.startswith('layers/') for name in archive.namelist())
        assert 'deformed.tet' in archive.namelist()
        assert not report['print_ready']
    next(s for s in app.slider if s.label=='Support-free weight').set_value(.5).run()
    assert not app.exception
    assert any('last successful run' in w.value for w in app.warning)


def test_generate_toolpaths_and_export_preview():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=30).run()
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
    next(r for r in app.radio if r.label=='Layer generation').set_value('Fixed count').run()
    next(n for n in app.number_input if n.label=='Path spacing (mm)').set_value(1.).run()
    next(b for b in app.button if b.label=='Generate layers and toolpaths').click().run()
    assert not app.exception
    assert 'visual' in app.session_state
    visual=app.session_state['visual']
    assert len(visual[2])==12 and visual[4]['total_curves']>0
    next(s for s in app.slider if s.label=='Toolpath layer').set_value(5).run()
    next(c for c in app.checkbox if c.label=='Show surface normals').check().run()
    assert not app.exception
    next(b for b in app.button if b.label=='Prepare ZIP').click().run()
    assert not app.exception
    with zipfile.ZipFile(io.BytesIO(app.session_state['export'][2])) as archive:
        assert 'preview.html' in archive.namelist()
        paths=json.loads(archive.read('toolpaths/0000.json'))
        assert paths['paths']
        assert len(paths['paths'][0]['points'])==len(paths['paths'][0]['normals'])
        assert 'toolpaths/0000.obj' in archive.namelist()
        assert 'visualization.json' in archive.namelist()
    next(n for n in app.number_input if n.label=='Path spacing (mm)').set_value(1.5).run()
    assert any('last generated toolpaths' in w.value for w in app.warning)


def test_cura_download_without_research_toolpaths():
    if not Path('/usr/bin/CuraEngine').exists():pytest.skip('CuraEngine not installed')
    pytest.importorskip('pyvista')
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=60).run()
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
    next(n for n in app.number_input if n.label=='Layer height (mm)').set_value(.5).run()
    next(b for b in app.button if b.label=='Slice with Cura').click().run()
    assert not app.exception and not app.error
    assert 'visual' not in app.session_state
    exported=app.session_state['cura_export']
    assert exported[3]['layers']>12
    assert 'G93' in exported[2]
    assert {'print.gcode','cura_deformed.gcode','cura.log','cura_deformed.stl','cura_report.json'}<=exported[4].keys()
    sidebar_download=next(b for b in app.get('download_button') if b.proto.id.endswith('-sidebar_gcode_download'))
    assert not sidebar_download.proto.disabled
    assert app.session_state['ready_gcode_download']==exported[2]
    next(n for n in app.number_input if n.label=='Line width (mm)').set_value(.5).run()
    assert app.session_state['ready_gcode_download'] is None
    assert any('Cura settings changed' in i.value for i in app.info)


def test_placement_applies_to_run_and_reset_marks_it_stale():
    import numpy as np
    from s3.mesh import read_tet
    mesh=read_tet(Path(__file__).parent/'fixtures/cantilever.tet')
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=30).run()
    next(n for n in app.number_input if n.label=='Model scale (%)').set_value(50.)
    next(n for n in app.number_input if n.label=='Rotate Z (degrees)').set_value(90.)
    next(c for c in app.checkbox if c.label=='Center XY on rotation axis').check()
    next(n for n in app.number_input if n.label=='X placement offset (mm)').set_value(30.)
    app.run()
    assert not app.exception
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
    assert not app.exception
    saved_id,result,report=app.session_state['experiment']
    np.testing.assert_allclose(np.ptp(result['points'],axis=0),
                               np.ptp(mesh.points,axis=0)[[1,0,2]]*.5,atol=1e-10)
    assert report['placement']['scale']==.5
    assert report['placement']['rotation_z']==90
    next(b for b in app.button if b.label=='Reset placement').click().run()
    assert not app.exception
    assert next(n for n in app.number_input if n.label=='Model scale (%)').value==100.
    assert any('last successful run' in w.value for w in app.warning)
    assert app.session_state['experiment'][0]==saved_id


def test_adaptive_preview_uses_generated_surface(monkeypatch):
    import numpy as np
    import s3.visual_pipeline
    from s3.surface import Surface
    from s3.toolpaths import Toolpath
    layer=Surface([[0,0,.123],[2,0,.123],[0,2,.123]],[[0,1,2]],level=.017)
    curve=Toolpath(np.array([[.2,.2,.123],[1.,.2,.123]]),
                  np.array([[0.,0.,1.],[0.,0.,1.]]),'contour',False,.017)
    def generated(*args,**kwargs):
        assert kwargs['adaptive'].first_layer_height==.2
        return [layer],[[curve]],dict(layers=dict(mode='adaptive'),toolpaths=[],
            total_curves=1,total_length=.8,empty_layer_indices=[])
    monkeypatch.setattr(s3.visual_pipeline,'build_visualization',generated)
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=30).run()
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
    next(r for r in app.radio if r.label=='Layer generation').set_value('Adaptive thickness').run()
    assert not any('physical Z' in c.value for c in app.caption)
    next(b for b in app.button if b.label=='Generate layers and toolpaths').click().run()
    assert not app.exception
    assert any('physical Z 0.1230–0.1230' in c.value for c in app.caption)
    next(n for n in app.number_input if n.label=='First-layer height (mm)').set_value(.15).run()
    assert not any('physical Z' in c.value for c in app.caption)
