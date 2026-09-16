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
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
    assert not app.exception
    assert 'experiment' in app.session_state
    next(b for b in app.button if b.label=='Prepare ZIP').click().run()
    assert not app.exception
    payload=app.session_state['export'][2]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        report=json.loads(archive.read('report.json'))
        assert report['implementation']=='paper'
        assert len(report['layers'])==12
        assert 'deformed.tet' in archive.namelist()
        assert not report['print_ready']
    next(s for s in app.slider if s.label=='Support-free weight').set_value(.5).run()
    assert not app.exception
    assert any('last successful run' in w.value for w in app.warning)


def test_generate_toolpaths_and_export_preview():
    app=AppTest.from_file(str(Path(__file__).parents[1]/'streamlit_app.py'),default_timeout=30).run()
    next(b for b in app.button if b.label=='Run paper pipeline').click().run()
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
