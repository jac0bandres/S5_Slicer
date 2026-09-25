"""UI reruns and placement changes must not repeat volume meshing."""
import numpy as np
import pytest

pytest.importorskip('streamlit')
pytest.importorskip('plotly')
from s3 import streamlit_app as app
from s3.mesh import TetMesh
from s3.placement import Placement


def test_mesh_cache_reuses_meshing_and_placement(monkeypatch):
    calls = []
    placements = []
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def tetrahedralize(payload, kind, repair, notes):
        calls.append((payload, kind, repair))
        notes.append('Meshed once')
        return TetMesh(points, np.array([[0, 1, 2, 3]]))

    original_apply = Placement.apply

    def apply(self, mesh):
        placements.append(self)
        return original_apply(self, mesh)

    monkeypatch.setattr(app, 'tetrahedralize', tetrahedralize)
    monkeypatch.setattr(Placement, 'apply', apply)
    app.load_mesh.clear()
    app.placed_mesh.clear()
    try:
        first, _ = app.placed_mesh(b'mesh-a', 'Z', 'stl', True, Placement())
        # Simulate a rerun after editing an unrelated widget, and ensure caller
        # mutation cannot corrupt cached geometry.
        first.points[:] = 99
        again, notes = app.placed_mesh(b'mesh-a', 'Z', 'stl', True, Placement())
        np.testing.assert_allclose(again.points, points)
        assert notes == ['Meshed once']
        assert len(calls) == len(placements) == 1

        moved, _ = app.placed_mesh(b'mesh-a', 'Z', 'stl', True, Placement(offset_x=5))
        np.testing.assert_allclose(moved.points, points + [5, 0, 0])
        resized, _ = app.placed_mesh(b'mesh-a', 'Z', 'stl', True,
                                     Placement(scale=2, rotation_z=90))
        np.testing.assert_allclose(resized.points[:2], [[1, 0, 0], [1, 2, 0]], atol=1e-12)
        y_up, _ = app.placed_mesh(b'mesh-a', 'Y', 'stl', True, Placement())
        np.testing.assert_allclose(y_up.points[2], [0, 0, 1])
        np.testing.assert_allclose(y_up.points[3], [0, -1, 0])
        np.testing.assert_allclose(app.source_axis_directions('Y', Placement()),
                                   [[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        assert len(calls) == 1
        assert len(placements) == 4

        app.placed_mesh(b'mesh-b', 'Z', 'stl', True, Placement())
        app.placed_mesh(b'mesh-b', 'Z', 'stl', False, Placement())
        assert len(calls) == 3
    finally:
        app.load_mesh.clear()
        app.placed_mesh.clear()


def test_run_button_executes_after_cached_reruns(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from s3 import pipeline, mesh as mesh_module

    runs = []
    reads = []
    original_run = pipeline.run_paper
    original_read = mesh_module.read_tet

    def run(*args, **kwargs):
        runs.append(True)
        return original_run(*args, **kwargs)

    def read(*args, **kwargs):
        reads.append(True)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pipeline, 'run_paper', run)
    monkeypatch.setattr(mesh_module, 'read_tet', read)
    app.load_mesh.clear()
    app.placed_mesh.clear()
    try:
        ui = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
        assert not ui.exception
        assert any('Placed size: X 30.000 × Y 10.000 × Z 30.000' in c.value
                   for c in ui.caption)
        next(s for s in ui.selectbox if s.label == 'Input up axis').set_value('Y').run()
        assert not ui.exception
        assert any('Placed size: X 30.000 × Y 30.000 × Z 10.000' in c.value
                   for c in ui.caption)
        assert any('Y-up file converted to Z-up' in c.value for c in ui.caption)
        # An ordinary widget edit reruns the app without running the solver.
        next(w for w in ui.number_input if w.label == 'Inner iterations').set_value(1).run()
        assert not runs
        for expected in (1, 2):
            ui.button(key='run_paper_pipeline').click().run()
            assert not ui.exception
            assert not ui.error
            assert len(runs) == expected
            assert 'experiment' in ui.session_state
            assert any(s.state == 'complete' for s in ui.status)
        assert len(reads) == 1
        assert [tab.label for tab in ui.tabs] == ['Geometry', 'Cura slicing', 'Diagnostics', 'Download']
        def rerun_on_cura_tab():
            # AppTest does not yet serialize stateful tab widgets. Supply the
            # selected label exactly as the browser does on each rerun.
            states = ui._tree.get_widget_states()
            container = ui.get('tab_container')[0].proto.tab_container
            states.widgets.add(id=container.id, string_value='Cura slicing')
            ui._run(states)
            assert ui.get('tab_container')[0].proto.tab_container.default_tab_index == 1

        rerun_on_cura_tab()
        next(w for w in ui.number_input if w.label == 'Layer height (mm)').set_value(.25)
        rerun_on_cura_tab()
        assert not ui.exception
        assert ui.session_state['workbench_view'] == 'Cura slicing'
        next(w for w in ui.number_input if w.label == 'Infill density (%)').set_value(30.)
        rerun_on_cura_tab()
        assert not ui.exception
        assert ui.session_state['workbench_view'] == 'Cura slicing'
        assert next(w for w in ui.number_input if w.label == 'Layer height (mm)').value == .25
        next(b for b in ui.button if b.label == 'Prepare ZIP').click().run()
        assert not ui.exception
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(ui.session_state['export'][1])) as archive:
            assert set(archive.namelist()) == {'result.npz', 'deformed.tet', 'report.json'}
    finally:
        app.load_mesh.clear()
        app.placed_mesh.clear()
