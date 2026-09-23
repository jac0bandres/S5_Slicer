"""Broken surfaces are repaired automatically without losing geometry."""
import numpy as np
import pytest
pytest.importorskip('trimesh')
pytest.importorskip('tetgen')
import trimesh
from s3.stl_import import read_surface, repair_surface, tetrahedralize, _covers


def stl(mesh):
    return mesh.export(file_type='stl')


def open_box():
    """A cube missing its top face — watertight only after hole filling."""
    box = trimesh.creation.box(extents=(10, 10, 10))
    faces = np.asarray(box.faces)
    keep = [i for i, f in enumerate(faces) if not np.all(box.vertices[f][:, 2] > 0)]
    return trimesh.Trimesh(box.vertices, faces[keep], process=False)


def overlapping_boxes():
    """Two interpenetrating cubes — a self-intersecting, two-component soup."""
    a = trimesh.creation.box(extents=(10, 10, 10))
    b = trimesh.creation.box(extents=(10, 10, 10))
    b.apply_translation([4, 4, 4])
    return trimesh.util.concatenate([a, b])


def test_clean_surface_skips_repair_entirely():
    notes = []
    mesh = tetrahedralize(stl(trimesh.creation.icosphere(subdivisions=2, radius=10)),
                          'stl', notes=notes)
    assert len(mesh.cells) > 100
    assert notes == [f'Volume mesh from TetGen (as-is): {len(mesh.cells)} tetrahedra.']


def test_open_surface_is_closed_and_tetrahedralized():
    pytest.importorskip('pymeshfix')
    notes = []
    mesh = tetrahedralize(stl(open_box()), 'stl', notes=notes)
    # The repaired cube is a closed 10 mm box, not the open shell TetGen refused.
    assert np.isclose(mesh.volumes.sum(), 1000., rtol=.02)
    assert any('as-is) failed' in note for note in notes)
    assert any('TetGen (repaired)' in note for note in notes)


def test_repair_disabled_reports_the_tetgen_failure():
    with pytest.raises(RuntimeError) as failure:
        tetrahedralize(stl(open_box()), 'stl', repair=False)
    assert 'even after automatic repair' not in str(failure.value)
    assert 'TetGen (as-is) failed' in str(failure.value)


def test_meshfix_dropping_a_component_falls_through_to_ftetwild():
    pytest.importorskip('pymeshfix')
    pytest.importorskip('wildmeshing')
    notes = []
    mesh = tetrahedralize(stl(overlapping_boxes()), 'stl', notes=notes)
    assert any('no longer spans the input bounding box' in note for note in notes)
    assert any('FloatTetWild' in note and 'tetrahedra' in note for note in notes)
    # Both cubes survive: the union spans [-5, 9] and is larger than one cube.
    assert np.allclose(mesh.points.min(axis=0), -5, atol=.2)
    assert np.allclose(mesh.points.max(axis=0), 9, atol=.2)
    assert mesh.volumes.sum() > 1200.


def test_meshfix_keeps_every_component_of_a_valid_multipart_model():
    pytest.importorskip('pymeshfix')
    a = trimesh.creation.box(extents=(10, 10, 10))
    b = trimesh.creation.box(extents=(4, 4, 4))
    b.apply_translation([20, 0, 0])
    points, faces = read_surface(stl(trimesh.util.concatenate([a, b])), 'stl')
    repaired, _faces = repair_surface(points, faces)
    assert _covers(repaired, points)


def test_empty_and_collapsed_inputs_are_rejected():
    with pytest.raises(ValueError):
        read_surface(stl(trimesh.Trimesh()), 'stl')
    degenerate = trimesh.Trimesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), process=False)
    with pytest.raises(ValueError):
        read_surface(stl(degenerate), 'stl')


def test_covers_tolerates_small_shrinkage_but_not_a_missing_half():
    points = np.array([[0., 0, 0], [10, 10, 10]])
    assert _covers(points * np.array([[1.], [.999]]), points)
    assert not _covers(points * np.array([[1.], [.5]]), points)
