import numpy as np
import pytest
pytest.importorskip('plotly')
from s3.gcode import render_gcode,GCodeConfig
from s3.gcode_preview import compatibility_preview, gcode_motion_preview
from s3.toolpaths import Toolpath


def test_unreachable_paths_still_have_preview():
    curve=Toolpath(np.array([[10.,0.,1.],[11.,0.,1.]]),
                   np.array([[0.,1.,1.],[0.,0.,1.]]),'contour',False,1.)
    fig,report=compatibility_preview([[curve]])
    assert report['waypoints']==2 and report['failed_waypoints']==1
    assert report['failures'][0]['waypoint']==1
    assert len(fig.data)==2 and list(fig.data[1].x)==[10.]
    assert 'plotly' in fig.to_html()


def test_preview_reconstructs_polar_positions_and_travel():
    curve=Toolpath(np.array([[0.,10.,1.],[0.,11.,1.]]),
                   np.array([[0.,0.,1.],[0.,0.,1.]]),'contour',False,1.)
    text,_=render_gcode([[curve]],GCodeConfig(layer_height=1.))
    fig=gcode_motion_preview(text)
    deposition=fig.data[0]
    np.testing.assert_allclose(list(deposition.x)[:2],[0.,0.],atol=1e-10)
    np.testing.assert_allclose(list(deposition.y)[:2],[10.,11.])
    assert len(fig.data[1].x)>0
