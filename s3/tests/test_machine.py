import numpy as np
import pytest
from s3.machine import S5Machine


def test_s5_pose_round_trip():
    machine=S5Machine()
    for b in [-120.,-45.,0.,25.]:
        for c in [-179.,40.,390.]:
            pose=np.array([100.,30.,b,c])
            position,direction=machine.forward(pose)
            np.testing.assert_allclose(machine.inverse(position,direction,previous_c=c),pose,atol=1e-12)


def test_s5_rejects_tangential_and_out_of_range_directions():
    machine=S5Machine()
    for direction in [[0,.5,1],[1,0,0]]:
        with pytest.raises(ValueError,match='unreachable'):
            machine.inverse([20,0,10],direction)


def test_origin_chooses_reachable_negative_tilt_and_unwraps_yaw():
    machine=S5Machine()
    p=np.array([0.,0.,5.]); n=np.array([1.,0.,0.])
    pose=machine.inverse(p,n,previous_c=179)
    assert pose[2]==-90
    actual_p,actual_n=machine.forward(pose)
    np.testing.assert_allclose(actual_p,p,atol=1e-12)
    np.testing.assert_allclose(actual_n,n,atol=1e-12)
    pose=machine.inverse([-20,-.01,3],[0,0,1],previous_c=179)
    assert 180<pose[3]<181
