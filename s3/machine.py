"""Geometric pose adapter for S5's default Core-R-Theta writer.

Units are millimetres and degrees. This checks orientation feasibility only;
axis travel, collision checks, extrusion and timed motion remain separate stages.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class S5Machine:
    nozzle_offset: float = 41.5
    minimum_b: float = -130.
    maximum_b: float = 30.
    angular_tolerance: float = 1e-6

    def __post_init__(self):
        if not np.isfinite([self.nozzle_offset, self.minimum_b, self.maximum_b,
                            self.angular_tolerance]).all():
            raise ValueError('Machine parameters must be finite')
        if self.nozzle_offset < 0 or self.minimum_b >= self.maximum_b or not 0 < self.angular_tolerance < 1:
            raise ValueError('Invalid machine geometry or limits')

    def forward(self, pose):
        """Return tip position and upward printing direction from [X,Z,B,C]."""
        x, z, b, c = np.asarray(pose, dtype=float)
        if not np.isfinite([x,z,b,c]).all():
            raise ValueError('Pose must be finite')
        b,c = np.deg2rad([b,c])
        radial = np.array([np.cos(c), np.sin(c), 0.])
        direction = np.sin(b)*radial + [0,0,np.cos(b)]
        position = (x+self.nozzle_offset*np.sin(b))*radial
        position[2] = z-self.nozzle_offset*(np.cos(b)-1)
        return position, direction

    def inverse(self, position, direction, previous_c=0., *, strict=True):
        """Exact radial-plane inverse; raise for unavailable nozzle orientations.

        With strict=False, project direction onto the radial plane and emit B
        without clipping or enforcing B limits. Tip position is preserved.
        C is unwrapped near previous_c. At the polar origin, the direction sets
        yaw; an axial direction preserves the previous yaw. No angle clipping.
        """
        p=np.asarray(position,dtype=float); n=np.asarray(direction,dtype=float)
        if p.shape!=(3,) or n.shape!=(3,) or not np.isfinite(np.r_[p,n,previous_c]).all() or np.linalg.norm(n)==0:
            raise ValueError('Expected finite position and nonzero direction')
        n=n/np.linalg.norm(n); r=np.linalg.norm(p[:2])
        if r>1e-10:
            c=np.arctan2(p[1],p[0])
            candidates=[c]
        elif np.linalg.norm(n[:2])>1e-12:
            c=np.arctan2(n[1],n[0]); candidates=[c,c+np.pi]
        else:
            candidates=[np.deg2rad(previous_c)]
        for c in candidates:
            radial=np.array([np.cos(c),np.sin(c),0.])
            b=np.arctan2(n@radial,n[2])
            reconstructed=np.sin(b)*radial+[0,0,np.cos(b)]
            error=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(n,reconstructed)), n@reconstructed))
            if strict and error>self.angular_tolerance:
                continue
            bdeg=np.rad2deg(b)
            if strict and not self.minimum_b<=bdeg<=self.maximum_b:
                continue
            cdeg=previous_c+(np.rad2deg(c)-previous_c+180)%360-180
            return np.array([r-self.nozzle_offset*np.sin(b),
                             p[2]+self.nozzle_offset*(np.cos(b)-1),bdeg,cdeg])
        raise ValueError('Requested direction is unreachable in the S5 radial plane or B limits')
