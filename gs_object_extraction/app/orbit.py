"""Orbit camera for the viewer: yaw and pitch around a target, pan in the view plane, zoom by distance.

A Gaussian PLY carries no up direction, so ``up`` is one of the six world axes
and can be switched in the viewer. COLMAP-based scenes usually have world -Y up.
"""

from dataclasses import dataclass
import numpy as np
from ..camera import Camera

AXES = {"+X": (1., 0., 0.), "-X": (-1., 0., 0.), "+Y": (0., 1., 0.),
        "-Y": (0., -1., 0.), "+Z": (0., 0., 1.), "-Z": (0., 0., -1.)}
PITCH_LIMIT = np.deg2rad(89.)


@dataclass
class Orbit:
    target: np.ndarray
    distance: float
    yaw: float = 0.
    pitch: float = np.deg2rad(20.)
    up: str = "-Y"
    fov_y: float = 50.

    @classmethod
    def framing(cls, means, up="-Y", fov_y=50.):
        """Look at the middle of the 1-99 percentile box so far floaters do not shrink the scene."""
        lo, hi = np.percentile(np.asarray(means, dtype=float), [1, 99], axis=0)
        radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
        return cls((lo + hi) / 2, radius / np.tan(np.deg2rad(fov_y) / 2) * 1.1, up=up, fov_y=fov_y)

    def _basis(self):
        u = np.asarray(AXES[self.up])
        ref = np.array([1., 0., 0.]) if abs(u[0]) < .9 else np.array([0., 0., 1.])
        e1 = np.cross(u, ref)
        e1 /= np.linalg.norm(e1)
        return u, e1, np.cross(u, e1)

    @property
    def eye(self):
        u, e1, e2 = self._basis()
        direction = np.cos(self.pitch) * (np.cos(self.yaw) * e1 + np.sin(self.yaw) * e2) + np.sin(self.pitch) * u
        return np.asarray(self.target, dtype=float) + self.distance * direction

    def camera(self, width, height):
        return Camera.look_at(self.eye, self.target, int(width), int(height), self.fov_y, AXES[self.up],
                              near=max(self.distance * 1e-3, 1e-4))

    def rotate(self, dx, dy, speed=.005):
        """Drag by (dx, dy) pixels: the scene turns with the mouse."""
        self.yaw -= dx * speed
        self.pitch = float(np.clip(self.pitch + dy * speed, -PITCH_LIMIT, PITCH_LIMIT))

    def pan(self, dx, dy, height):
        """Drag by (dx, dy) pixels: points at the target's depth follow the mouse exactly."""
        rows = self.camera(max(int(height), 1), max(int(height), 1)).world_to_camera[:3, :3]
        per_pixel = 2 * self.distance * np.tan(np.deg2rad(self.fov_y) / 2) / height
        self.target = np.asarray(self.target, dtype=float) - (rows[0] * dx + rows[1] * dy) * per_pixel

    def zoom(self, steps, factor=.9):
        """Positive steps move closer."""
        self.distance = max(self.distance * factor ** steps, 1e-6)

    def set_up(self, axis):
        if axis not in AXES:
            raise ValueError(f"up must be one of {', '.join(AXES)}")
        self.up, self.yaw = axis, 0.
