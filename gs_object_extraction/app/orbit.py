"""Orbit camera for the viewer: yaw and pitch around a target, pan in the view plane, zoom by distance.

A Gaussian PLY carries no up direction. ``estimate_up`` takes the normal of the
dominant plane near the scene centre (usually the floor) and points it toward
the side with more content; the viewer can also force one of the six axes.
"""

from dataclasses import dataclass, field
import numpy as np
from ..camera import Camera

AXES = {"+X": (1., 0., 0.), "-X": (-1., 0., 0.), "+Y": (0., 1., 0.),
        "-Y": (0., -1., 0.), "+Z": (0., 0., 1.), "-Z": (0., 0., -1.)}
DEFAULT_UP = AXES["-Y"]  # camera up in COLMAP's first view, the usual fallback
PITCH_LIMIT = np.deg2rad(89.)
PLANE_RATIO = .2  # least/middle variance below this counts as a clear plane
FRAMING_STEPS = (.6, 1., 1.5, 2.5, 4.)  # distances to try, in medians of the scene's spread
OPEN = .65  # a view is open when what it looks at sits at least this far out, of the way to the middle
FILLED = .95  # ... and it still has to look at the scene, not past it


def estimate_up(means):
    """Unit up vector: normal of the dominant plane among the central half of the Gaussians.

    The floor is a thin sheet with walls, plants and the object standing on it,
    so the long tail of heights marks the up side. Without a clear plane the
    COLMAP-style -Y is returned.
    """
    means = np.asarray(means, dtype=float)
    centre = np.median(means, axis=0)
    radius = np.linalg.norm(means - centre, axis=1)
    near = means[radius <= np.median(radius)]
    if len(near) < 10:
        return np.array(DEFAULT_UP)
    offsets = near - near.mean(axis=0)
    values, vectors = np.linalg.eigh(offsets.T @ offsets / len(near))
    if values[1] <= 0 or values[0] / values[1] > PLANE_RATIO:
        return np.array(DEFAULT_UP)
    normal = vectors[:, 0]
    heights = offsets @ normal
    skew = np.mean(heights ** 3)
    return normal if skew >= 0 else -normal


@dataclass
class Orbit:
    target: np.ndarray
    distance: float
    yaw: float = 0.
    pitch: float = np.deg2rad(20.)
    up: np.ndarray = field(default_factory=lambda: np.array(DEFAULT_UP))
    fov_y: float = 50.

    def __post_init__(self):
        self.target = np.asarray(self.target, dtype=float)
        self.up = _unit(self.up)

    @classmethod
    def framing(cls, means, up=DEFAULT_UP, fov_y=50., distance=.6):
        """Look at the median Gaussian from ``distance`` times the median spread around it."""
        means = np.asarray(means, dtype=float)
        centre = np.median(means, axis=0)
        spread = float(np.median(np.linalg.norm(means - centre, axis=1)))
        return cls(centre, max(distance * spread, 1e-6), up=up, fov_y=fov_y)

    def _basis(self):
        u = self.up
        ref = np.array([1., 0., 0.]) if abs(u[0]) < .9 else np.array([0., 0., 1.])
        e1 = np.cross(u, ref)
        e1 /= np.linalg.norm(e1)
        return u, e1, np.cross(u, e1)

    @property
    def eye(self):
        u, e1, e2 = self._basis()
        direction = np.cos(self.pitch) * (np.cos(self.yaw) * e1 + np.sin(self.yaw) * e2) + np.sin(self.pitch) * u
        return self.target + self.distance * direction

    def camera(self, width, height):
        return Camera.look_at(self.eye, self.target, int(width), int(height), self.fov_y, self.up,
                              near=max(self.distance * 1e-3, 1e-4))

    def rotate(self, dx, dy, speed=.005):
        """Drag by (dx, dy) pixels: the scene turns with the mouse."""
        self.yaw -= dx * speed
        self.pitch = float(np.clip(self.pitch + dy * speed, -PITCH_LIMIT, PITCH_LIMIT))

    def pan(self, dx, dy, height):
        """Drag by (dx, dy) pixels: points at the target's depth follow the mouse exactly."""
        rows = self.camera(max(int(height), 1), max(int(height), 1)).world_to_camera[:3, :3]
        per_pixel = 2 * self.distance * np.tan(np.deg2rad(self.fov_y) / 2) / height
        self.target = self.target - (rows[0] * dx + rows[1] * dy) * per_pixel

    def zoom(self, steps, factor=.9):
        """Positive steps move closer."""
        self.distance = max(self.distance * factor ** steps, 1e-6)

    def look_from(self, eye, target):
        """Put the orbit centre on ``target`` without moving the camera."""
        u, e1, e2 = self._basis()
        offset = np.asarray(eye, dtype=float) - np.asarray(target, dtype=float)
        self.distance = max(float(np.linalg.norm(offset)), 1e-6)
        d = offset / self.distance
        self.target = np.asarray(target, dtype=float)
        self.pitch = float(np.clip(np.arcsin(np.clip(d @ u, -1, 1)), -PITCH_LIMIT, PITCH_LIMIT))
        self.yaw = float(np.arctan2(d @ e2, d @ e1))

    def set_up(self, up):
        """New up vector, keeping the camera where it is."""
        eye = self.eye
        self.up = _unit(up)
        self.look_from(eye, self.target)


def _unit(vector):
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    if v.shape != (3,) or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("up must be a nonzero 3-vector")
    return v / norm


def frame_scene(renderer, means, up=DEFAULT_UP, fov_y=50., size=(160, 120), steps=FRAMING_STEPS,
                open_enough=OPEN, filled_enough=FILLED):
    """Stand as far back as the scene lets you, and look at the middle of it.

    A capture that circles one subject keeps background all around it, so the camera
    has to stay inside that shell; a scene shot along its subject, a train beside its
    track, needs room to see it whole. Two cheap renders per distance tell them apart.
    A blocked view looks at something close in front of it, relative to how far the
    camera stands. A view that has escaped the scene has empty frame around it. The
    furthest distance that is neither is where the scene opens up.
    """
    means = np.asarray(means, dtype=float)
    centre = np.median(means, axis=0)  # measured once: the scan only moves the camera
    spread = float(np.median(np.linalg.norm(means - centre, axis=1)))
    for distance in sorted(steps, reverse=True):  # the furthest that works wins, so start there
        orbit = Orbit(centre, max(distance * spread, 1e-6), up=up, fov_y=fov_y)
        depth, alpha = renderer.depth_image(orbit.camera(*size))
        solid = alpha > .5
        if solid.mean() >= filled_enough and float(np.median(depth[solid])) / orbit.distance >= open_enough:
            return orbit
    return Orbit(centre, max(.6 * spread, 1e-6), up=up, fov_y=fov_y)


def unproject(camera, x, y, depth):
    """World point seen at pixel (x, y) with camera-space depth ``depth``."""
    local = np.array([(x - camera.cx) / camera.fx * depth, (y - camera.cy) / camera.fy * depth, depth])
    rotation, translation = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
    return rotation.T @ (local - translation)
