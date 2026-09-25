"""Which of an object's Gaussians fall inside a box the user drew on the screen."""

import numpy as np


def project(camera, means):
    """Pixel x, y and camera depth of each mean."""
    rotation, translation = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
    local = np.asarray(means, dtype=float) @ rotation.T + translation
    depth = local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):  # behind the camera is dropped by the depth test
        return camera.fx * local[:, 0] / depth + camera.cx, camera.fy * local[:, 1] / depth + camera.cy, depth


def inside_box(camera, means, box):
    """Gaussians in front of the camera whose centre lands in ``box`` = (x0, y0, x1, y1), in any corner order.

    Depth does not matter: whatever lies behind the box along the same ray goes with it,
    so the box is best drawn from a view where the unwanted pieces stand clear of the object.
    """
    x0, y0, x1, y1 = box
    x, y, depth = project(camera, means)
    return ((depth > camera.near) & (x >= min(x0, x1)) & (x <= max(x0, x1))
            & (y >= min(y0, y1)) & (y <= max(y0, y1)))


EDGES = tuple((i, i | bit) for i in range(8) for bit in (1, 2, 4) if not i & bit)  # corners differing in one bit


def bounding_box(means, up):
    """Eight corners and the size of the box around ``means`` that stands on the up axis and turns to fit them.

    Corner ``i`` takes the low or high end of each axis by the bits of ``i``. The box holds every
    mean, strays included, so one left far from the object shows as a box wider than the object.
    """
    means = np.asarray(means, dtype=float)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    ref = np.array([1., 0., 0.]) if abs(up[0]) < .9 else np.array([0., 0., 1.])
    e1 = np.cross(up, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    flat = np.stack((means @ e1, means @ e2), axis=1)
    major = np.array([1., 0.])
    if len(means) > 2:
        major = np.linalg.eigh(np.cov((flat - flat.mean(axis=0)).T))[1][:, 1]  # the way the object is longest
    axes = np.stack((major[0] * e1 + major[1] * e2, -major[1] * e1 + major[0] * e2, up))
    coordinates = means @ axes.T
    low, high = coordinates.min(axis=0), coordinates.max(axis=0)
    corners = np.array([[(high if i >> k & 1 else low)[k] for k in range(3)] for i in range(8)]) @ axes
    return corners, high - low
