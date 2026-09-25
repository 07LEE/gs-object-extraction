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
