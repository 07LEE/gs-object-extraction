"""COLMAP binary sparse-model reader and conversion to Camera objects.

Only undistorted pinhole models are supported. COLMAP places the upper-left
pixel centre at (0.5, 0.5); this package and Graphdeco place it at (0, 0), so the
principal point moves by half a pixel after any resize to the render size.
Graphdeco training ignores the principal point and assumes the image centre;
``principal_point_offset`` exposes how far a camera is from that assumption.
"""

from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np
from .camera import Camera

_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4)}


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: tuple

    @property
    def focal(self):
        return (self.params[0], self.params[0]) if self.model == "SIMPLE_PINHOLE" else self.params[:2]

    @property
    def principal(self):
        return self.params[-2:]


@dataclass(frozen=True)
class ColmapImage:
    id: int
    name: str
    qvec: tuple
    tvec: tuple
    camera_id: int


def read_cameras_binary(path):
    cameras = {}
    with Path(path).open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            if model_id not in _MODELS:
                raise ValueError(f"COLMAP camera model id {model_id} is not an undistorted pinhole model")
            model, count = _MODELS[model_id]
            cameras[camera_id] = ColmapCamera(camera_id, model, width, height, struct.unpack(f"<{count}d", f.read(8 * count)))
    return cameras


def read_images_binary(path):
    """Registered images keyed by file name; 2D keypoints are skipped."""
    images = {}
    with Path(path).open("rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            image_id, *pose, camera_id = struct.unpack("<i7di", f.read(64))
            name = bytearray()
            while (byte := f.read(1)) != b"\0":
                name += byte
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            image = ColmapImage(image_id, name.decode(), tuple(pose[:4]), tuple(pose[4:]), camera_id)
            images[image.name] = image
    return images


def qvec_to_rotation(qvec):
    """COLMAP (w, x, y, z) world-to-camera quaternion to a rotation matrix."""
    w, x, y, z = np.asarray(qvec, dtype=float) / np.linalg.norm(qvec)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def principal_point_offset(camera):
    """Principal point minus image centre, in COLMAP pixels."""
    cx, cy = camera.principal
    return cx - camera.width / 2, cy - camera.height / 2


def camera_from_colmap(camera, image, width=None, height=None):
    """Camera for ``image`` rendered at width x height (default: COLMAP size)."""
    width, height = int(width or camera.width), int(height or camera.height)
    sx, sy = width / camera.width, height / camera.height
    (fx, fy), (cx, cy) = camera.focal, camera.principal
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = qvec_to_rotation(image.qvec)
    world_to_camera[:3, 3] = image.tvec
    return Camera(width, height, fx * sx, fy * sy, cx * sx - .5, cy * sy - .5, world_to_camera)
