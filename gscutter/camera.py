"""Rigid pinhole camera. Pixel centres are integers; camera axes are x right, y down, z forward."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _vector(value, name: str, length: int = 3) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


@dataclass(frozen=True)
class Camera:
    """Rigid pinhole camera; near/far bounds apply to Gaussian mean depth."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray
    near: float = 1e-3
    far: float = np.inf

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fx", "fy", "cx", "cy", "near", "far"):
            value = getattr(self, name)
            if not np.isscalar(value) or isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a scalar")
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a real scalar") from exc
            object.__setattr__(self, name, scalar)
        if not np.all(np.isfinite([self.fx, self.fy, self.cx, self.cy, self.near])):
            raise ValueError("camera intrinsics and near must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.near <= 0 or np.isnan(self.far) or self.far <= self.near:
            raise ValueError("camera requires 0 < near < far")
        matrix = np.array(self.world_to_camera, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("world_to_camera must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
            raise ValueError("world_to_camera must have homogeneous last row [0,0,0,1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
            raise ValueError("world_to_camera rotation must be orthonormal with determinant +1")
        matrix.setflags(write=False)
        object.__setattr__(self, "world_to_camera", matrix)

    @property
    def eye(self) -> np.ndarray:
        """Camera origin in world coordinates."""
        return -self.world_to_camera[:3, :3].T @ self.world_to_camera[:3, 3]

    @classmethod
    def look_at(
        cls,
        eye,
        target,
        width: int = 96,
        height: int = 96,
        fov_y_degrees: float = 55,
        up=(0, 1, 0),
        *,
        near: float = 1e-3,
        far: float = np.inf,
    ) -> Camera:
        """Look toward target; positive world ``up`` projects image-up.

        ``up`` must not be parallel to the viewing direction. With world +Y up,
        a camera at (0,0,3) looking at the origin has image-right world +X.
        """
        eye = _vector(eye, "eye")
        target = _vector(target, "target")
        up = _vector(up, "up")
        if not np.isscalar(fov_y_degrees) or not np.isfinite(fov_y_degrees) or not 0 < fov_y_degrees < 180:
            raise ValueError("fov_y_degrees must lie strictly between 0 and 180")
        forward = target - eye
        length = np.linalg.norm(forward)
        if not np.isfinite(length) or length < 1e-12:
            raise ValueError("eye and target must be distinct")
        forward = forward / length
        up_length = np.linalg.norm(up)
        if not np.isfinite(up_length) or up_length < 1e-12:
            raise ValueError("up must be nonzero")
        right = np.cross(forward, up / up_length)
        length = np.linalg.norm(right)
        if length < 1e-8:
            raise ValueError("up must not be parallel to the view direction")
        right = right / length
        down = np.cross(forward, right)
        matrix = np.eye(4)
        matrix[:3, :3] = np.stack((right, down, forward))
        matrix[:3, 3] = -matrix[:3, :3] @ eye
        focal = 0.5 * height / np.tan(np.deg2rad(fov_y_degrees) / 2)
        return cls(width, height, focal, focal, (width - 1) / 2, (height - 1) / 2, matrix, near, far)
