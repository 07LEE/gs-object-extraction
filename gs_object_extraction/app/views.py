"""Views the user has confirmed: the camera a view was rendered with and the object mask drawn on it."""

from dataclasses import dataclass
import numpy as np
from ..camera import Camera


@dataclass(frozen=True)
class MaskedView:
    camera: Camera
    mask: np.ndarray  # bool, camera.height x camera.width
    points: tuple  # (x, y) pixel prompts on the rendered view
    labels: tuple  # 1 object, 0 background

    def __post_init__(self):
        if self.mask.shape != (self.camera.height, self.camera.width) or self.mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean image of the camera's size")
        if not self.mask.any():
            raise ValueError("mask is empty")
