"""CUDA rendering of trained 3DGS scenes with gsplat, and matrix-free mask lifting.

Lifting sets colour features to zero with gradients on, renders, and back-propagates a
per-pixel label image: the gradient on Gaussian i is sum_p T_i(p) alpha_i(p) over the
labelled pixels, so one backward pass gives every Gaussian's contribution without an
N x H x W tensor. Nothing is optimised.
"""

import contextlib
from dataclasses import dataclass
import threading
import numpy as np


def validate_labels(labels, shape):
    labels = np.asarray(labels)
    if labels.shape != shape or not np.all(np.isin(labels, [-1, 0, 1])):
        raise ValueError("labels must match the image and contain only -1, 0, 1")
    return labels


@dataclass
class Frame:
    rgb: np.ndarray
    alpha: np.ndarray


@dataclass
class Lifted:
    """Per-Gaussian T*alpha summed over inside (label 1), outside (label 0) and all pixels."""

    inside: np.ndarray
    outside: np.ndarray
    total: np.ndarray

    @classmethod
    def zeros(cls, n):
        return cls(np.zeros(n), np.zeros(n), np.zeros(n))

    def __iadd__(self, other):
        self.inside += other.inside
        self.outside += other.outside
        self.total += other.total
        return self


class Exclusive:
    """One caller at a time on the GPU buffers a renderer holds.

    The buffers are read-only while drawing, but a background job and the window reach
    for the same renderer, and they share its screen-space buffer and the default CUDA
    stream. Calls are handed out one at a time; the window asks without blocking, so
    its event loop never queues behind a lift.
    """

    def __init__(self):
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def held(self, *, blocking=True):
        """Hold the renderer across a series of calls; yields whether it was obtained.

        ``blocking=False`` gives up at once instead of queueing behind a lift, which
        costs several frames' worth of time on any scene worth extracting: the window
        skips that frame rather than stall its event loop.
        """
        obtained = self._lock.acquire(blocking)
        try:
            yield obtained
        finally:
            if obtained:
                self._lock.release()


class GsplatRenderer(Exclusive):
    """Draws a trained scene and lifts masks through gsplat.

    Conventions are matched to Graphdeco 3DGS, which most PLY files are trained with, so a
    scene renders as trained:
    Graphdeco puts pixel i's centre at coordinate i and gsplat at i + 0.5, so the principal
    point is shifted by half a pixel; Gaussians whose mean is at depth 0.2 or less are
    dropped, as Graphdeco's frustum test does; rasterisation is classic, with the same 0.3
    low-pass on the 2D covariance and no antialiasing. gsplat caps a Gaussian's alpha at 0.999
    where Graphdeco caps it at 0.99.
    """

    NEAR = .2

    def __init__(self, scene):
        super().__init__()
        try:
            import torch
            from gsplat import rasterization
        except ImportError as exc:
            raise RuntimeError("rendering needs PyTorch and gsplat; see Installation in the README") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not accessible in this process")
        self.torch, self.rasterization = torch, rasterization
        self.n = len(scene.means)
        if self.n == 0:
            raise ValueError("renderer requires a nonempty scene")
        self.means = self._tensor(scene.means)
        self.scales = self._tensor(scene.scales)
        self.rotations = self._tensor(scene.quaternions)
        self.opacities = self._tensor(scene.opacities)
        self.sh = self._tensor(scene.sh)
        self.degree = int(np.sqrt(scene.sh.shape[1]))-1

    def _tensor(self, values):
        return self.torch.as_tensor(np.asarray(values).copy(), dtype=self.torch.float32, device="cuda").contiguous()

    def update_scales(self, scales):
        """Replace sizes in the existing GPU buffer; leave all other attributes intact."""
        scales = np.asarray(scales, dtype=np.float32)
        if scales.shape != (self.n, 3) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError("scales must be positive finite N x 3")
        with self._lock, self.torch.no_grad():
            self.scales.copy_(self.torch.as_tensor(np.ascontiguousarray(scales)))

    def _rasterize(self, camera, *, features=None, active=None, background=(0, 0, 0)):
        """Image (H x W x channels) and alpha (H x W) as tensors on the GPU."""
        opacity = self.opacities
        if active is not None:
            mask = np.asarray(active)
            if mask.shape != (self.n,) or mask.dtype != np.bool_:
                raise ValueError("active must be a boolean array of shape N")
            opacity = opacity*self._tensor(mask)
        view = self._tensor(camera.world_to_camera)[None]
        k = self._tensor([[camera.fx, 0, camera.cx+.5], [0, camera.fy, camera.cy+.5], [0, 0, 1]])[None]
        far = camera.far if np.isfinite(camera.far) else 1e10
        colors, degree = (self.sh, self.degree) if features is None else (features, None)
        image, alpha, _ = self.rasterization(self.means, self.rotations, self.scales, opacity, colors, view, k,
                                             camera.width, camera.height, near_plane=self.NEAR, far_plane=far,
                                             sh_degree=degree, backgrounds=self._tensor(background)[None],
                                             rasterize_mode="classic", packed=False)
        return image[0], alpha[0, ..., 0]

    def render_features(self, camera, features, *, active=None, background=(0, 0, 0)):
        features = np.asarray(features, dtype=float)
        if features.shape != (self.n, 3) or not np.isfinite(features).all():
            raise ValueError("features must be finite N x 3")
        with self._lock, self.torch.no_grad():
            image, _ = self._rasterize(camera, features=self._tensor(features), active=active, background=background)
        return image.cpu().numpy()

    def alpha_image(self, camera, *, active=None):
        """Coverage of the drawn Gaussians, H x W; where the object lands on screen."""
        with self._lock, self.torch.no_grad():
            _, alpha = self._rasterize(camera, features=self.torch.ones_like(self.means), active=active)
        return alpha.cpu().numpy()

    def render(self, camera, *, active=None, background=(0, 0, 0)):
        """RGB over ``background`` and alpha; ``active`` renders only those Gaussians."""
        with self._lock, self.torch.no_grad():
            image, alpha = self._rasterize(camera, active=active, background=background)
        return Frame(image.cpu().numpy(), alpha.cpu().numpy())

    def render_image(self, camera, *, active=None, background=(0, 0, 0)):
        """8-bit RGB (H x W x 3) for display, converted on the GPU."""
        with self._lock, self.torch.no_grad():
            image, _ = self._rasterize(camera, active=active, background=background)
            return (image.clamp(0, 1) * 255).round().to(self.torch.uint8).contiguous().cpu().numpy()

    def depth_image(self, camera, *, active=None):
        """Camera-space depth (alpha-weighted mean over the drawn Gaussians) and alpha, both H x W."""
        with self._lock, self.torch.no_grad():
            w2c = self._tensor(camera.world_to_camera)
            z = self.means @ w2c[2, :3] + w2c[2, 3]
            weighted, alpha = self._rasterize(camera, features=z[:, None].expand(-1, 3).contiguous(), active=active)
            return (weighted[..., 0] / alpha.clamp_min(1e-6)).cpu().numpy(), alpha.cpu().numpy()

    def lift(self, camera, labels, *, active=None):
        """Contributions to label-1, label-0 and all pixels; label -1 is ignored.

        ``active`` renders only those Gaussians (e.g. a selection on its own), so
        inactive ones get zero and ones they hid can become visible.
        """
        labels = validate_labels(labels, (camera.height, camera.width))
        torch = self.torch
        features = torch.zeros_like(self.means, requires_grad=True)
        with self._lock, torch.enable_grad():
            image, _ = self._rasterize(camera, features=features, active=active)
            weights = self._tensor(np.stack((labels == 1, labels == 0, np.ones_like(labels)), axis=-1))
            accumulated, = torch.autograd.grad(image, features, grad_outputs=weights)
        values = accumulated.detach().cpu().numpy().astype(np.float64)
        # Reject numerical failures; permit tiny signed rounding only.
        if not np.isfinite(values).all() or values.min(initial=0) < -1e-5:
            raise RuntimeError("lifting produced invalid contribution sums")
        np.maximum(values, 0, out=values)  # a fresh float64 copy, so clip it where it is
        return Lifted(values[:, 0], values[:, 1], values[:, 2])
