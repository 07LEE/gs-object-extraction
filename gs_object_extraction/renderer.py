"""Graphdeco CUDA rendering and matrix-free mask lifting.

Needs PyTorch and the ``diff_gaussian_rasterization`` extension (the gs_train
environment). Lifting sets colour features to zero with gradients on, renders,
and back-propagates a per-pixel label image: the gradient on Gaussian i is
sum_p T_i(p) alpha_i(p) over the labelled pixels, so one backward pass gives
every Gaussian's contribution without an N x H x W tensor. Nothing is optimised.
"""

from dataclasses import dataclass
import inspect
import numpy as np


def validate_labels(labels, shape):
    labels = np.asarray(labels)
    if labels.shape != shape or not np.all(np.isin(labels, [-1, 0, 1])):
        raise ValueError("labels must match the image and contain only -1, 0, 1")
    return labels


def projection_matrix(camera):
    """Column-vector camera->clip projection for Graphdeco's pixel centers."""
    near, far = max(camera.near, .001), camera.far
    if not np.isfinite(far):
        far = 1e6
    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0], p[1, 1] = 2*camera.fx/camera.width, 2*camera.fy/camera.height
    p[0, 2] = (2*camera.cx+1)/camera.width-1
    p[1, 2] = (2*camera.cy+1)/camera.height-1
    p[2, 2], p[2, 3], p[3, 2] = far/(far-near), -far*near/(far-near), 1
    return p


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


class GraphdecoRenderer:
    def __init__(self, scene):
        try:
            import torch
            import diff_gaussian_rasterization as dgr
        except ImportError as exc:
            raise RuntimeError("rendering needs PyTorch and diff_gaussian_rasterization; use the gs_train Python environment") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not accessible in this process")
        self.torch, self.dgr = torch, dgr
        self.n = len(scene.means)
        if self.n == 0:
            raise ValueError("renderer requires a nonempty scene")
        self.means = self._tensor(scene.means)
        self.scales = self._tensor(scene.scales)
        self.rotations = self._tensor(scene.quaternions)
        self.opacities = self._tensor(scene.opacities[:, None])
        self.sh = self._tensor(scene.sh)
        self.degree = int(np.sqrt(scene.sh.shape[1]))-1
        self.screen = torch.zeros_like(self.means)

    def _tensor(self, values):
        return self.torch.as_tensor(np.asarray(values).copy(), dtype=self.torch.float32, device="cuda").contiguous()

    def _settings(self, camera, background):
        view = self._tensor(camera.world_to_camera.T)
        proj = self._tensor(projection_matrix(camera).T)
        kwargs = dict(image_height=camera.height, image_width=camera.width,
                      tanfovx=camera.width/(2*camera.fx), tanfovy=camera.height/(2*camera.fy),
                      bg=self._tensor(background), scale_modifier=1., viewmatrix=view,
                      projmatrix=view@proj, sh_degree=self.degree, campos=self._tensor(camera.eye),
                      prefiltered=False, debug=False)
        if "antialiasing" in inspect.signature(self.dgr.GaussianRasterizationSettings).parameters:
            kwargs["antialiasing"] = False
        return self.dgr.GaussianRasterizationSettings(**kwargs)

    def _rasterize(self, camera, *, features=None, active=None, background=(0, 0, 0)):
        opacity = self.opacities
        if active is not None:
            mask = np.asarray(active)
            if mask.shape != (self.n,) or mask.dtype != np.bool_:
                raise ValueError("active must be a boolean array of shape N")
            opacity = opacity*self._tensor(mask[:, None])
        rasterizer = self.dgr.GaussianRasterizer(raster_settings=self._settings(camera, background))
        return rasterizer(means3D=self.means, means2D=self.screen, opacities=opacity,
                          shs=self.sh if features is None else None, colors_precomp=features,
                          scales=self.scales, rotations=self.rotations)

    def render_features(self, camera, features, *, active=None, background=(0, 0, 0)):
        features = np.asarray(features, dtype=float)
        if features.shape != (self.n, 3) or not np.isfinite(features).all():
            raise ValueError("features must be finite N x 3")
        with self.torch.no_grad():
            result = self._rasterize(camera, features=self._tensor(features), active=active, background=background)
        return result[0].permute(1, 2, 0).cpu().numpy()

    def _alpha(self, camera, active):
        """Feature-1 rendering is alpha under the same geometry and compositing."""
        return self._rasterize(camera, features=self.torch.ones_like(self.means), active=active)[0][0]

    def alpha_image(self, camera, *, active=None):
        """Coverage of the drawn Gaussians, H x W; where the object lands on screen."""
        with self.torch.no_grad():
            return self._alpha(camera, active).cpu().numpy()

    def render(self, camera, *, active=None, background=(0, 0, 0)):
        """RGB over ``background`` and alpha; ``active`` renders only those Gaussians."""
        with self.torch.no_grad():
            rgb = self._rasterize(camera, active=active, background=background)[0].permute(1, 2, 0).cpu().numpy()
            alpha = self._alpha(camera, active).cpu().numpy()
        return Frame(rgb, alpha)

    def render_image(self, camera, *, active=None, background=(0, 0, 0)):
        """8-bit RGB (H x W x 3) for display, converted on the GPU."""
        with self.torch.no_grad():
            color = self._rasterize(camera, active=active, background=background)[0]
            return (color.clamp(0, 1) * 255).round().to(self.torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()

    def depth_image(self, camera, *, active=None):
        """Camera-space depth (alpha-weighted mean over the drawn Gaussians) and alpha, both H x W."""
        with self.torch.no_grad():
            w2c = self._tensor(camera.world_to_camera)
            z = self.means @ w2c[2, :3] + w2c[2, 3]
            weighted = self._rasterize(camera, features=z[:, None].expand(-1, 3).contiguous(), active=active)[0][0]
            alpha = self._alpha(camera, active)
            return (weighted / alpha.clamp_min(1e-6)).cpu().numpy(), alpha.cpu().numpy()

    def lift(self, camera, labels, *, active=None):
        """Contributions to label-1, label-0 and all pixels; label -1 is ignored.

        ``active`` renders only those Gaussians (e.g. a selection on its own), so
        inactive ones get zero and ones they hid can become visible.
        """
        labels = validate_labels(labels, (camera.height, camera.width))
        torch = self.torch
        features = torch.zeros_like(self.means, requires_grad=True)
        with torch.enable_grad():
            image = self._rasterize(camera, features=features, active=active)[0]
            grad_pixels = self._tensor(np.stack((labels == 1, labels == 0, np.ones_like(labels)), axis=0))
            accumulated, = torch.autograd.grad(image, features, grad_outputs=grad_pixels)
        values = accumulated.detach().cpu().numpy().astype(np.float64)
        # Reject numerical failures; permit tiny signed rounding only.
        if not np.isfinite(values).all() or values.min(initial=0) < -1e-5:
            raise RuntimeError("lifting produced invalid contribution sums")
        values = np.maximum(values, 0)
        return Lifted(values[:, 0], values[:, 1], values[:, 2])
