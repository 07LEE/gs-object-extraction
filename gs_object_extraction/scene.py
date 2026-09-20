"""Validated, renderer-independent Gaussian splat data.

Scales and opacities are activated values, rotations are unit quaternions in
``(w, x, y, z)`` order, and SH coefficients have shape ``(N, K, 3)``.
"""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


SH_C0 = 0.28209479177387814
SUPPORTED_SH_COUNTS = (1, 4, 9, 16)


def _finite_array(value, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass
class GaussianScene:
    """Gaussian parameters with stable IDs and optional scalar PLY attributes.

    Construction copies input arrays. IDs must be unique integers; subsets and
    copies retain them. Empty scenes are valid, for example after a selection.
    """

    means: np.ndarray
    scales: np.ndarray
    quaternions: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray
    ids: np.ndarray | None = None
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.means = _finite_array(self.means, "means")
        if self.means.ndim != 2 or self.means.shape[1] != 3:
            raise ValueError("means must have shape (N, 3)")
        n = len(self.means)
        self.scales = _finite_array(self.scales, "scales")
        self.quaternions = _finite_array(self.quaternions, "quaternions")
        self.opacities = _finite_array(self.opacities, "opacities")
        self.sh = _finite_array(self.sh, "sh")
        for name, shape in (("scales", (n, 3)), ("quaternions", (n, 4)),
                            ("opacities", (n,))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if np.any(self.scales <= 0):
            raise ValueError("scales must be strictly positive")
        if np.any((self.opacities < 0) | (self.opacities > 1)):
            raise ValueError("opacities must lie in [0, 1]")
        if not np.allclose(np.linalg.norm(self.quaternions, axis=1), 1.0,
                           rtol=0, atol=1e-6):
            raise ValueError("quaternions must be unit length in wxyz order")
        if (self.sh.ndim != 3 or self.sh.shape[0] != n
                or self.sh.shape[2] != 3
                or self.sh.shape[1] not in SUPPORTED_SH_COUNTS):
            raise ValueError("sh must have shape (N, K, 3), with K in 1, 4, 9, 16")
        if self.ids is None:
            self.ids = np.arange(n, dtype=np.int64)
        else:
            ids = np.asarray(self.ids)
            if ids.shape != (n,) or ids.dtype.kind not in "iu":
                raise ValueError("ids must be an integer array of shape (N,)")
            if ids.dtype.kind == "u" and np.any(ids > np.iinfo(np.int64).max):
                raise ValueError("ids must fit signed 64-bit integers")
            self.ids = np.array(ids, dtype=np.int64, copy=True)
        ordered = np.sort(self.ids)
        if n and np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("ids must be unique")
        extras = {}
        for name, values in (self.extras or {}).items():
            if (not isinstance(name, str) or not name or not name.isascii()
                    or any(c.isspace() or ord(c) < 33 or ord(c) == 127 for c in name)):
                raise ValueError("extra attribute names must be nonempty printable ASCII tokens")
            array = np.array(values, copy=True)
            if array.shape != (n,) or array.dtype.kind not in "biuf":
                raise ValueError(f"extra attribute {name!r} must be a numeric scalar array of shape (N,)")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"extra attribute {name!r} must contain finite values")
            extras[name] = array
        self.extras = extras

    def __len__(self) -> int:
        return len(self.means)

    def copy(self) -> "GaussianScene":
        return GaussianScene(self.means, self.scales, self.quaternions,
                             self.opacities, self.sh, self.ids, self.extras)

    def subset(self, mask) -> "GaussianScene":
        """Return a copy selected by a boolean mask or integer index array."""
        index = np.asarray(mask)
        if index.ndim != 1 or index.dtype.kind not in "biu":
            raise ValueError("subset requires a one-dimensional boolean mask or integer indices")
        if index.dtype.kind == "b" and len(index) != len(self):
            raise ValueError("boolean subset mask must have length N")
        return GaussianScene(self.means[index], self.scales[index],
                             self.quaternions[index], self.opacities[index],
                             self.sh[index], self.ids[index],
                             {key: value[index] for key, value in self.extras.items()})

    @property
    def covariances(self) -> np.ndarray:
        """World-space covariance ``R @ diag(scales**2) @ R.T``."""
        w, x, y, z = self.quaternions.T
        rotation = np.empty((len(self), 3, 3), dtype=np.float64)
        rotation[:, 0, 0] = 1 - 2 * (y*y + z*z)
        rotation[:, 0, 1] = 2 * (x*y - z*w)
        rotation[:, 0, 2] = 2 * (x*z + y*w)
        rotation[:, 1, 0] = 2 * (x*y + z*w)
        rotation[:, 1, 1] = 1 - 2 * (x*x + z*z)
        rotation[:, 1, 2] = 2 * (y*z - x*w)
        rotation[:, 2, 0] = 2 * (x*z - y*w)
        rotation[:, 2, 1] = 2 * (y*z + x*w)
        rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
        return np.einsum("nik,nk,njk->nij", rotation, self.scales**2, rotation)

    @property
    def colors(self) -> np.ndarray:
        """Clipped DC-only RGB preview; full SH rendering can differ by view."""
        return np.clip(0.5 + SH_C0 * self.sh[:, 0, :], 0.0, 1.0)

    @classmethod
    def from_colors(cls, means, scales, colors, opacities=None,
                    quaternions=None) -> "GaussianScene":
        means = np.asarray(means, dtype=np.float64)
        colors = _finite_array(colors, "colors")
        if means.ndim != 2 or means.shape[1] != 3:
            raise ValueError("means must have shape (N, 3)")
        n = len(means)
        if colors.shape != (n, 3) or np.any((colors < 0) | (colors > 1)):
            raise ValueError("colors must have shape (N, 3) and lie in [0, 1]")
        if opacities is None:
            opacities = np.ones(n, dtype=np.float64)
        if quaternions is None:
            quaternions = np.zeros((n, 4), dtype=np.float64)
            quaternions[:, 0] = 1.0
        return cls(means, scales, quaternions, opacities,
                   ((colors - 0.5) / SH_C0)[:, None, :])
