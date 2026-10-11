"""Turn a scene so a chosen up direction points the way another one does, for saving the object standing on its floor."""

import numpy as np

from .scene import GaussianScene

C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def rotation_between(source, target) -> np.ndarray:
    """The shortest rotation taking the direction ``source`` onto ``target``."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross, cosine = np.cross(a, b), float(a @ b)
    if cosine < -1 + 1e-9:  # opposite: half a turn about anything perpendicular
        axis = np.cross(a, (1., 0., 0.) if abs(a[0]) < .9 else (0., 1., 0.))
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def _basis(directions, degree):
    """Real SH basis of one degree at unit directions, in the order and signs of the Graphdeco PLY."""
    x, y, z = directions.T
    xx, yy, zz = x * x, y * y, z * z
    if degree == 1:
        return np.stack([-C1 * y, C1 * z, -C1 * x], axis=1)
    if degree == 2:
        return np.stack([C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)], axis=1)
    return np.stack([C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                     C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                     C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)], axis=1)


def sh_rotation(rotation, degree) -> np.ndarray:
    """The matrix taking one degree's coefficients to those of the same colours seen after ``rotation``.

    A degree is closed under rotation, so the matrix is fitted from the basis at a spread of
    directions rather than derived, which keeps it right whatever sign convention the basis has.
    """
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(64, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    before = _basis(directions, degree)
    after = _basis(directions @ rotation, degree)  # the basis at R^T d, as rows
    return np.linalg.lstsq(before, after, rcond=None)[0]


def _quaternion(rotation) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    trace = np.trace(rotation)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        w, x, y, z = s / 4, (rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(rotation[i, i] - rotation[j, j] - rotation[k, k] + 1)
        parts = np.empty(4)
        parts[0] = (rotation[k, j] - rotation[j, k]) / s
        parts[1 + i] = s / 4
        parts[1 + j] = (rotation[j, i] + rotation[i, j]) / s
        parts[1 + k] = (rotation[k, i] + rotation[i, k]) / s
        w, x, y, z = parts
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def rotated(scene: GaussianScene, rotation, pivot) -> GaussianScene:
    """A copy of the scene turned by ``rotation`` about ``pivot``: positions, orientations and view-dependent colour."""
    rotation = np.asarray(rotation, dtype=np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    means = (scene.means - pivot) @ rotation.T + pivot
    w1, x1, y1, z1 = _quaternion(rotation)
    w2, x2, y2, z2 = scene.quaternions.T
    quaternions = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    sh = scene.sh.copy()
    for degree, start in ((1, 1), (2, 4), (3, 9)):
        end = start + 2 * degree + 1
        if sh.shape[1] >= end:
            sh[:, start:end, :] = np.einsum("mk,nkc->nmc", sh_rotation(rotation, degree), scene.sh[:, start:end, :])
    return GaussianScene(means, scene.scales, quaternions, scene.opacities, sh, scene.ids, scene.extras)


def stood_up(scene: GaussianScene, up, target) -> GaussianScene:
    """The scene turned about its centre so the direction ``up`` points along ``target``."""
    if len(scene) == 0:
        return scene
    return rotated(scene, rotation_between(up, target), scene.means.mean(axis=0))


def grounded(scene: GaussianScene, up) -> GaussianScene:
    """The scene moved so its centre sits over the origin and its lowest point, along ``up``, is on the origin's level."""
    if len(scene) == 0:
        return scene
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    heights = scene.means @ up
    centre = scene.means.mean(axis=0)
    sideways = centre - (centre @ up) * up
    means = scene.means - sideways - heights.min() * up
    return GaussianScene(means, scene.scales, scene.quaternions, scene.opacities, scene.sh, scene.ids, scene.extras)
