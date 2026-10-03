import numpy as np
import pytest

from gs_object_extraction.scene import GaussianScene
from gs_object_extraction.upright import _basis, rotated, rotation_between, stood_up


def random_scene(n=20, k=16, seed=0):
    rng = np.random.default_rng(seed)
    quaternions = rng.normal(size=(n, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return GaussianScene(rng.normal(size=(n, 3)), rng.uniform(.1, 1, (n, 3)), quaternions,
                         rng.uniform(.1, 1, n), rng.normal(size=(n, k, 3)))


def colour(sh, direction):
    """View-dependent colour of every Gaussian seen along one unit direction."""
    out = sh[:, 0, :] * 0.28209479177387814
    for degree, start in ((1, 1), (2, 4), (3, 9)):
        if sh.shape[1] >= start + 2 * degree + 1:
            out = out + np.einsum("k,nkc->nc", _basis(direction[None], degree)[0], sh[:, start:start + 2 * degree + 1, :])
    return out


@pytest.mark.parametrize("source", [(0., 1., 0.), (0.3, -0.8, 0.5), (0., -1., 0.), (1., 0., 0.)])
def test_rotation_takes_the_source_to_the_target_and_is_proper(source):
    target = np.array([0., -1., 0.])
    rotation = rotation_between(source, target)
    np.testing.assert_allclose(rotation @ (np.array(source) / np.linalg.norm(source)), target, atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


@pytest.mark.parametrize("k", [1, 4, 9, 16])
def test_colour_seen_along_a_turned_direction_matches_the_original(k):
    scene = random_scene(k=k)
    rotation = rotation_between((0.3, -0.8, 0.5), (0., -1., 0.))
    turned = rotated(scene, rotation, scene.means.mean(axis=0))
    for direction in np.random.default_rng(1).normal(size=(6, 3)):
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(colour(turned.sh, rotation @ direction), colour(scene.sh, direction), atol=1e-9)


def test_covariances_turn_with_the_scene():
    scene = random_scene()
    rotation = rotation_between((0.3, -0.8, 0.5), (0., -1., 0.))
    turned = rotated(scene, rotation, (0., 0., 0.))
    expected = np.einsum("ij,njk,lk->nil", rotation, scene.covariances, rotation)
    np.testing.assert_allclose(turned.covariances, expected, atol=1e-9)
    np.testing.assert_allclose(turned.means, scene.means @ rotation.T, atol=1e-12)


def test_stood_up_points_the_up_axis_at_the_target_about_the_centre():
    scene = random_scene()
    up, target = np.array([0.3, -0.8, 0.5]), np.array([0., -1., 0.])
    turned = stood_up(scene, up, target)
    np.testing.assert_allclose(turned.means.mean(axis=0), scene.means.mean(axis=0), atol=1e-12)
    rotation = rotation_between(up, target)
    np.testing.assert_allclose((turned.means - turned.means.mean(axis=0)) @ rotation, scene.means - scene.means.mean(axis=0), atol=1e-12)
    np.testing.assert_array_equal(turned.ids, scene.ids)
    assert len(stood_up(scene.subset(np.zeros(len(scene), bool)), up, target)) == 0


@pytest.mark.parametrize("up", [(0., -1., 0.), (0.3, -0.8, 0.5)])
def test_grounded_puts_the_lowest_point_and_the_centre_on_the_origin(up):
    from gs_object_extraction.upright import grounded
    scene = random_scene()
    scene.means += (4., -7., 2.)
    moved = grounded(scene, up)
    direction = np.array(up) / np.linalg.norm(up)
    assert (moved.means @ direction).min() == pytest.approx(0, abs=1e-12)
    centre = moved.means.mean(axis=0)
    np.testing.assert_allclose(centre - (centre @ direction) * direction, 0, atol=1e-12)
    shift = moved.means - scene.means
    np.testing.assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-12)  # moved, not reshaped
    np.testing.assert_array_equal(moved.sh, scene.sh)
    assert len(grounded(scene.subset(np.zeros(len(scene), bool)), up)) == 0


@pytest.mark.parametrize("degree, count", [(0, 1), (1, 4), (2, 9), (3, 16)])
def test_with_sh_degree_keeps_the_low_terms_and_everything_else(degree, count):
    scene = random_scene()
    reduced = scene.with_sh_degree(degree)
    np.testing.assert_array_equal(reduced.sh, scene.sh[:, :count])
    np.testing.assert_array_equal(reduced.means, scene.means)
    np.testing.assert_array_equal(reduced.ids, scene.ids)
    assert scene.sh.shape[1] == 16  # the original is untouched
    assert random_scene(k=4).with_sh_degree(2).sh.shape[1] == 4
