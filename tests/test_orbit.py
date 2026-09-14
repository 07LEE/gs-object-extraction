import numpy as np
import pytest
from gs_object_extraction.app.orbit import AXES, DEFAULT_UP, PITCH_LIMIT, Orbit, estimate_up, unproject

UPS = [*AXES.values(), (.1, -.9, -.4)]


def project(camera, point):
    p = camera.world_to_camera @ np.r_[point, 1.]
    return np.array([camera.fx * p[0] / p[2] + camera.cx, camera.fy * p[1] / p[2] + camera.cy]), p[2]


@pytest.mark.parametrize("up", UPS)
def test_target_sits_at_the_image_centre_and_up_points_image_up(up):
    orbit = Orbit(np.array([.3, -.2, 1.]), 4., yaw=.7, pitch=.4, up=up)
    camera = orbit.camera(200, 100)
    uv, depth = project(camera, orbit.target)
    np.testing.assert_allclose(uv, [camera.cx, camera.cy], atol=1e-9)
    assert depth == pytest.approx(4.)
    above, _ = project(camera, orbit.target + .1 * orbit.up)
    assert above[1] < camera.cy


def test_rotate_keeps_distance_and_clamps_pitch_short_of_the_pole():
    orbit = Orbit(np.zeros(3), 2.)
    orbit.rotate(120, 40)
    assert np.linalg.norm(orbit.eye - orbit.target) == pytest.approx(2.)
    orbit.rotate(0, 1e6)
    assert orbit.pitch == pytest.approx(PITCH_LIMIT)
    orbit.camera(64, 64)  # still a valid camera at the limit


def test_pan_moves_points_at_target_depth_with_the_mouse():
    orbit = Orbit(np.array([1., 2., 3.]), 5., yaw=.3, pitch=.2)
    before = orbit.target.copy()
    orbit.pan(30, -12, height=240)
    camera = orbit.camera(320, 240)
    uv, _ = project(camera, before)
    np.testing.assert_allclose(uv, [camera.cx + 30, camera.cy - 12], atol=1e-6)


def test_zoom_and_changing_up_keep_the_camera_in_place():
    orbit = Orbit(np.zeros(3), 10.)
    orbit.zoom(2)
    assert orbit.distance == pytest.approx(8.1)
    eye = orbit.eye.copy()
    orbit.set_up(AXES["+Z"])
    np.testing.assert_allclose(orbit.eye, eye, atol=1e-9)
    np.testing.assert_allclose(orbit.up, AXES["+Z"])
    with pytest.raises(ValueError):
        orbit.set_up((0, 0, 0))


def test_framing_starts_close_to_the_median_gaussian():
    rng = np.random.default_rng(0)
    means = np.vstack([rng.uniform(-1, 1, (5000, 3)), [[500., 500., 500.]]])
    orbit = Orbit.framing(means)
    assert np.linalg.norm(orbit.target) < .1 and .4 < orbit.distance < .8


def tilted_floor_scene(normal, above=True, seed=0):
    """A wide thin floor with walls and an object standing on one side of it."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.cross(normal, [1., 0., 0.])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    floor = rng.uniform(-3, 3, (6000, 2)) @ np.stack([a, b]) + rng.normal(0, .01, (6000, 1)) * normal
    standing = rng.uniform(-.3, .3, (800, 2)) @ np.stack([a, b]) + rng.uniform(0, 1.5, (800, 1)) * normal
    return np.vstack([floor, standing if above else -standing])


def test_estimate_up_is_the_floor_normal_on_the_side_things_stand():
    normal = np.array([.1, -.9, -.4])
    normal /= np.linalg.norm(normal)
    assert estimate_up(tilted_floor_scene(normal)) @ normal > .99
    assert estimate_up(tilted_floor_scene(normal, above=False)) @ normal < -.99


def test_estimate_up_falls_back_without_a_clear_plane():
    blob = np.random.default_rng(1).normal(0, 1, (3000, 3))
    np.testing.assert_allclose(estimate_up(blob), DEFAULT_UP)


def test_look_from_keeps_the_camera_and_centres_the_new_target():
    orbit = Orbit(np.zeros(3), 6., yaw=.4, pitch=.3, up=AXES["+Z"])
    eye = orbit.eye.copy()
    point = np.array([.5, -.3, .2])
    orbit.look_from(eye, point)
    np.testing.assert_allclose(orbit.eye, eye, atol=1e-9)
    camera = orbit.camera(160, 120)
    uv, _ = project(camera, point)
    np.testing.assert_allclose(uv, [camera.cx, camera.cy], atol=1e-6)


def test_unproject_inverts_projection():
    camera = Orbit(np.array([.2, .1, 0.]), 3., yaw=1., pitch=-.2).camera(200, 150)
    point = np.array([.4, -.25, .3])
    (u, v), depth = project(camera, point)
    np.testing.assert_allclose(unproject(camera, u, v, depth), point, atol=1e-9)
