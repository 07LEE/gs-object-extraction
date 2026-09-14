import numpy as np
import pytest
from gs_object_extraction.app.orbit import AXES, PITCH_LIMIT, Orbit


def project(camera, point):
    p = camera.world_to_camera @ np.r_[point, 1.]
    return np.array([camera.fx * p[0] / p[2] + camera.cx, camera.fy * p[1] / p[2] + camera.cy]), p[2]


@pytest.mark.parametrize("up", list(AXES))
def test_target_sits_at_the_image_centre_and_up_points_image_up(up):
    orbit = Orbit(np.array([.3, -.2, 1.]), 4., yaw=.7, pitch=.4, up=up)
    camera = orbit.camera(200, 100)
    uv, depth = project(camera, orbit.target)
    np.testing.assert_allclose(uv, [camera.cx, camera.cy], atol=1e-9)
    assert depth == pytest.approx(4.)
    above, _ = project(camera, orbit.target + .1 * np.asarray(AXES[up]))
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


def test_zoom_and_up_axis_changes():
    orbit = Orbit(np.zeros(3), 10.)
    orbit.zoom(2)
    assert orbit.distance == pytest.approx(8.1)
    orbit.set_up("+Z")
    assert orbit.up == "+Z" and orbit.yaw == 0
    with pytest.raises(ValueError):
        orbit.set_up("up")


def test_framing_ignores_far_floaters():
    rng = np.random.default_rng(0)
    means = np.vstack([rng.uniform(-1, 1, (5000, 3)), [[500., 500., 500.]]])
    orbit = Orbit.framing(means)
    assert np.linalg.norm(orbit.target) < .1 and orbit.distance < 10
