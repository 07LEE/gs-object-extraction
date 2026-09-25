import numpy as np
import pytest
from gs_object_extraction.app.orbit import (AXES, DEFAULT_UP, PITCH_LIMIT, Orbit, estimate_up, frame_scene,
                                            unproject)

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


class Scanned:
    """A scene the camera can only see across from far enough, and only up to a point."""

    def __init__(self, centre, sees_across_from, empty_beyond=None):
        self.centre = np.asarray(centre, dtype=float)
        self.sees_across_from, self.empty_beyond = sees_across_from, empty_beyond
        self.tried = []

    def depth_image(self, camera, *, active=None):
        rotation, translation = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
        distance = float(np.linalg.norm(-rotation.T @ translation - self.centre))
        self.tried.append(distance)
        shape = (camera.height, camera.width)
        if self.empty_beyond is not None and distance > self.empty_beyond:
            return np.full(shape, distance), np.zeros(shape)  # the scene has fallen out of the frame
        near = distance if distance >= self.sees_across_from else .2 * distance
        return np.full(shape, near), np.ones(shape)


def test_framing_stands_as_far_back_as_the_scene_allows():
    means = np.random.default_rng(0).normal(0, 1, (500, 3))
    spread = float(np.median(np.linalg.norm(means - np.median(means, axis=0), axis=1)))
    # Blocked up close, open from 1.5 medians out, and the scene stays in frame throughout.
    renderer = Scanned(np.median(means, axis=0), sees_across_from=1.4 * spread)
    assert frame_scene(renderer, means).distance == pytest.approx(4. * spread)
    # Open everywhere, but past 2 medians the frame is mostly empty.
    renderer = Scanned(np.median(means, axis=0), sees_across_from=0., empty_beyond=2. * spread)
    assert frame_scene(renderer, means).distance == pytest.approx(1.5 * spread)


def test_framing_falls_back_when_no_distance_works():
    means = np.random.default_rng(1).normal(0, 1, (500, 3))
    spread = float(np.median(np.linalg.norm(means - np.median(means, axis=0), axis=1)))
    renderer = Scanned(np.median(means, axis=0), sees_across_from=1e9)  # blocked wherever it stands
    orbit = frame_scene(renderer, means)
    assert orbit.distance == pytest.approx(.6 * spread)
    assert renderer.tried and max(renderer.tried) == pytest.approx(4. * spread)  # it did try the whole range


def test_looking_around_turns_the_view_while_the_camera_stays_where_it_is():
    orbit = Orbit(np.array([1., 2., 3.]), 4., yaw=.7, pitch=.3, up=(0, 0, 1))
    eye, distance = orbit.eye.copy(), orbit.distance
    forward = orbit.target - orbit.eye
    orbit.look(60, -25)
    np.testing.assert_allclose(orbit.eye, eye, atol=1e-9)
    assert orbit.distance == distance and not np.allclose(orbit.target - orbit.eye, forward)
    assert np.isclose(np.linalg.norm(orbit.target - orbit.eye), distance)
    orbit.look(-60, 25)  # the same drag backwards turns the view back
    np.testing.assert_allclose(orbit.target - orbit.eye, forward, atol=1e-9)


def test_looking_follows_the_drag_right_turns_right_and_down_turns_down():
    orbit = Orbit(np.zeros(3), 3., yaw=.4, pitch=.2, up=(0, 0, 1))
    rotation = orbit.camera(100, 100).world_to_camera[:3, :3]
    right, up = rotation[0].copy(), -rotation[1]  # the camera's image-right and image-up, in the world
    forward = rotation[2].copy()
    orbit.look(40, 0)  # drag right
    turned = orbit.camera(100, 100).world_to_camera[:3, 2]
    assert (turned - forward) @ right > 0
    orbit.look(-40, 0)
    orbit.look(0, 40)  # drag down
    turned = orbit.camera(100, 100).world_to_camera[:3, 2]
    assert (turned - forward) @ up < 0


def test_the_pitch_of_a_look_is_limited():
    orbit = Orbit(np.zeros(3), 3., yaw=0., pitch=0., up=(0, 0, 1))
    orbit.look(0, 1e6)
    assert abs(orbit.pitch) <= np.deg2rad(89.) + 1e-9
    orbit.look(0, -1e7)
    assert abs(orbit.pitch) <= np.deg2rad(89.) + 1e-9
