import pytest
import numpy as np
from gs_object_extraction.app.pick import inside_box, project
from gs_object_extraction.camera import Camera

CAMERA = Camera.look_at((0, 0, 0), (0, 0, 2), width=100, height=80)


def test_a_point_on_the_axis_lands_in_the_middle_of_the_frame():
    x, y, depth = project(CAMERA, [[0, 0, 2]])
    np.testing.assert_allclose([x[0], y[0], depth[0]], [CAMERA.cx, CAMERA.cy, 2])


def test_the_box_takes_what_lands_inside_at_any_depth_and_in_any_corner_order():
    near, far, aside = [0, 0, 1], [0, 0, 5], [1, 0, 2]
    means = np.array([near, far, aside])
    hit = inside_box(CAMERA, means, (60, 50, 40, 30))  # corners given the wrong way round
    np.testing.assert_array_equal(hit, [True, True, False])


def test_what_is_behind_the_camera_is_never_taken():
    means = np.array([[0, 0, -3], [0, 0, 3]])
    np.testing.assert_array_equal(inside_box(CAMERA, means, (0, 0, 100, 80)), [False, True])


def test_the_box_holds_every_point_stands_on_the_up_axis_and_turns_to_fit():
    from gs_object_extraction.app.pick import EDGES, bounding_box
    rng = np.random.default_rng(3)
    points = rng.uniform((-2, -.5, 0), (2, .5, 1), (500, 3))  # long in x, thin in y, standing on z
    turn = np.deg2rad(35)
    rotation = np.array([[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]])
    corners, size = bounding_box(points @ rotation.T, (0, 0, 1))
    np.testing.assert_allclose(sorted(size), sorted([4, 1, 1]), atol=.05)
    assert size[2] == pytest.approx(points[:, 2].max() - points[:, 2].min())  # the up axis is the box's height
    assert corners.shape == (8, 3) and len(EDGES) == 12
    axes = np.stack([corners[1] - corners[0], corners[2] - corners[0], corners[4] - corners[0]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    inside = (points @ rotation.T - corners[0]) @ axes.T
    extent = np.linalg.norm([corners[1] - corners[0], corners[2] - corners[0], corners[4] - corners[0]], axis=1)
    assert (inside >= -1e-9).all() and (inside <= extent + 1e-9).all()


def test_one_stray_far_away_widens_the_box():
    from gs_object_extraction.app.pick import bounding_box
    rng = np.random.default_rng(4)
    body = rng.uniform(-.5, .5, (300, 3))
    _, tight = bounding_box(body, (0, 0, 1))
    _, loose = bounding_box(np.vstack([body, [[6, 0, 0]]]), (0, 0, 1))
    assert loose.max() > 5 * tight.max() / 2


def test_a_single_point_gives_a_box_of_no_size():
    from gs_object_extraction.app.pick import bounding_box
    corners, size = bounding_box([[1, 2, 3]], (0, 1, 0))
    np.testing.assert_allclose(size, 0)
    np.testing.assert_allclose(corners, np.tile([1, 2, 3], (8, 1)))
