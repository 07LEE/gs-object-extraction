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
