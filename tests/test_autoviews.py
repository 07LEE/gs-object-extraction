"""Marking a ring of views around the object without a click in each one."""

import os
import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from gs_object_extraction.app.autoviews import (AutoMarkJob, DISTANCE, MAX_FRAME, MIN_PIXELS, NEAR, VIEWS,
                                                best_candidate, inner_part, object_frame, prompt_points, ring,
                                                surface_points, usable)
from gs_object_extraction.app.orbit import Orbit
from gs_object_extraction.app.views import MaskedView
from gs_object_extraction.camera import Camera

SIZE = (64, 48)
CAMERA = Camera.look_at((0, 0, -4), (0, 0, 0), width=SIZE[0], height=SIZE[1])


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def disk(shape, centre, radius):
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    return (ys - centre[0]) ** 2 + (xs - centre[1]) ** 2 <= radius ** 2


def test_ring_circles_the_object_at_one_height():
    up = np.array([0., 1., 0.])
    orbits = ring(np.array([1., 2., 3.]), 2., up, count=8, pitch=.2)
    assert len(orbits) == 8
    assert {round(o.pitch, 6) for o in orbits} == {.2}
    assert [round(o.yaw, 6) for o in orbits] == [round(i * 2 * np.pi / 8, 6) for i in range(8)]
    assert all(np.isclose(o.distance, DISTANCE * 2.) for o in orbits)
    eyes = np.array([o.eye for o in orbits])
    assert len(np.unique(eyes.round(6), axis=0)) == 8  # every view looks from somewhere else
    with pytest.raises(ValueError):
        ring(np.zeros(3), 1., up, count=0)


def test_object_frame_ignores_a_few_strays():
    points = np.concatenate([np.random.default_rng(0).normal(0, .1, (500, 3)), np.full((3, 3), 50.)])
    centre, radius = object_frame(points)
    assert np.allclose(centre, 0, atol=.05) and radius < 1
    assert object_frame(np.zeros((4, 3)))[1] > 0  # a radius is never zero


def test_surface_points_land_on_what_the_mask_covers():
    mask = np.zeros(SIZE[::-1], bool)
    mask[20:28, 30:38] = True

    class Renderer:
        def depth_image(self, camera, *, active=None):
            return np.full(SIZE[::-1], 4.), np.ones(SIZE[::-1])

    points = surface_points(Renderer(), [(CAMERA, mask)], step=1)
    assert len(points) == mask.sum()
    assert np.allclose(points[:, 2], 0, atol=1e-6)  # four in front of a camera four back is the origin plane
    centre, radius = object_frame(points)
    assert np.allclose(centre[2], 0, atol=1e-6) and radius > 0


def test_surface_points_need_something_solid():
    class Empty:
        def depth_image(self, camera, *, active=None):
            return np.full(SIZE[::-1], 4.), np.zeros(SIZE[::-1])

    with pytest.raises(ValueError, match="nothing solid"):
        surface_points(Empty(), [(CAMERA, np.ones(SIZE[::-1], bool))])


def test_prompts_sit_inside_the_outline_and_spread_out():
    coverage = disk(SIZE[::-1], (24, 32), 15)
    points = prompt_points(coverage, count=3)
    assert len(points) == 3
    assert all(coverage[y, x] for x, y in points)
    inner = inner_part(coverage)
    assert inner.sum() < coverage.sum() and all(inner[y, x] for x, y in points)
    spread = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in points for b in points)
    assert spread > 10  # not three clicks on the same spot


def test_prompts_avoid_a_hole_in_the_middle():
    coverage = disk(SIZE[::-1], (24, 32), 18) & ~disk(SIZE[::-1], (24, 32), 12)
    points = prompt_points(coverage, count=2)
    assert all(coverage[y, x] for x, y in points)


def test_prompts_survive_a_thin_silhouette():
    coverage = np.zeros(SIZE[::-1], bool)
    coverage[24, 10:54] = True  # one pixel tall: erosion would wipe it out
    points = prompt_points(coverage, count=3)
    assert all(coverage[y, x] for x, y in points)


def test_the_candidate_closest_to_the_silhouette_wins():
    coverage = disk(SIZE[::-1], (24, 32), 10)
    candidates = [disk(SIZE[::-1], (24, 32), 3), disk(SIZE[::-1], (24, 32), 10), np.ones(SIZE[::-1], bool)]
    mask, overlap = best_candidate(candidates, coverage)
    assert np.array_equal(mask, candidates[1]) and overlap > .9


def test_masks_that_miss_the_object_or_run_off_are_refused():
    coverage = disk(SIZE[::-1], (24, 32), 10)
    assert usable(coverage, coverage)
    assert not usable(None, coverage)
    assert not usable(np.zeros(SIZE[::-1], bool), coverage)
    assert not usable(disk(SIZE[::-1], (24, 32), 2), coverage)  # covers too little of it
    assert not usable(np.ones(SIZE[::-1], bool), coverage)  # swallows the frame
    grown = disk(SIZE[::-1], (24, 32), 10) | disk(SIZE[::-1], (10, 10), 12) | disk(SIZE[::-1], (40, 55), 12)
    assert grown.mean() < MAX_FRAME and not usable(grown, coverage)  # runs far past the outline


class Renderer:
    """Draws the object as a disk, and everything else as the wall behind it."""

    n = 6  # the first three Gaussians are the object

    def __init__(self, *, fail_from=None):
        self.fail_from = fail_from
        self.lifts = self.images = 0

    def lift(self, camera, labels, *, active=None):
        from gs_object_extraction.renderer import Lifted
        self.lifts += 1
        inside = np.array([1., 1., 1., 0., 0., 0.])
        outside = np.array([0., 0., 0., 1., 1., 1.])
        return Lifted(inside, outside, inside + outside)

    def depth_image(self, camera, *, active=None):
        return np.full((camera.height, camera.width), 4.), np.ones((camera.height, camera.width))

    def alpha_image(self, camera, *, active=None):
        if self.fail_from is not None and self.images >= self.fail_from:
            return np.zeros((camera.height, camera.width))  # the object is out of sight from here
        return disk((camera.height, camera.width), (camera.height // 2, camera.width // 2), 12).astype(float)

    def render_image(self, camera, *, active=None, background=(0, 0, 0)):
        self.images += 1
        return np.zeros((camera.height, camera.width, 3), np.uint8)


class Segmenter:
    """Returns the object, a part of it and the whole frame, as SAM2 would for one prompt."""

    def __init__(self):
        self.keys = []

    def candidates(self, image, key, points, labels, *, several=None):
        self.keys.append(key)
        shape = image.shape[:2]
        centre = (shape[0] // 2, shape[1] // 2)
        masks = np.stack([disk(shape, centre, 3), disk(shape, centre, 12), np.ones(shape, bool)])
        return masks, np.array([.9, .5, .95])  # the best mask is not the best scoring one


def marked_view():
    mask = disk(SIZE[::-1], (24, 32), 12)
    return MaskedView(CAMERA, mask, ((32, 24),), (1,))


def run_job(app, job, timeout=10):
    import time
    results = {}
    job.succeeded.connect(lambda marked, skipped: results.update(marked=marked, skipped=skipped))
    job.failed.connect(lambda message: results.update(failed=message))
    job.cancelled.connect(lambda: results.update(cancelled=True))
    job.progress.connect(lambda done, total: results.setdefault("progress", []).append((done, total)))
    job.start()
    deadline = time.monotonic() + timeout
    while job.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    job.wait(1000)
    app.processEvents()
    return results


def test_the_job_marks_a_ring_and_keeps_the_mask_that_matches(app):
    renderer, segmenter = Renderer(), Segmenter()
    means = np.concatenate([np.zeros((3, 3)), np.full((3, 3), 20.)])  # object at the origin, wall far away
    job = AutoMarkJob(renderer, segmenter, means, [marked_view()], (0., 1., 0.), SIZE, count=4)
    results = run_job(app, job)
    assert "failed" not in results and len(results["marked"]) == 4 and results["skipped"] == 0
    assert results["progress"][-1] == (4, 4)
    assert len(set(segmenter.keys)) == 4  # a fresh embedding per view, never a stale one
    view = results["marked"][0]
    assert view.mask.shape == (view.camera.height, view.camera.width)
    assert int(view.mask.sum()) == int(disk(SIZE[::-1], (24, 32), 12).sum())  # the middle candidate, not the best score
    assert len(view.points) == 3 and set(view.labels) == {1}


def test_views_that_show_nothing_are_skipped(app):
    renderer, segmenter = Renderer(fail_from=2), Segmenter()
    means = np.concatenate([np.zeros((3, 3)), np.full((3, 3), 20.)])
    job = AutoMarkJob(renderer, segmenter, means, [marked_view()], (0., 1., 0.), SIZE, count=4)
    results = run_job(app, job)
    assert len(results["marked"]) + results["skipped"] == 4 and results["skipped"] >= 1


def test_an_empty_selection_is_reported(app):
    class Nothing(Renderer):
        def lift(self, camera, labels, *, active=None):
            from gs_object_extraction.renderer import Lifted
            return Lifted.zeros(self.n)

    job = AutoMarkJob(Nothing(), Segmenter(), np.zeros((6, 3)), [marked_view()], (0., 1., 0.), SIZE, count=4)
    results = run_job(app, job)
    assert "marked" not in results and "select any Gaussian" in results["failed"]


def test_the_window_adds_the_marked_ring_to_its_views(app):
    """The button is only offered once a view is marked, and its result lands in the list."""
    from gs_object_extraction.app.window import MainWindow
    import time
    window = MainWindow()
    window.resize(*SIZE)
    window.show()
    app.processEvents()
    assert not window.auto_button.isEnabled()  # nothing marked yet
    window.scene = type("Scene", (), {"means": np.concatenate([np.zeros((3, 3)), np.full((3, 3), 20.)])})()
    window.viewport.renderer = Renderer()
    window.viewport.segmenter = Segmenter()
    window.views.append(marked_view())
    window.view_list.addItem("View 1")
    window.stages = {"selected": np.ones(6, bool), "cleaned": np.ones(6, bool)}
    window.update_extraction_state()
    assert window.auto_button.isEnabled()

    window.start_auto_mark()
    assert window.auto_job is not None and window.viewport.suspended
    assert not window.auto_button.isEnabled() and not window.extract_action.isEnabled()
    deadline = time.monotonic() + 10
    while window.auto_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    assert window.auto_job is None and not window.viewport.suspended
    assert len(window.views) == 1 + VIEWS and window.view_list.count() == 1 + VIEWS
    assert window.stages is None  # the old result no longer matches the views
    assert "Marked" in window.statusBar().currentMessage()
    assert window.auto_button.isEnabled() and not window.progress.isVisible()


def test_cancelling_stops_the_ring(app):
    import threading
    import time

    class Blocking(Segmenter):
        def __init__(self):
            super().__init__()
            self.started, self.release = threading.Event(), threading.Event()

        def candidates(self, image, key, points, labels, *, several=None):
            self.started.set()
            assert self.release.wait(5), "the test did not release the worker"
            return super().candidates(image, key, points, labels, several=several)

    renderer, segmenter = Renderer(), Blocking()
    means = np.concatenate([np.zeros((3, 3)), np.full((3, 3), 20.)])
    job = AutoMarkJob(renderer, segmenter, means, [marked_view()], (0., 1., 0.), SIZE, count=VIEWS)
    cancelled = []
    job.cancelled.connect(lambda: cancelled.append(True))
    job.start()
    assert segmenter.started.wait(5)
    job.requestInterruption()
    segmenter.release.set()
    deadline = time.monotonic() + 5
    while job.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    job.wait(1000)
    app.processEvents()
    assert cancelled and len(segmenter.keys) < VIEWS  # it stopped instead of finishing the ring
