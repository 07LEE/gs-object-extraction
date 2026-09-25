"""Exercise the extraction workflow through the GUI with real worker threads."""

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from gs_object_extraction.app.orbit import Orbit
from gs_object_extraction.extract import TRIM
from gs_object_extraction.app.views import MaskedView
from gs_object_extraction.app.viewport import BACKGROUND
from gs_object_extraction.app.window import MainWindow
from gs_object_extraction.camera import Camera
from gs_object_extraction.ply import load_ply, save_ply
from gs_object_extraction.renderer import Exclusive, Lifted
from gs_object_extraction.scene import GaussianScene


# The fourth Gaussian has an off-mask piece hidden by the fifth Gaussian until
# the scene is removed. All footprints are outside the ignored boundary band.
MASK = np.zeros((9, 30), bool)
MASK[:, 10:23] = True
CAMERA = Camera.look_at((0, 0, 0), (0, 0, 2), width=30, height=9)
FOOTPRINTS = ({12: 1., 16: 1.}, {20: 1.}, {2: 1., 5: 1.}, {16: .5, 27: 1.}, {27: 1.})
SELECTED = np.array([True, True, False, True, False])
CLEANED = np.array([True, True, False, False, False])


class FakeRenderer(Exclusive):
    def __init__(self, *, blocked=False, fail=False, empty=False, n=5):
        super().__init__()
        self.n = n
        self.blocked, self.fail, self.empty = blocked, fail, empty
        self.block_at = {1}  # lift calls that wait for the test
        self.started, self.release = threading.Event(), threading.Event()
        self.lift_threads, self.image_calls, self.depth_calls = [], [], []

    def lift(self, camera, labels, *, active=None):
        with self.held():  # as the real renderer does, so the window sees it busy
            return self._lift(camera, labels, active)

    def _lift(self, camera, labels, active):
        self.lift_threads.append(threading.get_ident())
        if self.blocked and len(self.lift_threads) in self.block_at:
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("test did not release the worker")
        if self.fail:
            raise RuntimeError("test renderer failed during lifting")
        assert labels.shape == (camera.height, camera.width) == MASK.shape
        active = np.ones(self.n, bool) if active is None else active
        weights = np.zeros((self.n, *MASK.shape))
        if not self.empty:
            for index, footprint in enumerate(FOOTPRINTS):
                for x, weight in footprint.items():
                    if active[index] and not (index == 3 and x == 27 and active[4]):
                        weights[index, 4, x] = weight
        return Lifted(weights[:, labels == 1].sum(1), weights[:, labels == 0].sum(1),
                      weights.sum(axis=(1, 2)))

    def update_scales(self, scales):
        self.scales = scales.copy()

    def render_image(self, camera, *, active=None, background=BACKGROUND):
        self.image_calls.append((None if active is None else active.copy(), tuple(background)))
        pixels = np.empty((camera.height, camera.width, 3), np.uint8)
        pixels[:] = np.round(np.asarray(background) * 255).astype(np.uint8)
        pixels[camera.height // 2, camera.width // 2] = [90, 120, 150]
        return pixels

    def depth_image(self, camera, *, active=None):
        self.depth_calls.append(None if active is None else active.copy())
        shape = (camera.height, camera.width)
        return np.full(shape, 2.), np.ones(shape)


def drag(view, start, end, button, modifiers=Qt.NoModifier):
    def mouse(kind, x, y, buttons):
        return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, buttons, modifiers)

    view.mousePressEvent(mouse(QEvent.MouseButtonPress, *start, button))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, *end, button))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, *end, Qt.NoButton))


def make_scene():
    rng = np.random.default_rng(41)
    rotations = np.zeros((5, 4))
    rotations[:, 0] = 1
    return GaussianScene(rng.normal(0, .2, (5, 3)), np.full((5, 3), .1), rotations,
                         np.linspace(.2, .9, 5), rng.normal(0, .1, (5, 16, 3)),
                         ids=np.array([41, 9, 101, 12, 17]),
                         extras={"confidence": rng.random(5),
                                 "category": np.arange(5, dtype=np.uint16),
                                 "nx": rng.random(5).astype(np.float32)})


def wait_until(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    app.processEvents()
    assert predicate(), "GUI worker did not reach the expected state before the timeout"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(monkeypatch):
    messages = []
    for name in ("warning", "critical"):
        monkeypatch.setattr(f"gs_object_extraction.app.window.QMessageBox.{name}",
                            lambda parent, title, message: messages.append((title, message)))
    return messages


@pytest.fixture
def make_window(app, tmp_path, dialogs):
    windows = []

    def create(renderer=None):
        renderer = renderer or FakeRenderer()
        window = MainWindow()
        windows.append((window, renderer))
        window.resize(640, 480)
        window.show()
        app.processEvents()
        window.scene = make_scene()
        window.source_path = tmp_path / f"source_{len(windows)}.ply"
        save_ply(window.scene, window.source_path)
        window.scene_renderer = renderer
        window.renderer_factory = lambda scene: FakeRenderer(n=len(scene))
        window.viewport.set_scene(renderer, Orbit(np.zeros(3), 3.))
        window.viewport.render_now()
        window.viewport._timer.stop()
        window.views.append(MaskedView(CAMERA, MASK.copy(), ((12, 4),), (1,)))
        window.view_list.addItem(f"View 1: {int(MASK.sum()):,} px")
        window.view_list.setCurrentRow(0)
        window.update_extraction_state()
        return window, renderer

    yield create

    # A failed assertion must never leave a live QThread owned by a lost window.
    for window, renderer in windows:
        renderer.release.set()
        window.cancel_extraction()
        wait_until(app, lambda: window.extraction_job is None)
        wait_until(app, lambda: window.export_job is None)
        window.close()
    app.processEvents()


def extract_object(app, window):
    window.start_extraction()
    assert window.extraction_job is not None
    wait_until(app, lambda: window.extraction_job is None)


def export_object(app, window, path):
    """Run one export to completion, returning whether it started."""
    started = window.export_ply(path)
    if started:
        wait_until(app, lambda: window.export_job is None)
    return started


def test_extract_cleans_hidden_fragments_previews_and_exports_full_gaussians(app, make_window, tmp_path, dialogs):
    window, renderer = make_window()
    original = window.scene.copy()
    extract_object(app, window)

    np.testing.assert_array_equal(window.stages["selected"], SELECTED)
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)
    assert renderer.lift_threads and all(t != threading.get_ident() for t in renderer.lift_threads)
    assert "1 removed" in window.result_label.text()
    assert window.preview_box.currentText() == "Object only"
    assert window.export_action.isEnabled() and window.select_action.isEnabled()
    window.set_selecting(True)  # in the object preview this selects Gaussians; it does not mark a view
    window.viewport.add_point(10, 10, 1)
    assert window.viewport.selecting and not window.viewport.points
    window.set_selecting(False)

    # The object is drawn from Gaussians of its own, so the whole of it is on screen.
    assert window.viewport.renderer is not renderer and len(window.object_scene) == int(CLEANED.sum())
    np.testing.assert_array_equal(window.object_scene.ids, original.ids[CLEANED])
    object_renderer, object_scene = window.viewport.renderer, window.object_scene
    for background, color in (("Black", 0), ("White", 255), ("Black", 0)):
        window.background_box.setCurrentText(background)
        window.viewport.render_now()
        assert window.viewport.renderer is object_renderer
        assert window.object_scene is object_scene
        active, rgb = object_renderer.image_calls[-1]
        np.testing.assert_array_equal(active, np.ones(int(CLEANED.sum()), bool))
        assert rgb == (color / 255,) * 3
        np.testing.assert_array_equal(window.viewport.pixels[0, 0], [color] * 3)
        assert window.viewport.pick(0, 0) is not None
        np.testing.assert_array_equal(object_renderer.depth_calls[-1], np.ones(int(CLEANED.sum()), bool))

    window.preview_box.setCurrentText("Scene")
    window.viewport.render_now()
    assert window.viewport.renderer is renderer and window.object_scene is None
    assert renderer.image_calls[-1] == (None, BACKGROUND)
    assert window.select_action.isEnabled() and not window.background_box.isEnabled()
    assert window.export_action.isEnabled()
    assert export_object(app, window, tmp_path / "object.ply")
    exported = load_ply(tmp_path / "object.ply")
    for name in ("means", "scales", "quaternions", "opacities", "sh"):
        np.testing.assert_allclose(getattr(exported, name), getattr(original, name)[CLEANED],
                                   rtol=3e-7, atol=3e-7)
        np.testing.assert_array_equal(getattr(window.scene, name), getattr(original, name))
    np.testing.assert_array_equal(exported.ids, original.ids[CLEANED])
    assert exported.extras.keys() == original.extras.keys()
    for name, values in original.extras.items():
        np.testing.assert_array_equal(exported.extras[name], values[CLEANED])
        assert exported.extras[name].dtype == values.dtype
    assert not dialogs


class EdgeRenderer(FakeRenderer):
    """The first Gaussian reaches past the masks, but not far enough to be dropped."""

    def lift(self, camera, labels, *, active=None):
        lifted = super().lift(camera, labels, active=active)
        if active is not None:  # the cleaning round is where the off-mask shares come from
            lifted.inside[0], lifted.outside[0] = .8, .2
        return lifted


@pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1",
                    reason="set GS_OBJECT_EXTRACTION_TEST_CUDA=1 in the GPU environment")
def test_background_reuse_matches_fresh_cuda_renderer(app, make_window):
    from gs_object_extraction.renderer import GsplatRenderer

    window, _ = make_window()
    window.renderer_factory = GsplatRenderer
    extract_object(app, window)
    renderer, scene = window.viewport.renderer, window.object_scene
    buffers = [getattr(renderer, name).data_ptr()
               for name in ("means", "scales", "rotations", "opacities", "sh")]
    reference = GsplatRenderer(scene)
    for name, background in (("Black", (0., 0., 0.)), ("White", (1., 1., 1.))):
        window.background_box.setCurrentText(name)
        window.viewport.render_now()
        assert window.viewport.renderer is renderer and window.object_scene is scene
        assert buffers == [getattr(renderer, key).data_ptr()
                           for key in ("means", "scales", "rotations", "opacities", "sh")]
        expected = reference.render_image(window.viewport.camera, background=background)
        np.testing.assert_array_equal(window.viewport.pixels, expected)


@pytest.mark.parametrize("cuda", [False, pytest.param(True, marks=pytest.mark.skipif(
    os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1", reason="requires CUDA"))])
def test_the_edge_trim_pulls_in_what_reaches_past_the_masks(app, make_window, tmp_path, dialogs, cuda):
    window, _ = make_window(EdgeRenderer())
    if cuda:
        from gs_object_extraction.renderer import GsplatRenderer
        window.renderer_factory = GsplatRenderer
    original = window.scene.copy()
    extract_object(app, window)
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)
    assert window.stages["off"][0] == pytest.approx(.2)

    shown = window.object_scene  # the preview holds what the export will write
    renderer = window.viewport.renderer
    if cuda:
        buffers = [getattr(renderer, name).data_ptr()
                   for name in ("means", "scales", "rotations", "opacities", "sh")]
    np.testing.assert_allclose(shown.scales[0], original.scales[0] * TRIM)
    np.testing.assert_allclose(shown.scales[1], original.scales[1])
    assert export_object(app, window, tmp_path / "trimmed.ply")
    exported = load_ply(tmp_path / "trimmed.ply")
    np.testing.assert_allclose(exported.scales[0], original.scales[0] * TRIM, rtol=3e-7)
    np.testing.assert_allclose(exported.scales[1], original.scales[1], rtol=3e-7)
    np.testing.assert_array_equal(window.scene.scales, original.scales)  # the scene itself is untouched

    for factor in (.5, .85, .7, 1.):
        window.trim_box.setValue(factor)
        app.processEvents()
        expected = original.scales[CLEANED].copy()
        expected[0] *= factor
        assert window.viewport.renderer is renderer and window.object_scene is shown
        np.testing.assert_allclose(shown.scales, expected)
        np.testing.assert_array_equal(window.scene.scales, original.scales)
        assert export_object(app, window, tmp_path / "adjusted.ply")
        np.testing.assert_allclose(load_ply(tmp_path / "adjusted.ply").scales, expected, rtol=3e-7)
        if cuda:
            assert buffers == [getattr(renderer, name).data_ptr()
                               for name in ("means", "scales", "rotations", "opacities", "sh")]
            fresh = GsplatRenderer(window.trimmed_object())
            window.viewport.render_now()
            camera = window.viewport.camera
            np.testing.assert_array_equal(window.viewport.pixels,
                fresh.render_image(camera, background=window.viewport.background))
            for actual, reference in zip(renderer.depth_image(camera), fresh.depth_image(camera)):
                np.testing.assert_allclose(actual, reference)
    assert export_object(app, window, tmp_path / "whole.ply")
    np.testing.assert_allclose(load_ply(tmp_path / "whole.ply").scales, original.scales[CLEANED], rtol=3e-7)
    assert not dialogs


@pytest.mark.parametrize("cuda", [False, pytest.param(True, marks=pytest.mark.skipif(
    os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1", reason="requires CUDA"))])
def test_preview_controls_and_export_preserve_world_coordinates(app, make_window, tmp_path, cuda):
    window, _ = make_window(EdgeRenderer())
    # An off-origin object with distinct axes and arbitrary rotations exposes
    # recentering, axis swaps and quaternion-order mistakes that spheres hide.
    window.scene.means += [12., -7., 3.]
    window.scene.scales *= [1., 2., 3.]
    rotations = np.random.default_rng(53).normal(size=(len(window.scene), 4))
    window.scene.quaternions = rotations / np.linalg.norm(rotations, axis=1)[:, None]
    original = window.scene.copy()
    if cuda:
        from gs_object_extraction.renderer import GsplatRenderer
        window.renderer_factory = GsplatRenderer
    extract_object(app, window)
    selected = window.stages["cleaned"]
    renderer = window.viewport.renderer
    source_covariances = original.covariances[selected]

    for index in range(window.up_box.count()):
        window.up_box.setCurrentIndex(index)
        window.reset_view()
        window.viewport.orbit.rotate(17, -9)
        window.viewport.orbit.pan(4, 2, window.viewport.height())
        camera_before = window.viewport.orbit.camera(64, 48).world_to_camera.copy()
        factor = (.5, .85, 1.)[index % 3]
        window.trim_box.setValue(factor)
        window.background_box.setCurrentIndex(index % 2)
        assert window.viewport.renderer is renderer
        np.testing.assert_array_equal(window.viewport.orbit.camera(64, 48).world_to_camera, camera_before)
        shown = window.object_scene
        for name in ("means", "quaternions", "sh", "opacities", "ids"):
            np.testing.assert_array_equal(getattr(shown, name), getattr(original, name)[selected])
        factors = np.array([factor, 1.])  # only the first Gaussian needs edge trim
        np.testing.assert_allclose(shown.covariances, source_covariances * factors[:, None, None] ** 2)
        if cuda:
            np.testing.assert_array_equal(renderer.means.cpu().numpy(), original.means[selected].astype(np.float32))
            np.testing.assert_array_equal(renderer.rotations.cpu().numpy(), original.quaternions[selected].astype(np.float32))
            np.testing.assert_array_equal(renderer.scales.cpu().numpy(), shown.scales.astype(np.float32))

        path = tmp_path / f"axis_{index}.ply"
        assert export_object(app, window, path)
        exported = load_ply(path)
        np.testing.assert_array_equal(exported.ids, original.ids[selected])
        # PLY stores coordinates as float32; no coordinate transform is allowed.
        np.testing.assert_array_equal(exported.means, original.means[selected].astype(np.float32))
        np.testing.assert_allclose(exported.quaternions, original.quaternions[selected], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(exported.covariances, shown.covariances, rtol=2e-6, atol=1e-8)
        for name in ("means", "scales", "quaternions", "sh", "opacities"):
            np.testing.assert_array_equal(getattr(window.scene, name), getattr(original, name))

    window.preview_box.setCurrentText("Scene")
    window.reset_view()
    window.preview_box.setCurrentText("Object only")
    np.testing.assert_array_equal(window.object_scene.means, original.means[selected])
    np.testing.assert_array_equal(window.object_scene.quaternions, original.quaternions[selected])


def test_trim_changed_in_scene_mode_is_used_for_export_and_next_preview(app, make_window, tmp_path):
    window, renderer = make_window(EdgeRenderer())
    extract_object(app, window)
    window.preview_box.setCurrentText("Scene")
    window.trim_box.setValue(.5)
    assert window.viewport.renderer is renderer and window.object_scene is None
    assert export_object(app, window, tmp_path / "object.ply")
    exported = load_ply(tmp_path / "object.ply")
    window.preview_box.setCurrentText("Object only")
    np.testing.assert_allclose(window.object_scene.scales, exported.scales, rtol=3e-7)
    np.testing.assert_allclose(exported.scales[0], window.scene.scales[0] * .5, rtol=3e-7)


def test_trim_upload_failure_returns_to_scene(app, make_window, monkeypatch):
    window, scene_renderer = make_window(EdgeRenderer())
    extract_object(app, window)

    def fail(scales):
        raise RuntimeError("GPU upload failed")

    monkeypatch.setattr(window.viewport.renderer, "update_scales", fail)
    window.trim_box.setValue(.5)
    assert window.preview_box.currentText() == "Scene"
    assert window.viewport.renderer is scene_renderer and window.object_scene is None
    assert "GPU upload failed" in window.statusBar().currentMessage()


@pytest.mark.parametrize("change", ["add", "remove", "open"])
def test_confirmed_view_or_scene_changes_invalidate_results(app, make_window, monkeypatch, tmp_path, change):
    window, _ = make_window()
    extract_object(app, window)
    if change == "add":
        window.preview_box.setCurrentText("Scene")
        window.viewport._timer.stop()
        window.viewport.camera, window.viewport.mask = CAMERA, MASK.copy()
        window.viewport.points, window.viewport.labels = [(12, 4)], [1]
        window.viewport.score = .9
        window.add_view()
        assert len(window.views) == 2
    elif change == "remove":
        window.remove_view()
        assert not window.views
    else:
        replacement = tmp_path / "replacement.ply"
        save_ply(make_scene(), replacement)
        monkeypatch.setattr("gs_object_extraction.renderer.GsplatRenderer", lambda scene: FakeRenderer())
        assert window.open_ply(replacement)
        wait_until(app, lambda: window.load_job is None)
        assert window.source_path == replacement.resolve() and not window.views
    assert window.stages is None and window.viewport.active is None
    assert window.preview_box.currentText() == "Scene"
    assert not window.export_action.isEnabled() and not window.preview_box.isEnabled()
    assert not window.export_ply(tmp_path / "stale.ply")
    assert not (tmp_path / "stale.ply").exists()


def test_extracting_locks_the_inputs_but_leaves_the_view_to_look_at(app, make_window, tmp_path, dialogs):
    """A job owns the renderer per call, not for its whole run, so the view stays live while it works."""
    window, renderer = make_window(FakeRenderer(blocked=True))
    window.start_extraction()
    wait_until(app, renderer.started.is_set)
    job = window.extraction_job
    frame_count = len(renderer.image_calls)
    original_eye = window.viewport.orbit.eye.copy()
    assert window.viewport.suspended and window.viewport.isEnabled()  # suspended marking, not a dead widget
    assert not window.open_action.isEnabled() and not window.extract_action.isEnabled()
    assert not window.scene_box.isEnabled() and not window.view_list.isEnabled()
    assert not window.add_view_action.isEnabled() and not window.select_action.isEnabled()
    window.start_extraction()
    assert window.extraction_job is job
    assert not window.open_ply(tmp_path / "unreadable.ply")
    window.remove_view()
    window.add_view()
    window.reset_view()
    window.set_up("+Z")
    window.set_selecting(True)
    window.viewport.add_point(10, 10, 1)
    assert len(window.views) == 1 and not window.viewport.points  # marking is what the job rules out
    np.testing.assert_array_equal(window.viewport.orbit.eye, original_eye)
    # The lift the job is in holds the renderer, so the frame is skipped instead of queueing behind it.
    window.viewport.render_now()
    assert len(renderer.image_calls) == frame_count and window.viewport.pick(0, 0) is None
    assert not renderer.depth_calls
    # A drag still turns the camera; the frame it asks for waits for the renderer rather than the job.
    drag(window.viewport, (60, 40), (90, 40), Qt.LeftButton, Qt.AltModifier)  # Alt + left orbits
    assert not np.array_equal(window.viewport.orbit.eye, original_eye)
    assert len(renderer.image_calls) == frame_count
    window.cancel_extraction()
    assert not window.cancel_button.isEnabled()
    renderer.release.set()
    wait_until(app, lambda: window.extraction_job is None)
    assert window.stages is None and not window.viewport.suspended and window.viewport.isEnabled()
    assert window.open_action.isEnabled() and window.extract_action.isEnabled()
    assert not window.progress.isVisible() and not window.cancel_button.isVisible()
    wait_until(app, lambda: len(renderer.image_calls) > frame_count)  # the dragged view arrives
    assert len(renderer.lift_threads) == 1 and not dialogs


def test_worker_error_restores_controls_and_reports_reason(app, make_window, dialogs):
    window, _ = make_window(FakeRenderer(fail=True))
    extract_object(app, window)
    assert window.stages is None and not window.viewport.suspended
    assert window.open_action.isEnabled() and window.extract_action.isEnabled()
    assert not window.export_action.isEnabled() and not window.progress.isVisible()
    assert dialogs == [("Cannot extract object", "test renderer failed during lifting")]


def test_close_waits_for_worker_cancellation_before_destroying_window(app, make_window, dialogs):
    window, renderer = make_window(FakeRenderer(blocked=True))
    window.start_extraction()
    wait_until(app, renderer.started.is_set)
    assert not window.close()
    assert window.isVisible() and window.extraction_job is not None
    assert window.extraction_job.isInterruptionRequested()
    renderer.release.set()
    wait_until(app, lambda: window.extraction_job is None and not window.isVisible())
    assert not dialogs


def test_close_lets_a_running_export_finish_writing_the_file(app, make_window, tmp_path, monkeypatch, dialogs):
    window, _ = make_window()
    extract_object(app, window)
    started, release = threading.Event(), threading.Event()

    def slow(scene, path):
        started.set()
        assert release.wait(5), "test did not release the export"
        save_ply(scene, path)

    monkeypatch.setattr("gs_object_extraction.app.export.save_ply", slow)
    target = tmp_path / "object.ply"
    try:
        assert window.export_ply(target)
        wait_until(app, started.is_set)
        assert not window.close()  # an interrupted write would leave nothing to keep
        assert window.isVisible() and window.export_job is not None
        release.set()
        wait_until(app, lambda: window.export_job is None and not window.isVisible())
    finally:
        release.set()
    assert len(load_ply(target)) == int(CLEANED.sum())
    assert not dialogs


def test_empty_extraction_has_no_preview_or_export(app, make_window, tmp_path, dialogs):
    window, _ = make_window(FakeRenderer(empty=True))
    extract_object(app, window)
    assert not window.stages["cleaned"].any()
    assert "No object Gaussians remain" in window.result_label.text()
    assert window.viewport.active is None and window.preview_box.currentText() == "Scene"
    assert not window.preview_box.isEnabled() and not window.export_action.isEnabled()
    assert not window.export_ply(tmp_path / "empty.ply") and not dialogs


def test_a_failed_export_is_reported_and_leaves_the_window_usable(app, make_window, tmp_path, monkeypatch, dialogs):
    """``save_ply`` keeps the target intact itself (see test_ply); the window has to say so."""
    window, _ = make_window()
    extract_object(app, window)
    target = tmp_path / "object.ply"
    target.write_bytes(b"previous export")
    before = set(tmp_path.iterdir())

    def fail(scene, path):
        raise OSError("test disk write failure")

    monkeypatch.setattr("gs_object_extraction.app.export.save_ply", fail)
    assert export_object(app, window, target)  # started, then failed on the worker
    assert target.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == before
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)
    assert window.export_action.isEnabled() and not window.progress.isVisible()
    assert dialogs == [("Cannot export object", "test disk write failure")]


def test_an_export_runs_off_the_gui_thread_and_locks_the_controls_while_it_does(
        app, make_window, tmp_path, monkeypatch, dialogs):
    window, _ = make_window()
    extract_object(app, window)
    threads, release = [], threading.Event()

    def slow(scene, path):
        threads.append(threading.get_ident())
        assert release.wait(5), "test did not release the export"
        save_ply(scene, path)

    monkeypatch.setattr("gs_object_extraction.app.export.save_ply", slow)
    try:
        assert window.export_ply(tmp_path / "object.ply")
        wait_until(app, lambda: bool(threads))
        assert threads[0] != threading.get_ident()  # the GUI thread is free while it writes
        assert not window.export_action.isEnabled() and not window.open_action.isEnabled()
        assert not window.preview_box.isEnabled() and window.progress.isVisible()
        release.set()
        wait_until(app, lambda: window.export_job is None)
    finally:
        release.set()
    assert window.export_action.isEnabled() and not window.progress.isVisible()
    assert len(load_ply(tmp_path / "object.ply")) == int(CLEANED.sum())
    assert not dialogs


def test_export_rejects_other_extension_without_overwriting_a_different_filename(app, make_window, tmp_path, dialogs):
    window, _ = make_window()
    extract_object(app, window)
    selected = tmp_path / "object.other"
    suffixed = tmp_path / "object.other.ply"
    selected.write_bytes(b"selected file")
    suffixed.write_bytes(b"previous PLY export")

    assert not window.export_ply(selected)
    assert selected.read_bytes() == b"selected file"
    assert suffixed.read_bytes() == b"previous PLY export"
    assert dialogs == [("Cannot export object", "Use the .ply file extension for the object export.")]
    assert window.export_action.isEnabled()


@pytest.mark.parametrize("alias", [False, True])
def test_export_refuses_source_file_and_hard_link_alias(app, make_window, tmp_path, dialogs, alias):
    window, _ = make_window()
    extract_object(app, window)
    original = window.source_path.read_bytes()
    target = window.source_path
    if alias:
        target = tmp_path / "source_alias.ply"
        os.link(window.source_path, target)
    assert not window.export_ply(target)
    assert target.read_bytes() == original and window.source_path.read_bytes() == original
    assert dialogs == [("Cannot export object", "Choose a different file from the source scene.")]


def test_export_uses_the_default_file_mode_and_keeps_an_existing_targets_mode(app, make_window, tmp_path, dialogs):
    import stat
    window, _ = make_window()
    extract_object(app, window)
    reference = tmp_path / "reference.txt"
    reference.write_text("x")  # an ordinary new file gets the umask default
    fresh = tmp_path / "fresh.ply"
    assert export_object(app, window, fresh)
    assert stat.S_IMODE(fresh.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
    shared = tmp_path / "shared.ply"
    shared.write_text("previous export")
    shared.chmod(0o664)
    assert export_object(app, window, shared)
    assert stat.S_IMODE(shared.stat().st_mode) == 0o664 and not dialogs


def test_a_second_extraction_in_the_same_window_can_be_cancelled_too(app, make_window, dialogs):
    window, renderer = make_window(FakeRenderer(blocked=True))
    for attempt in (1, 2):
        if attempt == 2:
            renderer.started.clear()
            renderer.release.clear()
            renderer.block_at = {len(renderer.lift_threads) + 1}
        lifts = len(renderer.lift_threads)
        window.start_extraction()
        wait_until(app, renderer.started.is_set)
        assert window.cancel_button.isVisible() and window.cancel_button.isEnabled(), f"attempt {attempt}"
        window.cancel_button.click()
        renderer.release.set()
        wait_until(app, lambda: window.extraction_job is None)
        assert window.stages is None and len(renderer.lift_threads) == lifts + 1
    assert not dialogs


def test_reset_view_frames_the_object_in_preview_and_the_scene_otherwise(app, make_window, dialogs):
    window, _ = make_window()
    extract_object(app, window)
    assert window.preview_box.currentText() == "Object only"
    window.reset_view()
    orbit, scene = window.viewport.orbit, window.scene
    np.testing.assert_allclose(orbit.target, np.median(scene.means[CLEANED], axis=0))
    camera = orbit.camera(window.viewport.width(), window.viewport.height())
    tan_x, tan_y = camera.width / (2 * camera.fx), camera.height / (2 * camera.fy)
    for mean, scale in zip(scene.means[CLEANED], scene.scales[CLEANED]):
        x, y, z = camera.world_to_camera[:3, :3] @ mean + camera.world_to_camera[:3, 3]
        radius = 3 * scale.max()  # every splat, 3 sigma wide, lies inside the view
        assert abs(x) + radius * np.hypot(1, tan_x) <= z * tan_x
        assert abs(y) + radius * np.hypot(1, tan_y) <= z * tan_y
    window.preview_box.setCurrentText("Scene")
    window.reset_view()
    np.testing.assert_allclose(window.viewport.orbit.target, Orbit.framing(scene.means, up=window.up_vector()).target)


def test_off_mask_limit_from_the_panel_reaches_the_extraction(app, make_window, dialogs):
    window, _ = make_window()
    assert window.off_threshold_box.value() == pytest.approx(0.35)
    window.off_threshold_box.setValue(.9)  # tolerate the junk Gaussian's two thirds of spill
    extract_object(app, window)
    np.testing.assert_array_equal(window.stages["cleaned"], SELECTED)
    window.off_threshold_box.setValue(.35)
    extract_object(app, window)
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)


def test_showing_a_second_object_frees_the_first_objects_renderer_first(app, make_window):
    window, scene_renderer = make_window()
    extract_object(app, window)
    window.preview_box.setCurrentText("Object only")
    first = window.viewport.renderer
    assert first is not scene_renderer
    seen = []

    def factory(scene):
        seen.append(window.viewport.renderer)  # what still holds GPU memory while the next copy is built
        return FakeRenderer(n=len(scene))

    window.renderer_factory = factory
    window.update_preview()
    assert seen == [scene_renderer]


def test_a_fresh_extraction_is_framed_once_and_switching_the_preview_keeps_the_camera(app, make_window):
    window, _ = make_window()
    window.viewport.orbit = Orbit(np.full(3, 5.), 40.)  # nowhere near the object
    extract_object(app, window)
    np.testing.assert_allclose(window.viewport.orbit.target, np.median(window.scene.means[CLEANED], axis=0))
    window.viewport.orbit.target = np.full(3, 7.)
    window.preview_box.setCurrentText("Scene")
    window.preview_box.setCurrentText("Object only")
    np.testing.assert_allclose(window.viewport.orbit.target, np.full(3, 7.))  # the user's camera is left alone


def test_choosing_a_view_stands_where_it_was_marked_and_dragging_leaves_it(app, make_window):
    window, _ = make_window()
    view = window.viewport
    view.orbit = Orbit(np.full(3, 4.), 9., yaw=1.)
    window.review_view(0)
    forward = window.views[0].camera.world_to_camera[2, :3]
    np.testing.assert_allclose(view.orbit.eye, CAMERA.eye, atol=1e-9)
    np.testing.assert_allclose((view.orbit.target - view.orbit.eye) / view.orbit.distance, forward, atol=1e-9)
    assert view.reviewing is not None and view.reviewing[0] is window.views[0]
    drag(view, (10, 10), (40, 10), Qt.LeftButton, Qt.AltModifier)
    assert view.reviewing is None


def test_removing_a_view_does_not_send_the_camera_to_the_next_one(app, make_window):
    window, _ = make_window()
    window.views.append(MaskedView(CAMERA, MASK.copy(), ((12, 4),), (1,)))
    window.view_list.addItem("View 2")
    window.view_list.setCurrentRow(1)  # the last row: taking it out moves the selection up to row 0
    window.viewport.orbit.target = np.full(3, 3.)
    window.viewport.view_changed()  # leaves the view the row just showed
    window.remove_view()
    assert len(window.views) == 1 and window.viewport.reviewing is None
    np.testing.assert_allclose(window.viewport.orbit.target, np.full(3, 3.))


def test_the_mask_overlay_has_a_solid_outline_over_a_translucent_fill(app):
    from gs_object_extraction.app.viewport import MASK_RGBA, OUTLINE_RGBA, _mask_image
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    image = _mask_image(mask)
    assert image.pixelColor(5, 10).alpha() == OUTLINE_RGBA[3] and image.pixelColor(6, 10).alpha() == OUTLINE_RGBA[3]
    assert image.pixelColor(10, 10).alpha() == MASK_RGBA[3]
    assert image.pixelColor(2, 2).alpha() == 0


def box_around(window, index):
    """A box on the frame on screen that holds exactly the object's ``index``-th Gaussian."""
    from gs_object_extraction.app.pick import project
    x, y, _ = project(window.viewport.camera, window.object_scene.means[[index]])
    return x[0] - 1e-3, y[0] - 1e-3, x[0] + 1e-3, y[0] + 1e-3


def show_object(app, window):
    extract_object(app, window)
    window.viewport.render_now()
    window.set_selecting(True)


def test_selecting_marks_gaussians_and_only_deleting_removes_them(app, make_window, tmp_path):
    window, _ = make_window()
    show_object(app, window)
    assert window.stages["cleaned"].sum() == 2 and not window.delete_action.isEnabled()
    assert sorted(k.toString() for k in window.delete_action.shortcuts()) == ["Del", "X"]
    window.pick_region(*box_around(window, 0), 0)
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)  # selected, not gone
    assert window.delete_action.isEnabled() and window.viewport.highlight.shape == (1, 3)
    np.testing.assert_allclose(window.viewport.highlight[0], window.object_scene.means[0])
    assert "1 selected" in window.pick_label.text()
    window.delete_selection()
    np.testing.assert_array_equal(window.stages["cleaned"], [False, True, False, False, False])
    assert len(window.object_scene) == 1 and window.viewport.renderer.n == 1
    assert window.viewport.selecting  # selecting goes on until the user leaves it
    assert window.viewport.highlight is None and not window.delete_action.isEnabled()
    assert window.undo_delete_action.isEnabled() and "1 deleted by hand" in window.result_label.text()
    window.viewport.render_now()
    assert export_object(app, window, tmp_path / "deleted.ply")
    assert len(load_ply(tmp_path / "deleted.ply")) == 1
    window.undo_delete()
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)
    assert len(window.object_scene) == 2 and "deleted by hand" not in window.result_label.text()


def test_shift_adds_ctrl_takes_away_and_a_plain_drag_replaces(app, make_window):
    window, _ = make_window()
    show_object(app, window)
    window.pick_region(*box_around(window, 0), 0)
    window.pick_region(*box_around(window, 1), 1)
    np.testing.assert_array_equal(window._picked, [True, True])
    window.pick_region(*box_around(window, 0), 2)
    np.testing.assert_array_equal(window._picked, [False, True])
    window.pick_region(*box_around(window, 0), 0)
    np.testing.assert_array_equal(window._picked, [True, False])
    window.clear_points()  # Esc
    assert window._picked is None and window.viewport.highlight is None


def test_the_selection_survives_moving_the_camera_and_deleting_all_of_it_is_refused(app, make_window):
    window, _ = make_window()
    show_object(app, window)
    window.pick_region(-1e6, -1e6, 1e6, 1e6, 0)
    window.viewport.orbit.rotate(30, 0)
    window.viewport.view_changed()
    window.viewport.render_now()
    assert window._picked.all() and window.viewport.highlight is not None
    window.delete_selection()
    assert window.stages["cleaned"].sum() == 2 and "whole object" in window.statusBar().currentMessage()
    assert not window.undo_delete_action.isEnabled()


def test_a_box_that_holds_nothing_selects_nothing(app, make_window):
    window, _ = make_window()
    show_object(app, window)
    window.pick_region(-9e6, -9e6, -8e6, -8e6, 0)
    assert window._picked is None and not window.delete_action.isEnabled()
    assert "Nothing selected" in window.statusBar().currentMessage()


def test_a_drag_in_the_object_preview_draws_a_box_and_a_click_selects_around_the_point(app, make_window):
    window, _ = make_window()
    show_object(app, window)
    view = window.viewport
    seen = []
    view.select_requested.connect(lambda *request: seen.append(request))
    eye = view.orbit.eye.copy()
    scale = view.camera.width / view.width()
    drag(view, (10, 10), (110, 90), Qt.LeftButton)
    np.testing.assert_allclose(seen[-1][:4], (10 * scale, 10 * scale, 110 * scale, 90 * scale))
    assert seen[-1][4] == 0
    np.testing.assert_allclose(view.orbit.eye, eye)  # the drag did not orbit
    for modifier, mode in ((Qt.NoModifier, 0), (Qt.ShiftModifier, 1), (Qt.ControlModifier, 2)):
        view.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(50, 60), QPointF(50, 60), Qt.LeftButton, Qt.LeftButton, modifier))
        view.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(50, 60), QPointF(50, 60), Qt.LeftButton, Qt.NoButton, modifier))
        x0, y0, x1, y1, seen_mode = seen[-1]
        assert (x1 - x0) == pytest.approx(24 * scale) and (x0 + x1) / 2 == pytest.approx(50 * scale) and seen_mode == mode


def test_select_mode_ends_with_the_object_preview_and_the_selection_with_the_object(app, make_window):
    window, _ = make_window()
    show_object(app, window)
    window.pick_region(*box_around(window, 0), 0)
    window.preview_box.setCurrentText("Scene")
    assert not window.viewport.selecting and not window.select_action.isChecked()
    assert window._picked is None and window.viewport.highlight is None and not window.delete_action.isEnabled()


def test_a_new_extraction_forgets_the_delete_history(app, make_window):
    window, _ = make_window()
    show_object(app, window)
    window.pick_region(*box_around(window, 0), 0)
    window.delete_selection()
    assert window.undo_delete_action.isEnabled()
    extract_object(app, window)
    assert not window.undo_delete_action.isEnabled()
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)


def test_the_x_key_deletes_the_selection_and_does_nothing_without_one(app, make_window):
    from PySide6.QtTest import QTest
    window, _ = make_window()
    show_object(app, window)
    window.activateWindow()
    QTest.keyClick(window, Qt.Key_X)
    assert window.stages["cleaned"].sum() == 2  # nothing selected, nothing to delete
    window.pick_region(*box_around(window, 0), 0)
    QTest.keyClick(window, Qt.Key_X)
    np.testing.assert_array_equal(window.stages["cleaned"], [False, True, False, False, False])


def test_the_box_round_the_object_follows_what_is_left_and_the_cube_over_the_view_hides_it(app, make_window):
    window, _ = make_window()
    view = window.viewport
    assert view.box_button.isVisible() and not view.box_button.isEnabled() and view.box is None  # greyed until there is a box
    show_object(app, window)
    assert view.box.shape == (8, 3) and view.box_button.isEnabled() and view.box_button.isChecked()
    both = view.box.copy()
    means = window.object_scene.means
    assert "×" in view.box_label.text() and view.box_label.isVisible()
    assert (both.min(axis=0) <= means.min(axis=0) + 1e-9).all() and (both.max(axis=0) >= means.max(axis=0) - 1e-9).all()
    assert view.box_button.x() + view.box_button.width() <= view.width()  # the cube sits inside the view, in its corner
    assert view.box_label.x() + view.box_label.width() < view.box_button.x()
    window.pick_region(*box_around(window, 0), 0)
    window.delete_selection()
    assert not np.allclose(view.box, both)  # one Gaussian is left, the box shrinks to it
    view.box_button.click()
    assert not view.show_box and not view.box_label.isVisible() and view.box_button.isVisible()
    view.box_button.click()  # on again
    window.preview_box.setCurrentText("Scene")
    assert view.box is not None and view.box_button.isEnabled() and view.box_label.isVisible()  # the scene gets the box too
    window.invalidate_result()
    assert view.box is None and view.box_button.isVisible() and not view.box_button.isEnabled() and not view.box_label.isVisible()


def fake_refine(renderer, targets, *, steps=None, on_step=None):
    on_step(1, 2)
    on_step(2, 2)
    n = renderer.n
    return {"opacities": np.full(n, .99), "sh": np.ones((n, 16, 3)), "scale_factors": np.full((n, 3), 1.5),
            "before": .5, "after": .9}


@pytest.fixture
def refining(monkeypatch):
    monkeypatch.setattr("gs_object_extraction.app.refining.refine", fake_refine)


def refine_object(app, window):
    window.start_refine()
    assert window.refine_job is not None
    wait_until(app, lambda: window.refine_job is None)


def test_refining_refits_the_object_and_the_export_carries_it(app, make_window, refining, tmp_path):
    window, _ = make_window()
    window.viewport.render_now()
    assert not window.refine_button.isEnabled()  # nothing extracted yet
    extract_object(app, window)
    assert window.refine_button.isEnabled() and not window.revert_button.isEnabled()
    original = window.trimmed_object(refined=False)
    refine_object(app, window)
    assert window.revert_button.isEnabled() and "masks filled 50.0% → 90.0%" in window.result_label.text()
    assert window.preview_box.currentText() == "Object only" and window.viewport.renderer.n == 2
    np.testing.assert_allclose(window.object_scene.opacities, .99)
    np.testing.assert_allclose(window.object_scene.sh, 1.)
    np.testing.assert_allclose(window.object_scene.scales, original.scales * 1.5)  # sizes come from the refit, on top of the trim
    np.testing.assert_array_equal(window.object_scene.means, original.means)  # positions stay
    np.testing.assert_array_equal(window.scene.opacities, make_scene().opacities)  # the scene is not touched
    window.viewport.render_now()
    assert export_object(app, window, tmp_path / "refined.ply")
    np.testing.assert_allclose(load_ply(tmp_path / "refined.ply").opacities, .99, atol=1e-3)


def test_what_is_deleted_after_a_refit_leaves_the_rest_refit(app, make_window, refining):
    window, _ = make_window()
    show_object(app, window)
    refine_object(app, window)
    window.viewport.render_now()
    window.pick_region(*box_around(window, 0), 0)
    window.delete_selection()
    assert len(window.object_scene) == 1
    np.testing.assert_allclose(window.object_scene.opacities, .99)
    np.testing.assert_allclose(window.object_scene.scales, window.trimmed_object(refined=False).scales * 1.5)
    window.undo_delete()
    assert len(window.object_scene) == 2
    np.testing.assert_allclose(window.object_scene.opacities, .99)


def test_reverting_puts_back_what_extraction_left_and_a_new_extraction_drops_the_refit(app, make_window, refining):
    window, _ = make_window()
    extract_object(app, window)
    left = window.trimmed_object(refined=False).opacities.copy()
    refine_object(app, window)
    window.revert_refine()
    np.testing.assert_allclose(window.object_scene.opacities, left)
    assert "Refined" not in window.result_label.text() and not window.revert_button.isEnabled()
    refine_object(app, window)
    extract_object(app, window)
    assert window._refined is None and not window.revert_button.isEnabled()


def test_a_failed_refit_is_reported_and_changes_nothing(app, make_window, dialogs, monkeypatch):
    def fail(renderer, targets, *, steps=None, on_step=None):
        raise RuntimeError("out of memory")
    monkeypatch.setattr("gs_object_extraction.app.refining.refine", fail)
    window, _ = make_window()
    extract_object(app, window)
    refine_object(app, window)
    assert dialogs == [("Cannot refine object", "out of memory")] and window._refined is None
    dialogs.clear()


def test_a_refit_can_be_cancelled_and_blocks_editing_meanwhile(app, make_window, monkeypatch):
    import time
    started = []

    def slow(renderer, targets, *, steps=None, on_step=None):
        started.append(True)
        for step in range(2000):
            time.sleep(.005)
            on_step(step, 2000)  # raises once cancelled
        raise AssertionError("was never cancelled")
    monkeypatch.setattr("gs_object_extraction.app.refining.refine", slow)
    window, _ = make_window()
    extract_object(app, window)
    window.start_refine()
    wait_until(app, lambda: bool(started))
    assert not window.extract_button.isEnabled() and not window.export_action.isEnabled() and not window.refine_button.isEnabled()
    window.cancel_job()
    wait_until(app, lambda: window.refine_job is None)
    assert window._refined is None and not window.revert_button.isEnabled()
    assert window.refine_button.isEnabled()


def test_the_object_and_background_toggles_sit_over_the_view_and_drive_the_preview(app, make_window):
    window, _ = make_window()
    view = window.viewport
    assert view.object_button.isVisible() and not view.object_button.isEnabled()  # a scene is open, nothing extracted yet
    assert view.background_button.isVisible() and view.box_button.isVisible()  # all three are always in place
    assert not view.background_button.isEnabled() and not view.box_button.isEnabled()
    extract_object(app, window)
    assert view.object_button.isEnabled() and view.object_button.isChecked()  # the object on its own is showing
    assert window.preview_box.currentText() == "Object only" and view.background_button.isEnabled() and view.box_button.isEnabled()
    view.object_button.click()  # back to the scene
    assert window.preview_box.currentText() == "Scene" and view.active is None
    assert view.background_button.isVisible() and view.box_button.isVisible() and view.object_button.isVisible()
    assert not view.background_button.isEnabled() and view.box_button.isEnabled()  # the scene has no background to set, but the box shows
    positions = [b.pos() for b in (view.object_button, view.background_button, view.box_button)]
    view.object_button.click()
    assert view.active is not None and view.background_button.isEnabled() and view.box_button.isEnabled()
    assert positions == [b.pos() for b in (view.object_button, view.background_button, view.box_button)]  # they did not move
    assert window.background_box.currentText() == "White"
    view.background_button.click()
    assert window.background_box.currentText() == "Black" and view.background == (0., 0., 0.)
    assert "black" in view.background_button.toolTip().lower().split("(")[0]
    xs = [b.x() for b in (view.object_button, view.background_button, view.box_button)]
    assert xs == sorted(xs) and view.box_button.x() + view.box_button.width() <= view.width()  # in a row, inside the view
    assert view.box_label.x() + view.box_label.width() < view.object_button.x()
    assert not any(hasattr(window, name) for name in ("box_check",))


def test_the_kept_gaussians_show_as_blue_points_once_the_icon_is_turned_on(app, make_window):
    from gs_object_extraction.app.pick import project
    from gs_object_extraction.app.viewport import CLOUD_RGBA
    window, _ = make_window()
    view = window.viewport
    view.set_map_cloud(window.scene.means)  # what opening a scene does
    assert view.cloud_button.isVisible() and view.cloud is None  # nothing kept yet
    show_object(app, window)
    assert view.cloud_button.isEnabled() and not view.cloud_button.isChecked() and not view.show_cloud  # off until asked for
    view.cloud_button.click()
    assert view.show_cloud
    np.testing.assert_allclose(view.cloud, window.scene.means[window.stages["cleaned"]])
    view.render_now()
    image = view._cloud_dots()
    x, y, _ = project(view.camera, view.cloud)
    for px, py in zip(np.round(x).astype(int), np.round(y).astype(int)):
        assert image.pixelColor(px, py).getRgb() == CLOUD_RGBA  # a blue dot where each kept Gaussian lands
    assert image.pixelColor(0, 0).alpha() == 0
    window.preview_box.setCurrentText("Scene")
    assert view.cloud is not None  # the points show over the scene as well
    view.cloud_button.click()
    assert not view.show_cloud
    window.invalidate_result()
    assert view.cloud is None and view.cloud_button.isEnabled()  # the scene's own points are still there


def test_a_huge_object_is_thinned_before_it_is_drawn_as_points(app, make_window):
    from gs_object_extraction.app import viewport as viewport_module
    window, _ = make_window()
    points = np.random.default_rng(6).normal(size=(viewport_module.CLOUD_LIMIT * 2 + 7, 3))
    window.viewport.set_cloud(points)
    assert len(window.viewport.cloud) <= viewport_module.CLOUD_LIMIT


def test_the_scene_shows_its_own_points_faintly_and_the_kept_ones_darker_over_them(app, make_window):
    from gs_object_extraction.app import viewport as viewport_module
    from gs_object_extraction.app.pick import project
    window, _ = make_window()
    view = window.viewport
    view.set_map_cloud(window.scene.means)  # what opening a scene does
    assert view.map_cloud is not None and view.cloud is None and view.cloud_button.isEnabled()
    view.orbit = Orbit(np.median(window.scene.means, axis=0), 6.)  # look at the points
    view.view_changed()
    view.render_now()
    image = view._map_dots()
    x, y, depth = project(view.camera, view.map_cloud)
    on_screen = (depth > view.camera.near) & (x >= 0) & (x < view.camera.width) & (y >= 0) & (y < view.camera.height)
    assert on_screen.any()
    for px, py in zip(np.round(x[on_screen]).astype(int), np.round(y[on_screen]).astype(int)):
        if 0 <= px < view.camera.width and 0 <= py < view.camera.height:
            assert image.pixelColor(px, py).getRgb() == viewport_module.MAP_RGBA  # a fainter dot for the scene's Gaussians
    assert viewport_module.MAP_RGBA[3] < viewport_module.CLOUD_RGBA[3] and viewport_module.MAP_LIMIT < viewport_module.CLOUD_LIMIT
    show_object(app, window)  # the object on its own: the scene's points are not drawn behind it
    assert view.active is not None and view.cloud is not None
    view.set_map_cloud(np.random.default_rng(2).normal(size=(viewport_module.MAP_LIMIT * 3 + 5, 3)))
    assert len(view.map_cloud) <= viewport_module.MAP_LIMIT
    view.clear()
    assert view.map_cloud is None and view.cloud is None
