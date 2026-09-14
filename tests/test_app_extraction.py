"""Exercise the extraction workflow through the GUI with real worker threads."""

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from gs_object_extraction.app.orbit import Orbit
from gs_object_extraction.app.views import MaskedView
from gs_object_extraction.app.viewport import BACKGROUND
from gs_object_extraction.app.window import MainWindow
from gs_object_extraction.camera import Camera
from gs_object_extraction.ply import load_ply, save_ply
from gs_object_extraction.renderer import Lifted
from gs_object_extraction.scene import GaussianScene


# The fourth Gaussian has an off-mask piece hidden by the fifth Gaussian until
# the scene is removed. All footprints are outside the ignored boundary band.
MASK = np.zeros((9, 30), bool)
MASK[:, 10:23] = True
CAMERA = Camera.look_at((0, 0, 0), (0, 0, 2), width=30, height=9)
FOOTPRINTS = ({12: 1., 16: 1.}, {20: 1.}, {2: 1., 5: 1.}, {16: .5, 27: 1.}, {27: 1.})
SELECTED = np.array([True, True, False, True, False])
CLEANED = np.array([True, True, False, False, False])


class FakeRenderer:
    n = 5

    def __init__(self, *, blocked=False, fail=False, empty=False):
        self.blocked, self.fail, self.empty = blocked, fail, empty
        self.started, self.release = threading.Event(), threading.Event()
        self.lift_threads, self.image_calls, self.depth_calls = [], [], []

    def lift(self, camera, labels, *, active=None):
        self.lift_threads.append(threading.get_ident())
        if self.blocked and len(self.lift_threads) == 1:
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
        window.close()
    app.processEvents()


def extract_object(app, window):
    window.start_extraction()
    assert window.extraction_job is not None
    wait_until(app, lambda: window.extraction_job is None)


def test_extract_cleans_hidden_fragments_previews_and_exports_full_gaussians(app, make_window, tmp_path, dialogs):
    window, renderer = make_window()
    original = window.scene.copy()
    extract_object(app, window)

    np.testing.assert_array_equal(window.stages["selected"], SELECTED)
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)
    assert renderer.lift_threads and all(t != threading.get_ident() for t in renderer.lift_threads)
    assert "Removed: 1" in window.result_label.text()
    assert window.preview_box.currentText() == "Object only"
    assert window.export_action.isEnabled() and not window.select_action.isEnabled()
    window.set_selecting(True)
    window.viewport.add_point(10, 10, 1)
    assert not window.viewport.selecting and not window.viewport.points

    for background, color in (("White", 255), ("Black", 0)):
        window.background_box.setCurrentText(background)
        window.viewport.render_now()
        active, rgb = renderer.image_calls[-1]
        np.testing.assert_array_equal(active, CLEANED)
        assert rgb == (color / 255,) * 3
        np.testing.assert_array_equal(window.viewport.pixels[0, 0], [color] * 3)
        assert window.viewport.pick(0, 0) is not None
        np.testing.assert_array_equal(renderer.depth_calls[-1], CLEANED)

    window.preview_box.setCurrentText("Scene")
    window.viewport.render_now()
    assert renderer.image_calls[-1] == (None, BACKGROUND)
    assert window.select_action.isEnabled() and not window.background_box.isEnabled()
    assert window.export_action.isEnabled()
    assert window.export_ply(tmp_path / "object.ply")
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
        monkeypatch.setattr("gs_object_extraction.renderer.GraphdecoRenderer", lambda scene: FakeRenderer())
        assert window.open_ply(replacement)
        assert window.source_path == replacement.resolve() and not window.views
    assert window.stages is None and window.viewport.active is None
    assert window.preview_box.currentText() == "Scene"
    assert not window.export_action.isEnabled() and not window.preview_box.isEnabled()
    assert not window.export_ply(tmp_path / "stale.ply")
    assert not (tmp_path / "stale.ply").exists()


def test_extracting_locks_renderer_and_inputs_then_cancellation_restores_controls(app, make_window, tmp_path, dialogs):
    window, renderer = make_window(FakeRenderer(blocked=True))
    window.start_extraction()
    wait_until(app, renderer.started.is_set)
    job = window.extraction_job
    frame_count = len(renderer.image_calls)
    original_eye = window.viewport.orbit.eye.copy()
    assert window.viewport.suspended and not window.viewport.isEnabled()
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
    window.viewport.render_now()
    assert window.viewport.pick(0, 0) is None
    assert len(window.views) == 1 and not window.viewport.points
    np.testing.assert_array_equal(window.viewport.orbit.eye, original_eye)
    assert len(renderer.image_calls) == frame_count and not renderer.depth_calls
    window.cancel_extraction()
    assert not window.cancel_button.isEnabled()
    renderer.release.set()
    wait_until(app, lambda: window.extraction_job is None)
    assert window.stages is None and not window.viewport.suspended and window.viewport.isEnabled()
    assert window.open_action.isEnabled() and window.extract_action.isEnabled()
    assert not window.progress.isVisible() and not window.cancel_button.isVisible()
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


def test_empty_extraction_has_no_preview_or_export(app, make_window, tmp_path, dialogs):
    window, _ = make_window(FakeRenderer(empty=True))
    extract_object(app, window)
    assert not window.stages["cleaned"].any()
    assert "No object Gaussians remain" in window.result_label.text()
    assert window.viewport.active is None and window.preview_box.currentText() == "Scene"
    assert not window.preview_box.isEnabled() and not window.export_action.isEnabled()
    assert not window.export_ply(tmp_path / "empty.ply") and not dialogs


def test_export_failure_preserves_existing_target_and_result(app, make_window, tmp_path, monkeypatch, dialogs):
    window, _ = make_window()
    extract_object(app, window)
    target = tmp_path / "object.ply"
    target.write_bytes(b"previous export")
    before = set(tmp_path.iterdir())

    def fail_after_partial_write(scene, path):
        path.write_bytes(b"partial output")
        raise OSError("test disk write failure")

    monkeypatch.setattr("gs_object_extraction.app.window.save_ply", fail_after_partial_write)
    assert not window.export_ply(target)
    assert target.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == before
    np.testing.assert_array_equal(window.stages["cleaned"], CLEANED)
    assert window.export_action.isEnabled()
    assert dialogs == [("Cannot export object", "test disk write failure")]
    assert QApplication.overrideCursor() is None


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
