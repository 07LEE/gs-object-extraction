import os
import threading
import time
import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication
from gs_object_extraction.app.orbit import Orbit
from gs_object_extraction.app.viewport import BACKGROUND, BUSY_RETRY_MS
from gs_object_extraction.app.window import MainWindow, TITLE
from gs_object_extraction.renderer import Exclusive
from gs_object_extraction.ply import save_ply
from gs_object_extraction.scene import GaussianScene

cuda = pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1",
                          reason="set GS_OBJECT_EXTRACTION_TEST_CUDA=1 in the GPU environment")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_window_starts_empty_with_open_and_quit(app):
    window = MainWindow()
    assert window.windowTitle() == TITLE
    assert [a.text() for a in window.file_menu.actions()] == ["&Open PLY...", "&Export object PLY...", "&Quit"]
    assert window.viewport.renderer is None and window.up_box.currentText() == "Auto"


def test_unreadable_file_is_reported_and_the_window_stays_empty(app, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.critical", lambda *args: shown.append(args[2]))
    bad = tmp_path / "bad.ply"
    bad.write_text("not a ply")
    window = MainWindow()
    assert open_and_wait(window, bad)  # the read happens in the background, the failure comes back from it
    assert "bad.ply" in shown[0] and window.scene is None and window.file_label.text() == "-"


def open_and_wait(window, path):
    """Open a scene the way the window does now, in the background, and let the job finish."""
    started = window.open_ply(path)
    while window.load_job is not None:
        QApplication.processEvents()
    return started


def mouse(kind, x, y, button, modifiers=Qt.NoModifier):
    buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, buttons, modifiers)


@cuda
def test_opened_scene_renders_and_drags_move_the_camera(app, tmp_path):
    rng = np.random.default_rng(0)
    scene = GaussianScene.from_colors(rng.normal(0, .2, (400, 3)), np.full((400, 3), .05),
                                      np.tile([.9, .4, .1], (400, 1)), np.full(400, .9))
    save_ply(scene, tmp_path / "blob.ply")
    window = MainWindow()
    window.resize(480, 360)
    window.show()
    assert open_and_wait(window, tmp_path / "blob.ply") and window.count_label.text() == "400"
    view = window.viewport
    view.render_now()
    cx, cy = view.image.width() // 2, view.image.height() // 2
    shown = view.image.pixelColor(cx, cy).getRgb()[:3]  # 8-bit values: QColor compares 16-bit channels
    assert shown != tuple(round(c * 255) for c in BACKGROUND)
    assert shown == tuple(int(v) for v in view.pixels[cy, cx])  # the frame on screen is the rendered one

    eye, target = view.orbit.eye.copy(), view.orbit.target.copy()
    for kind, x in ((QEvent.MouseButtonPress, 100), (QEvent.MouseMove, 160), (QEvent.MouseButtonRelease, 160)):  # Alt + left: orbit
        getattr(view, {QEvent.MouseButtonPress: "mousePressEvent", QEvent.MouseMove: "mouseMoveEvent",
                       QEvent.MouseButtonRelease: "mouseReleaseEvent"}[kind])(mouse(kind, x, 100, Qt.LeftButton, Qt.AltModifier))
    assert not np.allclose(view.orbit.eye, eye) and np.allclose(view.orbit.target, target)
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 100, Qt.MiddleButton))  # middle: pan, with no Alt
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 120, 90, Qt.MiddleButton))
    assert not np.allclose(view.orbit.target, target)


def test_the_camera_moves_with_alt_the_middle_or_the_right_button_and_zooms_on_an_alt_right_drag(app):
    window, view = ready_window(FakeSegmenter())
    view.set_scene(FlakyRenderer(), Orbit(np.zeros(3), 3.))
    view.render_now()
    eye, target, distance = view.orbit.eye.copy(), view.orbit.target.copy(), view.orbit.distance
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 100, Qt.LeftButton))  # a plain left drag is the tool's, not the camera's
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 180, 40, Qt.LeftButton))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 180, 40, Qt.LeftButton))
    assert np.allclose(view.orbit.eye, eye) and np.allclose(view.orbit.target, target)
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 100, Qt.RightButton))  # a plain right drag looks around
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 180, 40, Qt.RightButton))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 180, 40, Qt.RightButton))
    assert np.allclose(view.orbit.eye, eye) and not np.allclose(view.orbit.target, target)  # the camera stays, the view turns
    view.set_scene(FlakyRenderer(), Orbit(np.zeros(3), 3.))  # back where it began for the rest
    view.render_now()
    eye, target, distance = view.orbit.eye.copy(), view.orbit.target.copy(), view.orbit.distance
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 100, Qt.MiddleButton))  # the middle button always pans
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 180, 40, Qt.MiddleButton))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 180, 40, Qt.MiddleButton))
    assert not np.allclose(view.orbit.target, target) and np.isclose(view.orbit.distance, distance)
    view.orbit.target = target.copy()
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 100, Qt.RightButton, Qt.AltModifier))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 100, 60, Qt.RightButton, Qt.AltModifier))  # up is closer
    assert view.orbit.distance < distance and np.allclose(view.orbit.target, target)
    closer = view.orbit.distance
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 100, 120, Qt.RightButton, Qt.AltModifier))  # down is further
    assert view.orbit.distance > closer
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 100, 120, Qt.RightButton, Qt.AltModifier))


def test_an_alt_click_or_drag_in_select_mode_moves_the_camera_and_marks_nothing(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    view.set_scene(FlakyRenderer(), Orbit(np.zeros(3), 3.))
    view.render_now()
    window.select_action.trigger()
    eye = view.orbit.eye.copy()
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 80, Qt.LeftButton, Qt.AltModifier))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 100, 80, Qt.LeftButton, Qt.AltModifier))  # an Alt click
    assert view.points == [] and not fake.calls
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 80, Qt.LeftButton, Qt.AltModifier))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 160, 80, Qt.LeftButton, Qt.AltModifier))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 160, 80, Qt.LeftButton, Qt.AltModifier))
    assert not np.allclose(view.orbit.eye, eye) and view.points == [] and not fake.calls


@cuda
def test_double_click_on_the_object_makes_it_the_orbit_centre_without_moving_the_camera(app, tmp_path):
    scene = GaussianScene.from_colors(np.array([[0., 0., 0.]]), np.full((1, 3), .3), np.array([[.8, .8, .8]]), np.array([.99]))
    save_ply(scene, tmp_path / "ball.ply")
    window = MainWindow()
    window.resize(480, 360)
    window.show()
    open_and_wait(window, tmp_path / "ball.ply")
    view = window.viewport
    view.orbit = Orbit(np.array([2., 0., 0.]), 6.)  # look beside the ball first
    view.render_now()
    eye = view.orbit.eye.copy()
    camera = view.camera
    p = camera.world_to_camera @ np.array([0., 0., 0., 1.])
    x = (camera.fx * p[0] / p[2] + camera.cx) * view.width() / camera.width
    y = (camera.fy * p[1] / p[2] + camera.cy) * view.width() / camera.width
    view.mouseDoubleClickEvent(mouse(QEvent.MouseButtonDblClick, x, y, Qt.LeftButton))
    np.testing.assert_allclose(view.orbit.eye, eye, atol=1e-6)
    assert np.linalg.norm(view.orbit.target) < .6  # on the ball's near surface, not beside it
    assert view.pick(2, 2) is None  # empty background is not a pick


class FakeSegmenter:
    """A square around the click; ``part_of`` also offers the larger mask it belongs to."""

    def __init__(self, fail=False, part_of=0):
        self.calls, self.fail, self.part_of = [], fail, part_of

    def candidates(self, image, key, points, labels, *, several=None):
        if self.fail:
            raise FileNotFoundError("SAM2 checkpoint not found: x.pt (run scripts/setup_env.sh)")
        self.calls.append((key, list(points), list(labels)))
        x, y = points[0]
        masks, scores = [], []
        for reach, score in [(10, .9)] + ([(10 * self.part_of, .8)] if self.part_of else []):
            mask = np.zeros(image.shape[:2], bool)
            mask[max(0, y - reach):y + reach, max(0, x - reach):x + reach] = True
            masks.append(mask)
            scores.append(score)
        return np.stack(masks), np.array(scores)

    def predict(self, image, key, points, labels):
        masks, scores = self.candidates(image, key, points, labels)
        best = int(np.argmax(scores))
        return masks[best], float(scores[best])


def ready_window(segmenter):
    """A window whose viewport shows a frame without needing the GPU."""
    from PySide6.QtGui import QImage
    window = MainWindow()
    window.resize(480, 360)
    window.show()
    QApplication.processEvents()
    view = window.viewport
    view.segmenter = segmenter
    view.orbit = Orbit(np.zeros(3), 3.)
    view.camera = view.orbit.camera(view.width(), view.height())
    view.pixels = np.zeros((view.height(), view.width(), 3), np.uint8)
    view.image = QImage(view.width(), view.height(), QImage.Format_RGB888)
    view.frame = 1
    return window, view


def click(view, x, y, button):
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, x, y, button))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, x, y, button))


def drag(view, start, end, button, modifiers=Qt.NoModifier):
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, *start, button, modifiers))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, *end, button, modifiers))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, *end, button, modifiers))


def test_select_mode_clicks_make_a_mask_and_add_view_keeps_it(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    click(view, 100, 80, Qt.LeftButton)
    assert view.points == [] and not fake.calls  # navigation mode ignores clicks
    window.select_action.trigger()
    assert view.selecting and "object point" in window.hint.text()
    click(view, 100, 80, Qt.LeftButton)
    click(view, 200, 150, Qt.RightButton)
    assert view.labels == [1, 0] and view.points[0] == (100, 80) and fake.calls[-1][0] == view.frame
    assert "1 object / 1 background" in window.prompt_label.text() and window.add_button.isEnabled()
    window.undo_action.trigger()
    assert view.labels == [1] and view.mask is not None
    view.grab()  # paints the frame, the mask overlay and the points
    window.add_view_action.trigger()
    marked = window.views[0]
    assert len(window.views) == 1 and marked.labels == (1,) and marked.mask[80, 100]
    assert marked.mask.shape == (marked.camera.height, marked.camera.width)
    assert view.points == [] and window.view_list.count() == 1 and not window.add_button.isEnabled()


def test_moving_the_view_drops_unconfirmed_points_and_drags_are_not_clicks(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    eye = view.orbit.eye.copy()
    drag(view, (100, 80), (160, 80), Qt.LeftButton)  # without Alt the drag is a tool's, not the camera's: nothing moves
    assert view.points == [(100, 80)] and view.mask is not None and np.allclose(view.orbit.eye, eye)
    drag(view, (100, 80), (160, 80), Qt.LeftButton, Qt.AltModifier)  # with Alt the camera moves and the unconfirmed point goes
    assert view.points == [] and view.mask is None and not np.allclose(view.orbit.eye, eye)
    assert len(fake.calls) == 1


def test_segmentation_failure_is_reported_and_the_point_dropped(app, monkeypatch):
    shown = []
    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.warning", lambda *args: shown.append(args[2]))
    window, view = ready_window(FakeSegmenter(fail=True))
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    assert view.points == [] and view.mask is None and "setup_env.sh" in shown[0]


def test_removing_a_view_renumbers_the_rest(app):
    window, view = ready_window(FakeSegmenter())
    window.select_action.trigger()
    for x in (60, 120, 180):
        click(view, x, 80, Qt.LeftButton)
        window.add_view()
    window.view_list.setCurrentRow(0)
    window.remove_view()
    assert len(window.views) == 2 and window.views[0].points == ((120, 80),)
    assert window.view_list.item(0).text().startswith("View 1:")


@pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1",
                    reason="set GS_OBJECT_EXTRACTION_TEST_CUDA=1 in the GPU environment")
def test_real_sam2_click_marks_the_rendered_blob(app, tmp_path):
    from gs_object_extraction.app.segmenter import DEFAULT_CHECKPOINT
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("SAM2 checkpoint missing")
    rng = np.random.default_rng(0)
    scene = GaussianScene.from_colors(rng.normal(0, .2, (500, 3)), np.full((500, 3), .05),
                                      np.tile([.9, .4, .1], (500, 1)), np.full(500, .95))
    save_ply(scene, tmp_path / "blob.ply")
    window = MainWindow()
    window.resize(480, 360)
    window.show()
    open_and_wait(window, tmp_path / "blob.ply")
    view = window.viewport
    view.orbit = Orbit(np.zeros(3), 2.5)
    view.view_changed()
    view.render_now()
    window.select_action.trigger()
    click(view, view.width() / 2, view.height() / 2, Qt.LeftButton)
    assert view.mask is not None and view.mask[view.camera.height // 2, view.camera.width // 2]
    assert view.mask.mean() < .6  # the blob, not the whole frame
    window.add_view()
    assert len(window.views) == 1


def test_undo_that_fails_in_sam2_keeps_the_points_that_match_the_mask(app, monkeypatch):
    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.warning", lambda *args: None)
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    click(view, 200, 150, Qt.LeftButton)
    points, mask = list(view.points), view.mask.copy()
    fake.fail = True
    window.undo_action.trigger()
    assert view.points == points and np.array_equal(view.mask, mask)
    window.add_view()
    assert window.views[0].points == tuple(points)


def test_error_dialogs_do_not_show_a_wait_cursor(app, monkeypatch, tmp_path):
    cursors = []
    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.warning",
                        lambda *args: cursors.append(QApplication.overrideCursor()))
    window, view = ready_window(FakeSegmenter(fail=True))
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    window.scene = GaussianScene.from_colors(np.zeros((1, 3)), np.full((1, 3), .1), np.ones((1, 3)), np.ones(1))
    window.stages = {"selected": np.array([True]), "cleaned": np.array([True])}
    assert window.export_ply(tmp_path / "object.txt") is False
    assert cursors == [None, None] and QApplication.overrideCursor() is None


def test_background_points_alone_make_no_mask_and_cannot_be_added(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    window.select_action.trigger()
    click(view, 100, 80, Qt.RightButton)
    assert view.labels == [0] and view.mask is None and not fake.calls
    assert "add an object point" in window.prompt_label.text() and not window.add_button.isEnabled()
    window.add_view()
    assert window.views == []
    click(view, 200, 150, Qt.LeftButton)
    assert view.mask is not None and window.add_button.isEnabled()


def test_both_enter_keys_add_a_view(app):
    from PySide6.QtGui import QKeySequence
    shortcuts = MainWindow().add_view_action.shortcuts()
    assert QKeySequence("Return") in shortcuts and QKeySequence("Enter") in shortcuts


def test_sideways_wheel_does_not_drop_the_points(app):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QWheelEvent
    window, view = ready_window(FakeSegmenter())
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    distance = view.orbit.distance

    def wheel(dx, dy):
        view.wheelEvent(QWheelEvent(QPointF(100, 80), QPointF(100, 80), QPoint(0, 0), QPoint(dx, dy),
                                    Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    wheel(120, 0)
    assert view.points and view.orbit.distance == distance
    wheel(0, 120)
    assert view.points == [] and view.orbit.distance < distance


class FlakyRenderer(Exclusive):
    n = 10

    def __init__(self):
        super().__init__()
        self.fail = False

    def render_image(self, camera, active=None, background=(0, 0, 0)):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return np.zeros((camera.height, camera.width, 3), np.uint8)


def test_failed_render_is_shown_and_blocks_clicks_until_a_frame_renders(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    renderer = FlakyRenderer()
    view.set_scene(renderer, Orbit(np.zeros(3), 3.))
    view.render_now()
    renderer.fail = True
    view.view_changed()
    view.render_now()
    assert view.image is None and view.camera is None and "out of memory" in view.render_error
    assert "Render failed" in window.statusBar().currentMessage()
    view.grab()  # paints the failure message
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    assert view.points == [] and not fake.calls
    renderer.fail = False
    view.render_now()
    assert view.image is not None and view.render_error is None
    click(view, 100, 80, Qt.LeftButton)
    assert view.points and view.camera is not None


def test_unknown_sam2_checkpoint_name_stops_the_viewer_at_start(app, capsys):
    from gs_object_extraction.app.main import main
    with pytest.raises(SystemExit):
        main(["--sam2-checkpoint", "my_weights.pt"])
    assert "cannot tell the SAM2 model" in capsys.readouterr().err


def test_select_mode_drag_with_a_frame_drawn_before_release_adds_no_point(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    view.set_scene(FlakyRenderer(), Orbit(np.zeros(3), 3.))
    view.render_now()
    window.select_action.trigger()
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 80, Qt.LeftButton))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 160, 80, Qt.LeftButton))
    view.render_now()  # the app draws the frame while the button is still held
    assert view.pixels is not None
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 160, 80, Qt.LeftButton))
    assert view.points == [] and not fake.calls  # a drag without Alt is not a click, and it does not move the camera either


def test_small_jitter_during_a_click_still_counts_as_a_click(app):
    window, view = ready_window(FakeSegmenter())
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    eye = view.orbit.eye.copy()
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 200, 150, Qt.RightButton))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 202, 151, Qt.RightButton))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, 202, 151, Qt.RightButton))
    assert view.labels == [1, 0] and np.allclose(view.orbit.eye, eye)


def test_clicks_and_picks_map_widget_points_to_device_pixels_on_hidpi(app):
    from PySide6.QtGui import QImage
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    w, h = view.width(), view.height()
    view.camera = view.orbit.camera(2 * w, 2 * h)  # a frame rendered at devicePixelRatio 2
    view.pixels = np.zeros((2 * h, 2 * w, 3), np.uint8)
    view.image = QImage(2 * w, 2 * h, QImage.Format_RGB888)
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    assert view.points == [(200, 160)] and view.mask[160, 200]
    window.add_view()
    assert window.views[0].mask.shape == (2 * h, 2 * w)

    class OneSolidPixel(Exclusive):
        def depth_image(self, camera, active=None):
            alpha = np.zeros((camera.height, camera.width))
            alpha[160, 200] = 1
            return np.full(alpha.shape, 2.), alpha
    view.renderer = OneSolidPixel()
    assert view.pick(100, 80) is not None and view.pick(50, 40) is None


class RecordingPredictor:
    """Stands in for SAM2 and keeps every image it was asked to encode."""

    def __init__(self):
        self.images = []

    def set_image(self, image):
        self.images.append(np.array(image))

    def predict(self, point_coords, point_labels, multimask_output):
        count = 3 if multimask_output else 1
        masks = np.zeros((count, *self.images[-1].shape[:2]), bool)
        masks[:, :10, :10] = True
        return masks, np.linspace(.5, .9, count), None


class ShiftingRenderer(Exclusive):
    """Each frame is a different grey, so a stale embedding is detectable."""
    n = 10

    def __init__(self):
        super().__init__()
        self.frames = 0

    def render_image(self, camera, active=None, background=(0, 0, 0)):
        self.frames += 1
        return np.full((camera.height, camera.width, 3), 10 * self.frames, np.uint8)

    def depth_image(self, camera, *, active=None):
        shape = (camera.height, camera.width)
        return np.full(shape, 2.), np.ones(shape)


def test_a_busy_renderer_skips_the_frame_and_the_viewport_asks_again(app):
    """A job holds the renderer per call; the window keeps the frame it has instead of stalling."""
    import threading
    window, view = ready_window(FakeSegmenter())
    renderer = ShiftingRenderer()
    view.renderer = renderer
    before, frames = view.frame, renderer.frames
    taken, done = threading.Event(), threading.Event()

    def hold():
        with renderer.held():
            taken.set()
            assert done.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert taken.wait(5)
    view._timer.stop()
    view.render_now()
    assert view.frame == before and renderer.frames == frames  # nothing drawn
    assert view._timer.isActive() and view._timer.interval() == BUSY_RETRY_MS  # and it will try again
    done.set()
    holder.join(5)
    view.render_now()
    assert view.frame == before + 1 and renderer.frames == frames + 1


def test_each_new_frame_gets_a_new_sam2_embedding(app):
    from gs_object_extraction.app.segmenter import Segmenter
    predictor = RecordingPredictor()
    window, view = ready_window(Segmenter(predictor=predictor))
    view.set_scene(ShiftingRenderer(), Orbit(np.zeros(3), 3.))
    view.render_now()
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    click(view, 120, 90, Qt.LeftButton)
    assert len(predictor.images) == 1  # more points on the same frame reuse the embedding
    view.orbit.rotate(40, 0)
    view.view_changed()
    view.render_now()
    click(view, 100, 80, Qt.LeftButton)
    assert len(predictor.images) == 2 and np.array_equal(predictor.images[1], view.pixels)


class SlowSegmenter:
    """Loads only when released, so a test can look at the window mid-load."""

    def __init__(self):
        self.loaded = False
        self.builds = 0
        self.release = threading.Event()

    def load(self):
        assert self.release.wait(5)
        self.builds += 1
        self.loaded = True

    def warm(self):
        self.load()


def pump(app, predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.001)
    app.processEvents()
    assert predicate(), "the window did not reach the expected state before the timeout"


def slow_scene(monkeypatch, release):
    """Make opening a PLY block in the worker thread until release is set."""
    scene = GaussianScene.from_colors(np.zeros((4, 3)), np.full((4, 3), .1), np.full((4, 3), .5), np.full(4, .9))

    def load(path):
        assert release.wait(5)
        return scene

    monkeypatch.setattr("gs_object_extraction.app.loading.load_ply", load)
    monkeypatch.setattr("gs_object_extraction.renderer.GsplatRenderer", lambda scene: ShiftingRenderer())
    return scene


def test_the_window_stays_alive_while_a_scene_loads(app, tmp_path, monkeypatch):
    release = threading.Event()
    scene = slow_scene(monkeypatch, release)
    window = MainWindow()
    window.show()
    assert window.open_ply(tmp_path / "slow.ply")
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(1))
    pump(app, lambda: ticks)  # the event loop keeps turning: the window is not frozen
    assert window.progress.isVisible() and window.file_label.text() == "slow.ply"
    assert not window.open_action.isEnabled() and not window.scene_box.isEnabled()
    assert not window.open_ply(tmp_path / "other.ply")  # one load at a time
    release.set()
    pump(app, lambda: window.load_job is None)
    assert window.scene is scene and window.count_label.text() == "4"
    assert window.viewport.renderer is not None and window.open_action.isEnabled() and not window.progress.isVisible()


def test_closing_while_a_scene_loads_waits_for_the_thread(app, tmp_path, monkeypatch):
    release = threading.Event()
    slow_scene(monkeypatch, release)
    window = MainWindow()
    window.show()
    window.open_ply(tmp_path / "slow.ply")
    app.processEvents()
    assert window.close() is False and window.isVisible()  # a running thread cannot be destroyed
    release.set()
    pump(app, lambda: window.load_job is None and not window.isVisible())


def test_select_mode_loads_sam2_once_in_the_background(app):
    segmenter = SlowSegmenter()
    window, view = ready_window(segmenter)
    window.set_selecting(True)
    assert window.model_job is not None and not segmenter.loaded
    window.set_selecting(True)  # no second model for a second look at select mode
    assert "SAM2" in window.statusBar().currentMessage()
    segmenter.release.set()
    pump(app, lambda: window.model_job is None)
    assert segmenter.builds == 1 and segmenter.loaded
    window.set_selecting(False)
    window.set_selecting(True)
    assert window.model_job is None  # already loaded


def test_a_failed_sam2_load_is_reported_without_a_dialog(app, monkeypatch):
    class BrokenSegmenter(SlowSegmenter):
        def load(self):
            raise FileNotFoundError("SAM2 checkpoint not found: x.pt (run scripts/setup_env.sh)")

    window, view = ready_window(BrokenSegmenter())
    window.set_selecting(True)
    pump(app, lambda: window.model_job is None)
    assert "setup_env.sh" in window.statusBar().currentMessage()


def test_a_click_that_catches_a_part_offers_the_bigger_mask(app):
    """SAM2 answers a click on a leaf with the leaf; the whole plant is among its other candidates."""
    fake = FakeSegmenter(part_of=4)
    window, view = ready_window(fake)
    window.set_selecting(True)
    click(view, 100, 80, Qt.LeftButton)
    assert view.bigger is not None and window.bigger_button.isVisible()
    assert "part of the object" in window.prompt_label.text()
    small = int(view.mask.sum())
    window.bigger_button.click()
    assert view.mask.sum() > 4 * small and view.bigger is None and not window.bigger_button.isVisible()
    assert view.score == .8 and "px" in window.statusBar().currentMessage()
    taken = int(view.mask.sum())
    window.add_view()
    assert int(window.views[0].mask.sum()) == taken  # the view keeps the mask that was on screen


def test_a_click_that_catches_the_object_offers_nothing(app):
    window, view = ready_window(FakeSegmenter())
    window.set_selecting(True)
    click(view, 100, 80, Qt.LeftButton)
    assert view.bigger is None and not window.bigger_button.isVisible()
    assert not window.viewport.take_bigger()  # nothing to take
    window.undo_action.trigger()
    assert view.bigger is None and not window.bigger_button.isVisible()


def test_sam2_checkpoint_option_reaches_the_window(app, monkeypatch):
    import gs_object_extraction.app.main as main_module
    made = {}

    class FakeApp:
        def __init__(self, argv):
            pass

        def exec(self):
            return 0

    class FakeWindow:
        def __init__(self, checkpoint):
            made["checkpoint"] = checkpoint

        def show(self):
            pass

        def open_ply(self, path):
            made["ply"] = path

    monkeypatch.setattr(main_module, "QApplication", FakeApp)
    monkeypatch.setattr(main_module, "MainWindow", FakeWindow)
    assert main_module.main(["scene.ply", "--sam2-checkpoint", "/w/sam2.1_hiera_large.pt"]) == 0
    assert made == {"checkpoint": "/w/sam2.1_hiera_large.pt", "ply": "scene.ply"}


def test_a_file_that_cannot_be_read_leaves_the_open_scene_alone(app, tmp_path, monkeypatch):
    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.critical", lambda *args: None)
    window = MainWindow()
    window.scene = object()
    window.file_label.setText("kept.ply")
    window.views.append(object())
    bad = tmp_path / "bad.ply"
    bad.write_text("not a ply")
    open_and_wait(window, bad)
    assert window.scene is not None and len(window.views) == 1 and window.file_label.text() == "kept.ply"


def test_a_failed_upload_brings_the_previous_work_back(app, tmp_path, monkeypatch):
    def failing_upload(scene):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.critical", lambda *args: None)
    monkeypatch.setattr("gs_object_extraction.app.loading.load_ply", lambda path: GaussianScene.from_colors(
        np.zeros((2, 3)), np.full((2, 3), .1), np.full((2, 3), .5), np.full(2, .9)))
    monkeypatch.setattr("gs_object_extraction.renderer.GsplatRenderer", failing_upload)
    old = GaussianScene.from_colors(np.zeros((4, 3)), np.full((4, 3), .1), np.full((4, 3), .5), np.full(4, .9))
    window = MainWindow()
    window.renderer_factory = lambda scene: ShiftingRenderer()
    window.scene, window.scene_renderer, window.source_path = old, ShiftingRenderer(), tmp_path / "old.ply"
    window.file_label.setText("old.ply")
    window.count_label.setText("4")
    window.views.append(object())
    window.view_list.addItem("View 1: 9 px")
    assert open_and_wait(window, tmp_path / "new.ply")
    assert window.scene is old and window.source_path == tmp_path / "old.ply"
    assert len(window.views) == 1 and window.view_list.count() == 1
    assert window.file_label.text() == "old.ply" and window.count_label.text() == "4"
    assert window.viewport.renderer is window.scene_renderer is not None


def test_a_restore_that_fails_keeps_the_work_for_another_try(app, tmp_path, monkeypatch):
    def failing_upload(scene):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.critical", lambda *args: None)
    monkeypatch.setattr("gs_object_extraction.app.loading.load_ply", lambda path: GaussianScene.from_colors(
        np.zeros((2, 3)), np.full((2, 3), .1), np.full((2, 3), .5), np.full(2, .9)))
    monkeypatch.setattr("gs_object_extraction.renderer.GsplatRenderer", failing_upload)
    old = GaussianScene.from_colors(np.zeros((4, 3)), np.full((4, 3), .1), np.full((4, 3), .5), np.full(4, .9))
    window = MainWindow()
    window.renderer_factory = failing_upload
    window.scene, window.scene_renderer, window.source_path = old, ShiftingRenderer(), tmp_path / "old.ply"
    window.views.append(object())
    window.view_list.addItem("View 1: 9 px")
    open_and_wait(window, tmp_path / "new.ply")
    assert window.scene is old and len(window.views) == 1 and window.scene_renderer is None
    window.renderer_factory = lambda scene: ShiftingRenderer()  # the GPU has room again
    open_and_wait(window, tmp_path / "new.ply")  # another failed open: the views must not be doubled
    assert window.scene is old and len(window.views) == 1 and window.view_list.count() == 1
    assert window.scene_renderer is not None and window._kept is None


def test_a_right_click_still_marks_the_background_and_a_right_drag_looks_instead(app):
    fake = FakeSegmenter()
    window, view = ready_window(fake)
    view.set_scene(FlakyRenderer(), Orbit(np.zeros(3), 3.))
    view.render_now()
    window.select_action.trigger()
    click(view, 100, 80, Qt.LeftButton)
    click(view, 200, 150, Qt.RightButton)  # a click: a background point
    assert view.labels == [1, 0]
    eye, target = view.orbit.eye.copy(), view.orbit.target.copy()
    calls = len(fake.calls)
    drag(view, (200, 150), (260, 120), Qt.RightButton)  # a drag: the camera looks around, and marks nothing
    assert np.allclose(view.orbit.eye, eye) and not np.allclose(view.orbit.target, target)
    assert len(fake.calls) == calls and view.labels == []  # the view moved, so the unconfirmed points went, as with any camera move
