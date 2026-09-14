import os
import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import QApplication
from gs_object_extraction.app.orbit import Orbit
from gs_object_extraction.app.viewport import BACKGROUND
from gs_object_extraction.app.window import MainWindow, TITLE
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
    assert [a.text() for a in window.file_menu.actions()] == ["&Open PLY...", "&Quit"]
    assert window.viewport.renderer is None and window.up_box.currentText() == "Auto"


def test_unreadable_file_is_reported_and_the_window_stays_empty(app, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr("gs_object_extraction.app.window.QMessageBox.critical", lambda *args: shown.append(args[2]))
    bad = tmp_path / "bad.ply"
    bad.write_text("not a ply")
    window = MainWindow()
    assert window.open_ply(bad) is False
    assert "bad.ply" in shown[0] and window.scene is None


def mouse(kind, x, y, button):
    buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, buttons, Qt.NoModifier)


@cuda
def test_opened_scene_renders_and_drags_move_the_camera(app, tmp_path):
    rng = np.random.default_rng(0)
    scene = GaussianScene.from_colors(rng.normal(0, .2, (400, 3)), np.full((400, 3), .05),
                                      np.tile([.9, .4, .1], (400, 1)), np.full(400, .9))
    save_ply(scene, tmp_path / "blob.ply")
    window = MainWindow()
    window.resize(480, 360)
    window.show()
    assert window.open_ply(tmp_path / "blob.ply") and window.count_label.text() == "400"
    view = window.viewport
    view.render_now()
    centre = view.image.pixelColor(view.image.width() // 2, view.image.height() // 2)
    assert centre != QColor.fromRgbF(*BACKGROUND)

    eye, target = view.orbit.eye.copy(), view.orbit.target.copy()
    for kind, x in ((QEvent.MouseButtonPress, 100), (QEvent.MouseMove, 160), (QEvent.MouseButtonRelease, 160)):
        getattr(view, {QEvent.MouseButtonPress: "mousePressEvent", QEvent.MouseMove: "mouseMoveEvent",
                       QEvent.MouseButtonRelease: "mouseReleaseEvent"}[kind])(mouse(kind, x, 100, Qt.LeftButton))
    assert not np.allclose(view.orbit.eye, eye) and np.allclose(view.orbit.target, target)
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, 100, 100, Qt.RightButton))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, 120, 90, Qt.RightButton))
    assert not np.allclose(view.orbit.target, target)


@cuda
def test_double_click_on_the_object_makes_it_the_orbit_centre_without_moving_the_camera(app, tmp_path):
    scene = GaussianScene.from_colors(np.array([[0., 0., 0.]]), np.full((1, 3), .3), np.array([[.8, .8, .8]]), np.array([.99]))
    save_ply(scene, tmp_path / "ball.ply")
    window = MainWindow()
    window.resize(480, 360)
    window.show()
    window.open_ply(tmp_path / "ball.ply")
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
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def predict(self, image, key, points, labels):
        if self.fail:
            raise FileNotFoundError("SAM2 checkpoint not found: x.pt (run scripts/setup_env.sh)")
        self.calls.append((key, list(points), list(labels)))
        mask = np.zeros(image.shape[:2], bool)
        x, y = points[0]
        mask[max(0, y - 10):y + 10, max(0, x - 10):x + 10] = True
        return mask, .9


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


def drag(view, start, end, button):
    view.mousePressEvent(mouse(QEvent.MouseButtonPress, *start, button))
    view.mouseMoveEvent(mouse(QEvent.MouseMove, *end, button))
    view.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, *end, button))


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
    drag(view, (100, 80), (160, 80), Qt.LeftButton)
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
    window.open_ply(tmp_path / "blob.ply")
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
