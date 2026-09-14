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
