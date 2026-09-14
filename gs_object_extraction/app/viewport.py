"""3D view widget: renders the scene on the GPU, turns mouse input into orbit moves and collects mask prompts.

Left drag orbits, right or middle drag pans, the wheel zooms toward the orbit
centre, and a double-click on the scene makes that point the orbit centre.
In select mode a left click adds an object point and a right click a background
point; SAM2 turns the points into a mask on the view on screen. Moving the view
drops points that were not confirmed.
"""

import time
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget
from .orbit import unproject

BACKGROUND = (.13, .13, .14)
MIN_PICK_ALPHA = .5
CLICK_SLOP = 4  # pixels a press may move and still count as a click
MASK_RGBA = (255, 140, 0, 110)
POINT_COLORS = {1: QColor(60, 220, 90), 0: QColor(235, 60, 60)}


class Viewport(QWidget):
    rendered = Signal(float)  # milliseconds spent on the last frame
    prompts_changed = Signal()
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.renderer = None
        self.active = None  # None for the scene, boolean selection for object preview
        self.background = BACKGROUND
        self.suspended = False
        self.orbit = None
        self.segmenter = None
        self.selecting = False
        self.image = None
        self.pixels = None  # uint8 RGB of the frame on screen
        self.camera = None  # camera of the frame on screen
        self.frame = 0  # increments with every rendered frame; keys the SAM2 embedding
        self.points, self.labels = [], []
        self.mask = self.score = self._overlay = None
        self._press = None
        self._dragging = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.render_now)

    def set_scene(self, renderer, orbit):
        self.renderer, self.orbit = renderer, orbit
        self.active, self.background = None, BACKGROUND
        self.view_changed()

    def clear(self):
        self._timer.stop()
        self.renderer = self.orbit = self.image = self.pixels = self.camera = None
        self.active, self.background = None, BACKGROUND
        self.clear_prompts()

    def set_preview(self, active=None, background=BACKGROUND):
        """Render and pick only the selection while keeping the current camera."""
        if active is not None:
            active = np.asarray(active)
            if self.renderer is None or active.shape != (self.renderer.n,) or active.dtype != np.bool_:
                raise ValueError("preview selection must be a boolean array of shape N")
            self.set_selecting(False)
        self.active, self.background = active, background
        self.view_changed()

    def set_suspended(self, on):
        """Give extraction exclusive use of the renderer until its thread finishes."""
        self.suspended = bool(on)
        self._press, self._dragging = None, False
        self.setEnabled(not on)
        if on:
            self._timer.stop()
        else:
            self.request()

    def request(self):
        """Render on the next event-loop turn; repeated requests collapse into one frame."""
        if not self.suspended:
            self._timer.start()

    def view_changed(self):
        """The camera or the widget size changed: prompts on the old frame no longer apply."""
        self.pixels = self.camera = None
        self.clear_prompts()
        self.request()

    def render_now(self):
        if self.suspended or self.renderer is None or self.orbit is None:
            return
        ratio = self.devicePixelRatioF()
        width, height = max(int(self.width() * ratio), 1), max(int(self.height() * ratio), 1)
        start = time.perf_counter()
        self.camera = self.orbit.camera(width, height)
        self.pixels = self.renderer.render_image(self.camera, active=self.active, background=self.background)
        self.image = QImage(self.pixels.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.frame += 1
        self.clear_prompts()
        self.rendered.emit((time.perf_counter() - start) * 1000)
        self.update()

    def pick(self, x, y):
        """World point under widget position (x, y), or None where nothing solid is drawn."""
        if self.suspended or self.renderer is None or self.camera is None:
            return None
        depth, alpha = self.renderer.depth_image(self.camera, active=self.active)
        px, py = self._to_pixels(x, y)
        if alpha[py, px] < MIN_PICK_ALPHA:
            return None
        return unproject(self.camera, px, py, float(depth[py, px]))

    def _to_pixels(self, x, y):
        scale = self.camera.width / max(self.width(), 1)
        return min(max(int(x * scale), 0), self.camera.width - 1), min(max(int(y * scale), 0), self.camera.height - 1)

    # Prompts

    def add_point(self, x, y, label):
        """Add a prompt at widget position (x, y) and update the mask."""
        if self.suspended or self.active is not None or self.pixels is None or self.camera is None or self.segmenter is None:
            return
        self.points.append(self._to_pixels(x, y))
        self.labels.append(int(label))
        self._segment()

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.labels.pop()
            self._segment()

    def clear_prompts(self):
        self.points, self.labels = [], []
        self.mask = self.score = self._overlay = None
        self.update()
        self.prompts_changed.emit()

    def _segment(self):
        if not self.points:
            self.clear_prompts()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mask, score = self.segmenter.predict(self.pixels, self.frame, self.points, self.labels)
        except Exception as exc:
            self.points.pop()  # keep the prompts that still have a valid mask
            self.labels.pop()
            self.failed.emit(str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.mask, self.score = mask, score
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask] = MASK_RGBA
        self._overlay = QImage(rgba.data, mask.shape[1], mask.shape[0], 4 * mask.shape[1], QImage.Format_RGBA8888).copy()
        self.update()
        self.prompts_changed.emit()

    def set_selecting(self, on):
        self.selecting = bool(on) and self.active is None and not self.suspended
        self.setCursor(Qt.CrossCursor if self.selecting else Qt.ArrowCursor)

    # Qt events

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor.fromRgbF(*self.background))
        if self.image is None:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a Gaussian PLY file (Ctrl+O)")
            return
        painter.drawImage(self.rect(), self.image)
        if self._overlay is not None:
            painter.drawImage(self.rect(), self._overlay)
        if self.points:
            scale = self.width() / self.camera.width
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5))
            for (px, py), label in zip(self.points, self.labels):
                painter.setBrush(POINT_COLORS[label])
                painter.drawEllipse(QPointF((px + .5) * scale, (py + .5) * scale), 5, 5)

    def resizeEvent(self, event):
        self.view_changed()

    def mousePressEvent(self, event):
        self._press = (event.button(), event.position(), event.position())
        self._dragging = False

    def mouseMoveEvent(self, event):
        if self._press is None or self.orbit is None:
            return
        button, start, last = self._press
        if not self._dragging and (event.position() - start).manhattanLength() <= CLICK_SLOP:
            return
        self._dragging = True
        delta = event.position() - last
        self._press = (button, start, event.position())
        if button == Qt.LeftButton:
            self.orbit.rotate(delta.x(), delta.y())
        elif button in (Qt.RightButton, Qt.MiddleButton):
            self.orbit.pan(delta.x(), delta.y(), self.height())
        self.view_changed()

    def mouseReleaseEvent(self, event):
        press, self._press = self._press, None
        if press is None or self._dragging or not self.selecting:
            return
        if press[0] in (Qt.LeftButton, Qt.RightButton):
            self.add_point(event.position().x(), event.position().y(), 1 if press[0] == Qt.LeftButton else 0)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton or self.orbit is None or self.selecting:
            return
        point = self.pick(event.position().x(), event.position().y())
        if point is not None:
            self.orbit.look_from(self.orbit.eye, point)
            self.view_changed()

    def wheelEvent(self, event):
        if self.orbit is not None:
            self.orbit.zoom(event.angleDelta().y() / 120)
            self.view_changed()
