"""3D view widget: renders the scene on the GPU and turns mouse input into orbit moves.

Left drag orbits, right or middle drag pans, the wheel zooms toward the orbit
centre, and a double-click on the scene makes that point the orbit centre.
"""

import time
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget
from .orbit import unproject

BACKGROUND = (.13, .13, .14)
MIN_PICK_ALPHA = .5


class Viewport(QWidget):
    rendered = Signal(float)  # milliseconds spent on the last frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.renderer = None
        self.orbit = None
        self.image = None
        self.camera = None  # camera of the frame on screen
        self._drag = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.render_now)

    def set_scene(self, renderer, orbit):
        self.renderer, self.orbit = renderer, orbit
        self.request()

    def clear(self):
        self.renderer = self.orbit = self.image = self.camera = None
        self.update()

    def request(self):
        """Render on the next event-loop turn; repeated requests collapse into one frame."""
        self._timer.start()

    def render_now(self):
        if self.renderer is None or self.orbit is None:
            return
        ratio = self.devicePixelRatioF()
        width, height = max(int(self.width() * ratio), 1), max(int(self.height() * ratio), 1)
        start = time.perf_counter()
        self.camera = self.orbit.camera(width, height)
        pixels = self.renderer.render_image(self.camera, background=BACKGROUND)
        self.image = QImage(pixels.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.rendered.emit((time.perf_counter() - start) * 1000)
        self.update()

    def pick(self, x, y):
        """World point under widget position (x, y), or None where nothing solid is drawn."""
        if self.renderer is None or self.camera is None:
            return None
        depth, alpha = self.renderer.depth_image(self.camera)
        scale = self.camera.width / max(self.width(), 1)
        px, py = min(int(x * scale), self.camera.width - 1), min(int(y * scale), self.camera.height - 1)
        if alpha[py, px] < MIN_PICK_ALPHA:
            return None
        return unproject(self.camera, px, py, float(depth[py, px]))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor.fromRgbF(*BACKGROUND))
        if self.image is not None:
            painter.drawImage(self.rect(), self.image)
        else:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.rect(), Qt.AlignCenter, "Open a Gaussian PLY file (Ctrl+O)")

    def resizeEvent(self, event):
        self.request()

    def mousePressEvent(self, event):
        self._drag = (event.button(), event.position())

    def mouseMoveEvent(self, event):
        if self._drag is None or self.orbit is None:
            return
        button, last = self._drag
        delta = event.position() - last
        self._drag = (button, event.position())
        if button == Qt.LeftButton:
            self.orbit.rotate(delta.x(), delta.y())
        elif button in (Qt.RightButton, Qt.MiddleButton):
            self.orbit.pan(delta.x(), delta.y(), self.height())
        self.request()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton or self.orbit is None:
            return
        point = self.pick(event.position().x(), event.position().y())
        if point is not None:
            self.orbit.look_from(self.orbit.eye, point)
            self.request()

    def wheelEvent(self, event):
        if self.orbit is not None:
            self.orbit.zoom(event.angleDelta().y() / 120)
            self.request()
