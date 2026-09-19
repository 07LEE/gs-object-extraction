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
BUSY_RETRY_MS = 16  # wait this long before asking again for a renderer a background job is using
MASK_RGBA = (255, 140, 0, 110)
PARTIAL = 2.  # a candidate this much larger than the mask means the click probably caught a part
POINT_COLORS = {1: QColor(60, 220, 90), 0: QColor(235, 60, 60)}


class Viewport(QWidget):
    rendered = Signal(float)  # milliseconds spent on the last frame
    render_failed = Signal(str)
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
        self.render_error = None  # message of the last failed render, shown instead of a stale frame
        self.pixels = None  # uint8 RGB of the frame on screen
        self.camera = None  # camera of the frame on screen
        self.frame = 0  # increments with every rendered frame; keys the SAM2 embedding
        self.points, self.labels = [], []
        self.mask = self.score = self._overlay = None
        self.bigger = self.bigger_score = None  # a larger mask SAM2 offered for the same click
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
        self.renderer = self.orbit = self.image = self.pixels = self.camera = self.render_error = None
        self.active, self.background = None, BACKGROUND
        self.clear_prompts()

    def set_preview(self, active=None, background=BACKGROUND, renderer=None):
        """Render and pick only the selection while keeping the current camera.

        ``renderer`` swaps in another renderer, which is how the object is shown
        on its own: it is drawn from Gaussians of its own, not from the scene.
        """
        if renderer is not None:
            self.renderer = renderer
        if active is not None:
            active = np.asarray(active)
            if self.renderer is None or active.shape != (self.renderer.n,) or active.dtype != np.bool_:
                raise ValueError("preview selection must be a boolean array of shape N")
            self.set_selecting(False)
        self.active, self.background = active, background
        self.view_changed()

    def set_suspended(self, on):
        """A background job is running: the view can still be moved, but not marked.

        The job and this widget share one renderer, which hands out exclusive use per
        call, so the view keeps drawing between the job's own renders. What stops is
        marking: the job reads the views marked so far and appends to them.
        """
        self.suspended = bool(on)
        self._press, self._dragging = None, False
        self.clear_prompts()
        self.request()

    def request(self, delay=0):
        """Render on the next event-loop turn; repeated requests collapse into one frame."""
        self._timer.start(delay)

    def view_changed(self):
        """The camera or the widget size changed: prompts on the old frame no longer apply."""
        self.pixels = self.camera = None
        self.clear_prompts()
        self.request()

    def render_now(self):
        if self.renderer is None or self.orbit is None:
            return
        ratio = self.devicePixelRatioF()
        width, height = max(int(self.width() * ratio), 1), max(int(self.height() * ratio), 1)
        start = time.perf_counter()
        camera = self.orbit.camera(width, height)
        try:
            with self.renderer.held(blocking=False) as free:
                if not free:  # a job has the renderer; keep the frame on screen and ask again
                    self.request(BUSY_RETRY_MS)
                    return
                pixels = self.renderer.render_image(camera, active=self.active, background=self.background)
        except Exception as exc:  # e.g. CUDA out of memory: show why instead of a stale frame
            self.image = self.pixels = self.camera = None
            self.render_error = str(exc)
            self.clear_prompts()
            self.render_failed.emit(self.render_error)
            return
        self.camera, self.pixels, self.render_error = camera, pixels, None
        self.image = QImage(self.pixels.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.frame += 1
        self.clear_prompts()
        self.rendered.emit((time.perf_counter() - start) * 1000)
        self.update()

    def pick(self, x, y):
        """World point under widget position (x, y), or None where nothing solid is drawn."""
        if self.renderer is None or self.camera is None:
            return None
        with self.renderer.held(blocking=False) as free:
            if not free:  # a job has the renderer; a double-click is not worth waiting for
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
        if not self._segment():
            self.points.pop()
            self.labels.pop()
            self.update()
            self.prompts_changed.emit()

    def undo_point(self):
        if not self.points:
            return
        point, label = self.points.pop(), self.labels.pop()
        if not self.points:
            self.clear_prompts()
        elif not self._segment():
            self.points.append(point)  # keep the prompts that match the mask on screen
            self.labels.append(label)
            self.update()
            self.prompts_changed.emit()

    def clear_prompts(self):
        self.points, self.labels = [], []
        self.mask = self.score = self._overlay = None
        self.bigger = self.bigger_score = None
        self.update()
        self.prompts_changed.emit()

    def take_bigger(self):
        """Swap in the larger mask SAM2 offered for the same click."""
        if self.bigger is None:
            return False
        self.mask, self.score = self.bigger, self.bigger_score
        self.bigger = self.bigger_score = None
        self._show_mask()
        self.prompts_changed.emit()
        return True

    def _show_mask(self):
        mask = self.mask
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask] = MASK_RGBA
        self._overlay = QImage(rgba.data, mask.shape[1], mask.shape[0], 4 * mask.shape[1],
                               QImage.Format_RGBA8888).copy()
        self.update()

    def _segment(self):
        """Mask for the current points; False when SAM2 fails, leaving points and mask for the caller to restore.

        A lone click on a plant or a box often comes back with one leaf or one face.
        SAM2 offers other candidates for the same click, and a much larger one is the
        sign of that, so it is kept aside for the user to take in one go.
        """
        if 1 not in self.labels:  # background points alone do not describe an object
            self.mask = self.score = self._overlay = None
            self.bigger = self.bigger_score = None
            self.update()
            self.prompts_changed.emit()
            return True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            masks, scores = self.segmenter.candidates(self.pixels, self.frame, self.points, self.labels)
        except Exception as exc:
            error = str(exc)
        else:
            error = None
        finally:
            QApplication.restoreOverrideCursor()  # before any dialog, so it does not show a wait cursor
        if error is not None:
            self.failed.emit(error)
            return False
        best = int(np.argmax(scores))
        self.mask, self.score = masks[best], float(scores[best])
        areas = np.array([mask.sum() for mask in masks], dtype=float)
        widest = int(np.argmax(areas))
        self.bigger = self.bigger_score = None
        if areas[widest] > PARTIAL * max(areas[best], 1):
            self.bigger, self.bigger_score = masks[widest], float(scores[widest])
        self._show_mask()
        self.prompts_changed.emit()
        return True

    def set_selecting(self, on):
        self.selecting = bool(on) and self.active is None and not self.suspended
        self.setCursor(Qt.CrossCursor if self.selecting else Qt.ArrowCursor)

    # Qt events

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor.fromRgbF(*self.background))
        if self.image is None:
            painter.setPen(QColor(170, 170, 170))
            message = f"Render failed: {self.render_error}" if self.render_error else "Open a Gaussian PLY file (Ctrl+O)"
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, message)
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
        steps = event.angleDelta().y() / 120
        if self.orbit is not None and steps:  # sideways scrolling does not zoom, so the points stay
            self.orbit.zoom(steps)
            self.view_changed()
