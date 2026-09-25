"""3D view widget: renders the scene on the GPU, turns mouse input into orbit moves and collects mask prompts.

Left drag orbits, right or middle drag pans, the wheel zooms toward the orbit
centre, and a double-click on the scene makes that point the orbit centre.
In select mode a left click adds an object point and a right click a background
point; SAM2 turns the points into a mask on the view on screen. Moving the view
drops points that were not confirmed.
"""

import time
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QToolButton, QWidget
from .orbit import unproject
from .pick import EDGES, project

BACKGROUND = (.13, .13, .14)
MIN_PICK_ALPHA = .5
CLICK_SLOP = 4  # pixels a press may move and still count as a click
PICK_RADIUS = 12  # pixels around a click in the object preview that are selected with it
HIGHLIGHT_RGBA = (255, 40, 40, 255)
BOX_COLOR = QColor(255, 190, 0)
OVERLAY_MARGIN = 10  # pixels between the corner of the view and the controls drawn over it
OVERLAY_STYLE = ("QToolButton, QLabel { background: rgba(0, 0, 0, 120); color: white; border-radius: 4px; padding: 4px; }"
                 "QToolButton:checked { background: rgba(230, 150, 0, 210); }"
                 "QToolButton:disabled { background: rgba(0, 0, 0, 50); }")
BUSY_RETRY_MS = 16  # wait this long before asking again for a renderer a background job is using
MASK_RGBA = (255, 140, 0, 110)
OUTLINE_RGBA = (255, 255, 255, 230)
OUTLINE = 2  # pixels
PARTIAL = 2.  # a candidate this much larger than the mask means the click probably caught a part
POINT_COLORS = {1: QColor(60, 220, 90), 0: QColor(235, 60, 60)}


def _mask_image(mask):
    """The mask as a translucent fill with a solid outline, which shows on an object of any colour."""
    inner = mask.copy()
    for _ in range(OUTLINE):  # peel one pixel per pass, so what is left of the mask is its outline
        peeled = inner.copy()
        peeled[1:] &= inner[:-1]
        peeled[:-1] &= inner[1:]
        peeled[:, 1:] &= inner[:, :-1]
        peeled[:, :-1] &= inner[:, 1:]
        peeled[0] = peeled[-1] = False
        peeled[:, 0] = peeled[:, -1] = False
        inner = peeled
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask] = MASK_RGBA
    rgba[mask & ~inner] = OUTLINE_RGBA
    return QImage(rgba.data, mask.shape[1], mask.shape[0], 4 * mask.shape[1], QImage.Format_RGBA8888).copy()


def _icon(draw, size=22):
    """An icon drawn in white (or as ``draw`` chooses) rather than loaded, so there is no file to ship."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(255, 255, 255), 1.6))
    draw(painter, size)
    painter.end()
    return QIcon(pixmap)


def _draw_cube(painter, size):
    a, b, c = size * .12, size * .38, size * .88  # the back square is up and to the right of the front one
    front = [QPointF(a, b), QPointF(c - (b - a), b), QPointF(c - (b - a), c), QPointF(a, c)]
    back = [QPointF(p.x() + (b - a), p.y() - (b - a)) for p in front]
    for square in (front, back):
        painter.drawPolygon(square)
    for near, far in zip(front, back):
        painter.drawLine(near, far)


def _draw_object_only(painter, size):
    """A dashed frame for the scene with the object solid inside it."""
    painter.setPen(QPen(QColor(255, 255, 255, 170), 1.3, Qt.DashLine))
    painter.drawRoundedRect(QRectF(size * .08, size * .08, size * .84, size * .84), 3, 3)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    painter.drawRoundedRect(QRectF(size * .32, size * .32, size * .36, size * .36), 2, 2)


def _draw_background(painter, size, black):
    painter.setPen(QPen(QColor(255, 255, 255) if black else QColor(70, 70, 70), 1.6))
    painter.setBrush(QColor(0, 0, 0) if black else QColor(255, 255, 255))
    painter.drawEllipse(QRectF(size * .18, size * .18, size * .64, size * .64))


class Viewport(QWidget):
    rendered = Signal(float)  # milliseconds spent on the last frame
    render_failed = Signal(str)
    prompts_changed = Signal()
    failed = Signal(str)
    select_requested = Signal(float, float, float, float, int)  # a box in the frame's own pixels; 0 replaces, 1 adds to, 2 takes from the selection

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
        self.highlight = None  # world points of the Gaussians selected in the object preview, drawn as red dots
        self._highlight_image = None  # (frame, image) of those dots on the frame on screen
        self.box = None  # corners of the box round the object shown on its own
        self.show_box = True
        # Drawn over the view, top right: the object on its own or the scene, the background it stands on, the box round it and its size
        self.object_button = self._overlay_button(_icon(_draw_object_only), "Show the object on its own. Off: the whole scene")
        self.background_button = self._overlay_button(_icon(lambda p, s: _draw_background(p, s, False)), "")
        self.background_button.toggled.connect(self._show_background)
        self._show_background(False)
        self.box_button = self._overlay_button(_icon(_draw_cube), "Bounding box round the Gaussians that are left, so stray ones "
                                               "show as extra room. Its size is in the scene's own units")
        self.box_button.setChecked(True)
        self.box_button.toggled.connect(self.set_box_visible)
        self.box_label = QLabel(self)
        self.box_label.setStyleSheet(OVERLAY_STYLE)
        self.box_label.hide()
        self._box = None  # (start, current) of the selection box being dragged, in widget pixels
        self.image = None
        self.render_error = None  # message of the last failed render, shown instead of a stale frame
        self.pixels = None  # uint8 RGB of the frame on screen
        self.camera = None  # camera of the frame on screen
        self.frame = 0  # increments with every rendered frame; keys the SAM2 embedding
        self.points, self.labels = [], []
        self.mask = self.score = self._overlay = None
        self.bigger = self.bigger_score = None  # a larger mask SAM2 offered for the same click
        self.reviewing = None  # (MaskedView, its overlay) while a marked view is shown; moving the view ends it
        self._press = None
        self._dragging = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.render_now)

    def _overlay_button(self, icon, tip):
        button = QToolButton(self)
        button.setIcon(icon)
        button.setCheckable(True)
        button.setToolTip(tip)
        button.setStyleSheet(OVERLAY_STYLE)
        button.hide()
        return button

    def _show_background(self, black):
        self.background_button.setIcon(_icon(lambda p, s: _draw_background(p, s, black)))
        self.background_button.setToolTip("Background of the object on its own: " + ("black" if black else "white") +
                                          " (click for " + ("white" if black else "black") + ")")

    def _sync_overlay(self):
        """The three controls stay where they are once a scene is open; the ones that mean nothing for what is on screen are greyed out."""
        opened = self.renderer is not None
        for button in (self.object_button, self.background_button, self.box_button):
            button.setVisible(opened)
        self.background_button.setEnabled(self.active is not None)
        self.box_button.setEnabled(self.box is not None)
        self.box_label.setVisible(self.box is not None and self.show_box)
        self._place_overlay()

    def set_scene(self, renderer, orbit):
        self.renderer, self.orbit = renderer, orbit
        self.active, self.background = None, BACKGROUND
        self._sync_overlay()
        self.view_changed()

    def clear(self):
        self._timer.stop()
        self.renderer = self.orbit = self.image = self.pixels = self.camera = self.render_error = None
        self.active, self.background = None, BACKGROUND
        self.selecting, self.highlight, self._highlight_image, self._box, self.box = False, None, None, None, None
        self._sync_overlay()
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
        if (active is not None) != (self.active is not None):  # the scene and the object are marked and selected differently
            self.set_selecting(False)
            self.set_highlight(None)
        self.active, self.background = active, background
        self._sync_overlay()
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
        self.pixels = self.camera = self.reviewing = None
        self.clear_prompts()
        self.request()

    def show_view(self, view):
        """Stand where a marked view was taken and draw its mask and points over the frame.

        Moving the camera afterwards, or resizing the widget, puts the frame back to a plain one.
        """
        if self.orbit is None or self.renderer is None:
            return False
        camera = view.camera
        forward = camera.world_to_camera[2, :3]
        self.orbit.fov_y = float(np.rad2deg(2 * np.arctan(camera.height / 2 / camera.fy)))
        self.orbit.look_from(camera.eye, camera.eye + forward * self.orbit.distance)
        self.view_changed()
        self.reviewing = (view, _mask_image(view.mask))
        return True

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
        self.reviewing = None
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
        self._overlay = _mask_image(self.mask)
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
        """Click the scene to mark the object, or box-select Gaussians of the object shown on its own."""
        self.selecting = bool(on) and not self.suspended
        self._box = None
        self.setCursor(Qt.CrossCursor if self.selecting else Qt.ArrowCursor)
        self.update()

    def set_box(self, corners, size=None):
        """The box round the extracted object, drawn over the scene or the object alone as twelve lines, and its size; ``None`` for no box."""
        self.box = None if corners is None else np.asarray(corners, dtype=float)
        self.box_label.setText("" if size is None else " × ".join(f"{v:.2f}" for v in size))
        self._sync_overlay()
        self.update()

    def set_box_visible(self, on):
        self.show_box = bool(on)
        self.box_label.setVisible(self.box is not None and self.show_box)
        self.update()

    def _place_overlay(self):
        """Top right corner, from the right: the box, the background, the object or scene, and the box's size."""
        x = self.width() - OVERLAY_MARGIN
        top = OVERLAY_MARGIN
        for button in (self.box_button, self.background_button, self.object_button):
            button.adjustSize()
            if button.isVisibleTo(self):
                x -= button.width()
                button.move(x, top)
                x -= 6
        self.box_label.adjustSize()
        self.box_label.move(x - self.box_label.width(), top + (self.box_button.height() - self.box_label.height()) // 2)

    def set_highlight(self, points):
        """Draw the selected Gaussians as red dots; ``None`` for no selection."""
        self.highlight = None if points is None or not len(points) else np.asarray(points, dtype=float)
        self._highlight_image = None
        self.update()

    def _highlight_dots(self):
        camera = self.camera
        if self._highlight_image is not None and self._highlight_image[0] == self.frame:
            return self._highlight_image[1]
        x, y, depth = project(camera, self.highlight)
        near = depth > camera.near
        x, y = np.round(x[near]).astype(int), np.round(y[near]).astype(int)
        rgba = np.zeros((camera.height, camera.width, 4), np.uint8)
        keep = (x >= 1) & (x < camera.width - 1) & (y >= 1) & (y < camera.height - 1)
        x, y = x[keep], y[keep]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rgba[y + dy, x + dx] = HIGHLIGHT_RGBA
        image = QImage(rgba.data, camera.width, camera.height, 4 * camera.width, QImage.Format_RGBA8888).copy()
        self._highlight_image = (self.frame, image)
        return image

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
        if self.box is not None and self.show_box and self.camera is not None:
            x, y, depth = project(self.camera, self.box)
            scale = self.width() / self.camera.width
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(BOX_COLOR, 1.5))
            for i, j in EDGES:
                if depth[i] > self.camera.near and depth[j] > self.camera.near:  # a corner behind the camera has no place on screen
                    painter.drawLine(QPointF(x[i] * scale, y[i] * scale), QPointF(x[j] * scale, y[j] * scale))
        if self.highlight is not None and self.camera is not None:
            painter.drawImage(self.rect(), self._highlight_dots())
        if self._overlay is not None:
            painter.drawImage(self.rect(), self._overlay)
        if self.reviewing is not None:
            view, overlay = self.reviewing
            painter.drawImage(self.rect(), overlay)
            self._draw_points(painter, view.points, view.labels, view.camera.width)
        if self.points:
            self._draw_points(painter, self.points, self.labels, self.camera.width)
        if self._box is not None:
            start, end = self._box
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine))
            painter.setBrush(QColor(235, 60, 60, 60))
            painter.drawRect(QRectF(start, end).normalized())

    def _draw_points(self, painter, points, labels, width):
        scale = self.width() / width
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        for (px, py), label in zip(points, labels):
            painter.setBrush(POINT_COLORS[label])
            painter.drawEllipse(QPointF((px + .5) * scale, (py + .5) * scale), 5, 5)

    def resizeEvent(self, event):
        self._place_overlay()
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
        if button == Qt.LeftButton and self.selecting and self.active is not None:
            self._box = (start, event.position())
            self.update()
            return
        if button == Qt.LeftButton:
            self.orbit.rotate(delta.x(), delta.y())
        elif button in (Qt.RightButton, Qt.MiddleButton):
            self.orbit.pan(delta.x(), delta.y(), self.height())
        self.view_changed()

    def mouseReleaseEvent(self, event):
        press, self._press = self._press, None
        if press is not None and press[0] == Qt.LeftButton and self.selecting and self.active is not None and self.camera is not None:
            self._pick(press[1], event.position(), self._dragging, event.modifiers())
            return
        if press is None or self._dragging or not self.selecting:
            return
        if press[0] in (Qt.LeftButton, Qt.RightButton):
            self.add_point(event.position().x(), event.position().y(), 1 if press[0] == Qt.LeftButton else 0)

    def _pick(self, start, end, dragged, modifiers):
        """A drag selects inside its box, a click selects around the point; Shift adds, Ctrl takes away."""
        self._box = None
        self.update()
        if not dragged:
            start = QPointF(end.x() - PICK_RADIUS, end.y() - PICK_RADIUS)
            end = QPointF(end.x() + PICK_RADIUS, end.y() + PICK_RADIUS)
        mode = 1 if modifiers & Qt.ShiftModifier else 2 if modifiers & Qt.ControlModifier else 0
        scale = self.camera.width / max(self.width(), 1)
        self.select_requested.emit(start.x() * scale, start.y() * scale, end.x() * scale, end.y() * scale, mode)

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
