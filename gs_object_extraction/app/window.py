"""Mark an object in several views, extract it, preview it and export a Gaussian PLY."""

import os
from pathlib import Path
import stat
import tempfile
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QLabel, QListWidget, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QToolButton, QVBoxLayout, QWidget)
from ..ply import load_ply, save_ply
from ..extract import OFF_MASK
from .extraction import ExtractionJob
from .orbit import AXES, DEFAULT_UP, Orbit, estimate_up
from .segmenter import DEFAULT_CHECKPOINT, Segmenter
from .viewport import BACKGROUND, Viewport
from .views import MaskedView

TITLE = "3D Gaussian Splatting Object Extraction"
NAVIGATE_HINT = "Left drag: orbit   Right drag: pan   Wheel: zoom   Double-click: rotation centre   S: select"
SELECT_HINT = ("Left click: object point   Right click: background point   Backspace: undo   Esc: clear   "
               "Enter: add view   S: navigate")
PREVIEW_HINT = "Object preview   Drag: orbit / pan   Wheel: zoom   Ctrl+Shift+S: export"



def _file_mode(path):
    """Mode for a written file: keep an existing target's mode, otherwise the umask default."""
    if path.exists():
        return stat.S_IMODE(path.stat().st_mode)
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask

class MainWindow(QMainWindow):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1280, 800)
        self.scene = None
        self.source_path = None
        self.auto_up = None
        self.views = []
        self.stages = None
        self.extraction_job = None
        self._close_pending = False
        self.viewport = Viewport(self)
        self.viewport.segmenter = Segmenter(checkpoint)
        self.setCentralWidget(self.viewport)
        self.viewport.rendered.connect(lambda ms: self.statusBar().showMessage(f"Render {ms:.0f} ms"))
        self.viewport.render_failed.connect(lambda message: self.statusBar().showMessage(f"Render failed: {message}"))
        self.viewport.prompts_changed.connect(self.update_prompt_state)
        self.viewport.failed.connect(lambda message: QMessageBox.warning(self, "Cannot make a mask", message))

        self._make_actions()
        panel = QWidget()
        column = QVBoxLayout(panel)
        self.scene_box = self._scene_box()
        column.addWidget(self.scene_box)
        column.addWidget(self._object_box())
        column.addWidget(self._extraction_box())
        column.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        dock = QDockWidget("Scene", self)
        dock.setMinimumWidth(300)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addActions([self.open_action, self.export_action, self.quit_action])
        self.select_menu = self.menuBar().addMenu("&Select")
        self.select_menu.addActions([self.select_action, self.add_view_action, self.undo_action, self.clear_action])
        self.menuBar().addMenu("&Extract").addAction(self.extract_action)
        self.hint = QLabel(NAVIGATE_HINT)
        self.statusBar().addPermanentWidget(self.hint)
        self.statusBar().showMessage("Open a Gaussian PLY file")
        self.update_prompt_state()
        self.update_extraction_state()

    def _action(self, text, shortcut, slot, checkable=False):
        action = QAction(text, self)
        shortcuts = shortcut if isinstance(shortcut, tuple) else (shortcut,)
        action.setShortcuts([QKeySequence(s) for s in shortcuts])
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def _make_actions(self):
        self.open_action = self._action("&Open PLY...", QKeySequence.Open, self.choose_ply)
        self.quit_action = self._action("&Quit", QKeySequence.Quit, self.close)
        self.select_action = self._action("&Select mode", "S", self.set_selecting, checkable=True)
        self.add_view_action = self._action("&Add view", ("Return", "Enter"), self.add_view)  # main and keypad Enter
        self.undo_action = self._action("&Undo point", "Backspace", self.viewport.undo_point)
        self.clear_action = self._action("&Clear points", "Esc", self.viewport.clear_prompts)
        self.extract_action = self._action("&Extract object", "Ctrl+E", self.start_extraction)
        self.export_action = self._action("&Export object PLY...", "Ctrl+Shift+S", self.choose_export)

    def _scene_box(self):
        box = QGroupBox("Scene")
        form = QFormLayout(box)
        self.file_label = QLabel("-")
        self.count_label = QLabel("-")
        self.up_box = QComboBox()
        self.up_box.addItems(["Auto", *AXES])
        self.up_box.currentTextChanged.connect(self.set_up)
        reset = QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        form.addRow("File", self.file_label)
        form.addRow("Gaussians", self.count_label)
        form.addRow("Up axis", self.up_box)
        form.addRow(reset)
        return box

    def _object_box(self):
        box = QGroupBox("Object")
        column = QVBoxLayout(box)
        select = QToolButton()
        select.setDefaultAction(self.select_action)
        select.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.prompt_label = QLabel()
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setMinimumHeight(3 * self.prompt_label.fontMetrics().height())  # room for the mask line
        self.add_button = QPushButton("Add view")
        self.add_button.clicked.connect(self.add_view)
        self.view_list = QListWidget()
        self.view_list.currentRowChanged.connect(self.update_extraction_state)
        self.remove_button = QPushButton("Remove view")
        self.remove_button.clicked.connect(self.remove_view)
        for widget in (select, self.prompt_label, self.add_button, QLabel("Views"), self.view_list, self.remove_button):
            column.addWidget(widget)
        return box

    def _extraction_box(self):
        box = QGroupBox("Extraction")
        column = QVBoxLayout(box)
        self.extract_button = QPushButton("Extract object")
        self.extract_button.clicked.connect(self.start_extraction)
        self.off_threshold_box = QDoubleSpinBox()
        self.off_threshold_box.setRange(.05, .95)
        self.off_threshold_box.setSingleStep(.05)
        self.off_threshold_box.setValue(OFF_MASK)
        self.off_threshold_box.setToolTip("Lower removes more pieces that stray outside the mask, at the cost of thinner edges")
        self.result_label = QLabel("Add views, then extract the object")
        self.result_label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.hide()
        self.cancel_button = QPushButton("Cancel extraction")
        self.cancel_button.clicked.connect(self.cancel_extraction)
        self.cancel_button.hide()
        self.preview_box = QComboBox()
        self.preview_box.addItems(["Scene", "Object only"])
        self.preview_box.currentIndexChanged.connect(self.update_preview)
        self.background_box = QComboBox()
        self.background_box.addItems(["White", "Black"])
        self.background_box.currentIndexChanged.connect(self.update_preview)
        self.export_button = QPushButton("Export object PLY...")
        self.export_button.clicked.connect(self.choose_export)
        for widget in (QLabel("Off-mask limit"), self.off_threshold_box,
                       self.extract_button, self.result_label, self.progress, self.cancel_button,
                       QLabel("Preview"), self.preview_box, QLabel("Object background"),
                       self.background_box, self.export_button):
            column.addWidget(widget)
        return box

    def choose_ply(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Gaussian PLY", "", "PLY files (*.ply)")
        if path:
            self.open_ply(path)

    def open_ply(self, path):
        """Keep the scene on read failure; release it before allocating a new GPU scene."""
        if self.extraction_job is not None:
            return False
        from ..renderer import GraphdecoRenderer
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            scene = load_ply(path)
            self.scene = self.source_path = None
            self.invalidate_result()
            self.views.clear()
            self.view_list.clear()
            self.viewport.clear()  # release the previous scene's GPU memory first
            self.file_label.setText("-")
            self.file_label.setToolTip("")
            self.count_label.setText("-")
            renderer = GraphdecoRenderer(scene)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot open file", f"{Path(path).name}: {exc}")
            self.update_extraction_state()
            return False
        QApplication.restoreOverrideCursor()
        self.scene = scene
        self.source_path = Path(path).resolve()
        self.file_label.setText(Path(path).name)
        self.file_label.setToolTip(str(path))
        self.count_label.setText(f"{len(scene.means):,}")
        self.auto_up = estimate_up(scene.means)
        self.viewport.set_scene(renderer, Orbit.framing(scene.means, up=self.up_vector()))
        self.update_extraction_state()
        return True

    def up_vector(self):
        choice = self.up_box.currentText()
        if choice == "Auto":
            return DEFAULT_UP if self.auto_up is None else self.auto_up
        return AXES[choice]

    def set_up(self, choice):
        if self.extraction_job is not None:
            return
        if self.viewport.orbit is not None:
            self.viewport.orbit.set_up(self.up_vector())
            self.viewport.view_changed()

    def reset_view(self):
        if self.extraction_job is None and self.scene is not None:
            if self.viewport.active is None:
                self.viewport.orbit = Orbit.framing(self.scene.means, up=self.up_vector())
            else:
                # Fit the whole extracted object, including splat extents.
                import numpy as np
                active = self.viewport.active
                means, scales = self.scene.means[active], self.scene.scales[active]
                centre = np.median(means, axis=0)
                radius = float(np.max(np.linalg.norm(means - centre, axis=1) + 3 * scales.max(axis=1)))
                aspect = self.viewport.width() / max(self.viewport.height(), 1)
                half_fov = min(np.deg2rad(25), np.arctan(np.tan(np.deg2rad(25)) * aspect))
                self.viewport.orbit = Orbit(centre, max(1.1 * radius / np.sin(half_fov), 1e-6), up=self.up_vector())
            self.viewport.view_changed()

    def set_selecting(self, on):
        self.viewport.set_selecting(on)
        self.select_action.setChecked(self.viewport.selecting)
        self.hint.setText(PREVIEW_HINT if self.viewport.active is not None else
                          SELECT_HINT if self.viewport.selecting else NAVIGATE_HINT)
        self.update_prompt_state()

    def update_prompt_state(self):
        view = self.viewport
        if view.active is not None:
            text = "Choose Scene in Preview to mark more views"
        elif not view.points:
            text = "Click the object" if view.selecting else "Press S to mark the object"
        else:
            objects = sum(view.labels)
            text = f"{objects} object / {len(view.labels) - objects} background points"
            if view.mask is not None:
                text += f", mask {int(view.mask.sum()):,} px (score {view.score:.2f})"
            elif not objects:
                text += ", add an object point"
        self.prompt_label.setText(text)
        ready = (self.extraction_job is None and view.active is None and 1 in view.labels
                 and view.mask is not None and bool(view.mask.any()))
        self.add_button.setEnabled(ready)
        self.add_view_action.setEnabled(ready)

    def add_view(self):
        view = self.viewport
        if (self.extraction_job is not None or view.active is not None or 1 not in view.labels
                or view.mask is None or not view.mask.any()):
            return
        marked = MaskedView(view.camera, view.mask.copy(), tuple(view.points), tuple(view.labels))
        self.invalidate_result()
        self.views.append(marked)
        self.view_list.addItem(f"View {len(self.views)}: {int(marked.mask.sum()):,} px")
        view.clear_prompts()
        self.update_extraction_state()
        self.statusBar().showMessage(f"{len(self.views)} view(s) marked")

    def remove_view(self):
        if self.extraction_job is not None:
            return
        row = self.view_list.currentRow()
        if 0 <= row < len(self.views):
            self.invalidate_result()
            del self.views[row]
            self.view_list.takeItem(row)
            for i, marked in enumerate(self.views):
                self.view_list.item(i).setText(f"View {i + 1}: {int(marked.mask.sum()):,} px")
            self.update_extraction_state()

    def update_extraction_state(self, *_):
        busy = self.extraction_job is not None
        ready = self.scene is not None and self.viewport.renderer is not None and bool(self.views) and not busy
        result = self.stages is not None and bool(self.stages["cleaned"].any())
        preview = self.viewport.active is not None
        self.extract_action.setEnabled(ready)
        self.extract_button.setEnabled(ready)
        self.export_action.setEnabled(result and not busy)
        self.export_button.setEnabled(result and not busy)
        self.preview_box.setEnabled(result and not busy)
        self.background_box.setEnabled(result and preview and not busy)
        self.open_action.setEnabled(not busy)
        self.scene_box.setEnabled(not busy)
        self.off_threshold_box.setEnabled(not busy)
        self.view_list.setEnabled(not busy)
        self.remove_button.setEnabled(not busy and self.view_list.currentRow() >= 0)
        for action in (self.select_action, self.undo_action, self.clear_action):
            action.setEnabled(not busy and not preview)
        self.update_prompt_state()

    def invalidate_result(self):
        self.stages = None
        self.preview_box.setCurrentIndex(0)
        if self.viewport.active is not None:
            self.viewport.set_preview()
        self.result_label.setText("Add views, then extract the object")
        self.update_extraction_state()

    def update_preview(self, *_):
        if self.extraction_job is not None:
            return
        active = None
        background = BACKGROUND
        if self.preview_box.currentIndex() == 1 and self.stages is not None and self.stages["cleaned"].any():
            active = self.stages["cleaned"]
            background = (1., 1., 1.) if self.background_box.currentIndex() == 0 else (0., 0., 0.)
            self.set_selecting(False)
        self.viewport.set_preview(active, background)
        self.hint.setText(PREVIEW_HINT if active is not None else
                          SELECT_HINT if self.viewport.selecting else NAVIGATE_HINT)
        self.update_extraction_state()

    def start_extraction(self):
        if self.extraction_job is not None or self.scene is None or self.viewport.renderer is None or not self.views:
            return
        self.set_selecting(False)
        self.viewport.clear_prompts()
        self.viewport.set_suspended(True)
        job = ExtractionJob(self.viewport.renderer, tuple(self.views), self,
                            off_threshold=self.off_threshold_box.value())
        self.extraction_job = job
        job.progress.connect(self.extraction_progress)
        job.succeeded.connect(self.extraction_succeeded)
        job.failed.connect(self.extraction_failed)
        job.cancelled.connect(self.extraction_cancelled)
        job.finished.connect(self.extraction_finished)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.statusBar().showMessage("Extracting object...")
        self.update_extraction_state()
        job.start()

    def extraction_progress(self, done, total, stage):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        if self.cancel_button.isEnabled():
            self.statusBar().showMessage(f"{stage}: {done}/{total}")

    def extraction_succeeded(self, stages):
        self.stages = stages
        selected, cleaned = int(stages["selected"].sum()), int(stages["cleaned"].sum())
        self.result_label.setText(f"Selected: {selected:,}\nRemoved: {selected - cleaned:,}\nObject: {cleaned:,} Gaussians")
        if not cleaned:
            self.result_label.setText("No object Gaussians remain. Add or revise the marked views and extract again.")
        self.preview_box.setCurrentIndex(1 if cleaned else 0)
        self.statusBar().showMessage(f"Extraction complete: {cleaned:,} object Gaussians")

    def extraction_failed(self, message):
        self.statusBar().showMessage("Extraction failed")
        if not self._close_pending:
            QMessageBox.warning(self, "Cannot extract object", message)

    def extraction_cancelled(self):
        self.statusBar().showMessage("Extraction cancelled")

    def extraction_finished(self):
        job, self.extraction_job = self.extraction_job, None
        job.deleteLater()
        self.progress.hide()
        self.cancel_button.hide()
        self.update_preview()
        self.viewport.set_suspended(False)
        self.update_extraction_state()
        if self._close_pending:
            self.close()

    def cancel_extraction(self):
        if self.extraction_job is not None:
            self.extraction_job.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Cancelling after the current view...")

    def choose_export(self):
        if not self.export_action.isEnabled():
            return
        default = self.source_path.with_name(f"{self.source_path.stem}_object.ply") if self.source_path else Path("object.ply")
        dialog = QFileDialog(self, "Export object PLY", str(default.parent), "PLY files (*.ply)")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("ply")
        dialog.selectFile(default.name)
        if dialog.exec():
            self.export_ply(dialog.selectedFiles()[0])

    def export_ply(self, path):
        """Atomically save the cleaned subset, retaining Gaussian IDs and all attributes."""
        if self.extraction_job is not None or self.scene is None or self.stages is None or not self.stages["cleaned"].any():
            return False
        path = Path(path)
        temporary = error = None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if path.suffix.lower() != ".ply":
                raise ValueError("Use the .ply file extension for the object export.")
            if self.source_path is not None and (path.resolve() == self.source_path
                    or (path.exists() and self.source_path.exists() and path.samefile(self.source_path))):
                raise ValueError("Choose a different file from the source scene.")
            with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
            save_ply(self.scene.subset(self.stages["cleaned"]), temporary)
            os.chmod(temporary, _file_mode(path))  # the temporary file is private (0600) until now
            os.replace(temporary, path)
        except Exception as exc:
            error = str(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            QApplication.restoreOverrideCursor()  # before the dialog, so it does not show a wait cursor
        if error is not None:
            QMessageBox.warning(self, "Cannot export object", error)
            return False
        self.statusBar().showMessage(f"Saved {int(self.stages['cleaned'].sum()):,} Gaussians to {path}")
        return True

    def closeEvent(self, event):
        if self.extraction_job is not None:
            self._close_pending = True
            self.cancel_extraction()
            event.ignore()
            return
        super().closeEvent(event)
