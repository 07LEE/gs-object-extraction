"""Main window: open a Gaussian PLY, view it, and mark the object with clicks on several views."""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QFormLayout, QGroupBox, QLabel,
                               QListWidget, QMainWindow, QMessageBox, QPushButton, QToolButton, QVBoxLayout, QWidget)
from ..ply import load_ply
from .orbit import AXES, DEFAULT_UP, Orbit, estimate_up
from .segmenter import DEFAULT_CHECKPOINT, Segmenter
from .viewport import Viewport
from .views import MaskedView

TITLE = "3D Gaussian Splatting Object Extraction"
NAVIGATE_HINT = "Left drag: orbit   Right drag: pan   Wheel: zoom   Double-click: rotation centre   S: select"
SELECT_HINT = ("Left click: object point   Right click: background point   Backspace: undo   Esc: clear   "
               "Enter: add view   S: navigate")


class MainWindow(QMainWindow):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1280, 800)
        self.scene = None
        self.auto_up = None
        self.views = []
        self.viewport = Viewport(self)
        self.viewport.segmenter = Segmenter(checkpoint)
        self.setCentralWidget(self.viewport)
        self.viewport.rendered.connect(lambda ms: self.statusBar().showMessage(f"Render {ms:.0f} ms"))
        self.viewport.prompts_changed.connect(self.update_prompt_state)
        self.viewport.failed.connect(lambda message: QMessageBox.warning(self, "Cannot make a mask", message))

        self._make_actions()
        panel = QWidget()
        column = QVBoxLayout(panel)
        column.addWidget(self._scene_box())
        column.addWidget(self._object_box())
        column.addStretch()
        dock = QDockWidget("Scene", self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.file_menu = self.menuBar().addMenu("&File")
        self.file_menu.addActions([self.open_action, self.quit_action])
        self.select_menu = self.menuBar().addMenu("&Select")
        self.select_menu.addActions([self.select_action, self.add_view_action, self.undo_action, self.clear_action])
        self.hint = QLabel(NAVIGATE_HINT)
        self.statusBar().addPermanentWidget(self.hint)
        self.statusBar().showMessage("Open a Gaussian PLY file")
        self.update_prompt_state()

    def _action(self, text, shortcut, slot, checkable=False):
        action = QAction(text, self)
        action.setShortcut(QKeySequence(shortcut))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        self.addAction(action)
        return action

    def _make_actions(self):
        self.open_action = self._action("&Open PLY...", QKeySequence.Open, self.choose_ply)
        self.quit_action = self._action("&Quit", QKeySequence.Quit, self.close)
        self.select_action = self._action("&Select mode", "S", self.set_selecting, checkable=True)
        self.add_view_action = self._action("&Add view", "Return", self.add_view)
        self.undo_action = self._action("&Undo point", "Backspace", self.viewport.undo_point)
        self.clear_action = self._action("&Clear points", "Esc", self.viewport.clear_prompts)

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
        self.add_button = QPushButton("Add view")
        self.add_button.clicked.connect(self.add_view)
        self.view_list = QListWidget()
        remove = QPushButton("Remove view")
        remove.clicked.connect(self.remove_view)
        for widget in (select, self.prompt_label, self.add_button, QLabel("Views"), self.view_list, remove):
            column.addWidget(widget)
        return box

    def choose_ply(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Gaussian PLY", "", "PLY files (*.ply)")
        if path:
            self.open_ply(path)

    def open_ply(self, path):
        """Load and show a PLY; on failure keep the current scene and report why."""
        from ..renderer import GraphdecoRenderer
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            scene = load_ply(path)
            self.viewport.clear()  # release the previous scene's GPU memory first
            renderer = GraphdecoRenderer(scene)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Cannot open file", f"{Path(path).name}: {exc}")
            return False
        QApplication.restoreOverrideCursor()
        self.scene = scene
        self.views.clear()
        self.view_list.clear()
        self.file_label.setText(Path(path).name)
        self.file_label.setToolTip(str(path))
        self.count_label.setText(f"{len(scene.means):,}")
        self.auto_up = estimate_up(scene.means)
        self.viewport.set_scene(renderer, Orbit.framing(scene.means, up=self.up_vector()))
        return True

    def up_vector(self):
        choice = self.up_box.currentText()
        if choice == "Auto":
            return DEFAULT_UP if self.auto_up is None else self.auto_up
        return AXES[choice]

    def set_up(self, choice):
        if self.viewport.orbit is not None:
            self.viewport.orbit.set_up(self.up_vector())
            self.viewport.view_changed()

    def reset_view(self):
        if self.scene is not None:
            self.viewport.orbit = Orbit.framing(self.scene.means, up=self.up_vector())
            self.viewport.view_changed()

    def set_selecting(self, on):
        self.viewport.set_selecting(on)
        self.hint.setText(SELECT_HINT if on else NAVIGATE_HINT)
        self.update_prompt_state()

    def update_prompt_state(self):
        view = self.viewport
        if not view.points:
            text = "Click the object" if view.selecting else "Press S to mark the object"
        else:
            objects = sum(view.labels)
            text = f"{objects} object / {len(view.labels) - objects} background points"
            if view.mask is not None:
                text += f", mask {int(view.mask.sum()):,} px (score {view.score:.2f})"
        self.prompt_label.setText(text)
        ready = view.mask is not None and bool(view.mask.any())
        self.add_button.setEnabled(ready)
        self.add_view_action.setEnabled(ready)

    def add_view(self):
        view = self.viewport
        if view.mask is None or not view.mask.any():
            return
        self.views.append(MaskedView(view.camera, view.mask.copy(), tuple(view.points), tuple(view.labels)))
        self.view_list.addItem(f"View {len(self.views)}: {int(view.mask.sum()):,} px")
        view.clear_prompts()
        self.statusBar().showMessage(f"{len(self.views)} view(s) marked")

    def remove_view(self):
        row = self.view_list.currentRow()
        if 0 <= row < len(self.views):
            del self.views[row]
            self.view_list.takeItem(row)
            for i, marked in enumerate(self.views):
                self.view_list.item(i).setText(f"View {i + 1}: {int(marked.mask.sum()):,} px")
