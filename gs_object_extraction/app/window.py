"""Main window: open a Gaussian PLY, view it, and choose the up axis."""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QFormLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QWidget)
from ..ply import load_ply
from .orbit import AXES, DEFAULT_UP, Orbit, estimate_up
from .viewport import Viewport

TITLE = "3D Gaussian Splatting Object Extraction"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITLE)
        self.resize(1280, 800)
        self.scene = None
        self.auto_up = None
        self.viewport = Viewport(self)
        self.setCentralWidget(self.viewport)
        self.viewport.rendered.connect(lambda ms: self.statusBar().showMessage(f"Render {ms:.0f} ms"))

        panel = QWidget()
        form = QFormLayout(panel)
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
        dock = QDockWidget("Scene", self)
        dock.setWidget(panel)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.file_menu = self.menuBar().addMenu("&File")
        self.open_action = QAction("&Open PLY...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self.choose_ply)
        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence.Quit)
        self.quit_action.triggered.connect(self.close)
        self.file_menu.addActions([self.open_action, self.quit_action])
        self.statusBar().addPermanentWidget(QLabel("Left drag: orbit   Right drag: pan   Wheel: zoom   "
                                                   "Double-click: rotation centre"))
        self.statusBar().showMessage("Open a Gaussian PLY file")

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
            self.viewport.request()

    def reset_view(self):
        if self.scene is not None:
            self.viewport.orbit = Orbit.framing(self.scene.means, up=self.up_vector())
            self.viewport.request()
