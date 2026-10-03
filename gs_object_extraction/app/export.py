"""Background PLY writing, so the window keeps painting while a large object is saved."""

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QPushButton, QVBoxLayout

from ..ply import save_ply
from ..upright import grounded, stood_up
from .orbit import DEFAULT_UP

DETAIL = ("Keep all", "Degree 2", "Degree 1", "Degree 0")  # SH degree 3 down to 0, by position


@dataclass
class ExportOptions:
    """How the object is turned and trimmed on its way out; the defaults save it as it is."""

    upright: bool = False
    ground: bool = False
    sh_degree: int = 3

    def apply(self, scene, up):
        """The scene as it will be written; ``up`` is the Up axis the object was worked on with."""
        if self.upright:
            scene, up = stood_up(scene, up, DEFAULT_UP), DEFAULT_UP
        if self.ground:
            scene = grounded(scene, up)
        if self.sh_degree < 3:
            scene = scene.with_sh_degree(self.sh_degree)
        return scene


class ExportDialog(QDialog):
    """The choices before saving, with the confirming button on the right."""

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export options")
        self.upright_box = QCheckBox("Save standing on its floor")
        self.upright_box.setToolTip("Turn the object so the Up axis points the way the Graphdeco viewers expect (-Y). "
                                    "Left off, the object keeps the scene's own orientation.")
        self.ground_box = QCheckBox("Put its floor at the origin")
        self.ground_box.setToolTip("Move the object so its lowest point along the Up axis is at zero "
                                   "and it is centred over the origin. Left off, it keeps its place in the scene.")
        self.detail_box = QComboBox()
        self.detail_box.addItems(DETAIL)
        self.detail_box.setToolTip("Colour detail to save. Lower degrees make a smaller file "
                                   "and colours that change less with the viewing direction.")
        self.upright_box.setChecked(options.upright)
        self.ground_box.setChecked(options.ground)
        self.detail_box.setCurrentIndex(3 - options.sh_degree)
        form = QFormLayout()
        form.addRow(self.upright_box)
        form.addRow(self.ground_box)
        form.addRow("Colour detail", self.detail_box)
        cancel, ok = QPushButton("Cancel"), QPushButton("OK")
        ok.setDefault(True)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        column = QVBoxLayout(self)
        column.addLayout(form)
        column.addLayout(buttons)

    def options(self):
        return ExportOptions(self.upright_box.isChecked(), self.ground_box.isChecked(), 3 - self.detail_box.currentIndex())


class ExportJob(QThread):
    """Write one object PLY off the GUI thread.

    The scene handed over is the window's own private copy of the trimmed object, so
    nothing on the GUI thread can change it while this one writes it out.
    """

    succeeded = Signal(int, str)  # Gaussians written, where they went
    failed = Signal(str)

    def __init__(self, scene, path, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.path = path

    def run(self):
        try:
            save_ply(self.scene, self.path)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(len(self.scene), str(self.path))
