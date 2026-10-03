"""Background PLY writing, so the window keeps painting while a large object is saved."""

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ..ply import save_ply
from ..upright import grounded, stood_up
from .orbit import DEFAULT_UP

COMPRESSION = ("None", "Low", "Medium", "High")  # colour terms kept: SH degree 3 down to 0, by position


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
        self.setMinimumWidth(320)
        self.upright_box = QCheckBox("Stand on its floor")
        self.upright_box.setToolTip("Turn the object so its Up axis points -Y, as in Graphdeco's own scenes. "
                                    "Left off, it keeps the scene's own orientation.")
        self.ground_box = QCheckBox("Floor at the origin")
        self.ground_box.setToolTip("Move the object so its lowest point is at zero and it is centred over the origin. "
                                   "Left off, it keeps its place in the scene.")
        self.compression_box = QComboBox()
        self.compression_box.addItems(COMPRESSION)
        self.compression_box.setToolTip("Higher compression makes a smaller file but drops how colour shifts with the viewing direction, "
                                        "so shine and reflections fade. None keeps everything.")
        self.upright_box.setChecked(options.upright)
        self.ground_box.setChecked(options.ground)
        self.compression_box.setCurrentIndex(3 - options.sh_degree)
        position = QGroupBox("Position")
        QVBoxLayout(position).addWidget(self.upright_box)
        position.layout().addWidget(self.ground_box)
        compression = QGroupBox("Compression")
        level = QHBoxLayout(compression)
        level.addWidget(QLabel("Level"))
        level.addWidget(self.compression_box, 1)
        cancel, ok = QPushButton("Cancel"), QPushButton("OK")
        ok.setDefault(True)
        cancel.clicked.connect(self.reject)
        ok.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        column = QVBoxLayout(self)
        column.addWidget(position)
        column.addWidget(compression)
        column.addLayout(buttons)

    def options(self):
        return ExportOptions(self.upright_box.isChecked(), self.ground_box.isChecked(), 3 - self.compression_box.currentIndex())


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
