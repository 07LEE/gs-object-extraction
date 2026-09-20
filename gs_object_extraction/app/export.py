"""Background PLY writing, so the window keeps painting while a large object is saved."""

from PySide6.QtCore import QThread, Signal
from ..ply import save_ply


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
