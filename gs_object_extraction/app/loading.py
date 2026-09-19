"""Background loading, so the window keeps painting while a scene or SAM2 comes up."""

from PySide6.QtCore import QThread, Signal
from ..ply import load_ply
from .orbit import Orbit, estimate_up, frame_scene


class SceneLoadJob(QThread):
    """Read a Gaussian PLY, put it on the GPU, estimate the up axis and frame the scene.

    Everything the first frame needs happens here, so the window keeps answering
    while a scene of several million Gaussians comes up.
    """

    progress = Signal(str)  # the stage now running
    succeeded = Signal(object, object, object, object)  # scene, renderer, estimated up axis, first view
    failed = Signal(str)

    def __init__(self, path, chosen_up=None, parent=None):
        super().__init__(parent)
        self.path = path
        self.chosen_up = chosen_up  # None follows the estimate, as the Auto up axis does

    def run(self):
        from ..renderer import GsplatRenderer
        try:
            self.progress.emit("Reading the PLY")
            scene = load_ply(self.path)
            if self.isInterruptionRequested():
                return
            self.progress.emit("Uploading to the GPU")
            renderer = GsplatRenderer(scene)
            if self.isInterruptionRequested():
                return
            self.progress.emit("Framing the scene")
            up = estimate_up(scene.means)
            orbit = frame_scene(renderer, scene.means, up=up if self.chosen_up is None else self.chosen_up)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(scene, renderer, up, orbit)


class SegmenterLoadJob(QThread):
    """Build the SAM2 model ahead of the first click, and warm it while we are here.

    A freshly built model spends several times as long on its first embedding as on
    the ones after it, so the warmup belongs on this thread rather than in the first
    click. See ``Segmenter.warm`` for what that came to when it was measured.
    """

    failed = Signal(str)

    def __init__(self, segmenter, parent=None):
        super().__init__(parent)
        self.segmenter = segmenter

    def run(self):
        try:
            self.segmenter.warm()
        except Exception as exc:
            self.failed.emit(str(exc))
