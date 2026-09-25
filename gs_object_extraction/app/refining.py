"""Background refit of the extracted object, with cooperative cancellation."""

from PySide6.QtCore import QThread, Signal
from ..refine import refine, targets_from


class _Cancelled(Exception):
    pass


class RefineJob(QThread):
    progress = Signal(int, int, str)  # completed steps, total steps, stage
    succeeded = Signal(object)  # {"opacities", "sh", "before", "after"}
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, scene_renderer, object_scene, views, make_renderer, parent=None):
        super().__init__(parent)
        self.scene_renderer, self.object_scene, self.make_renderer = scene_renderer, object_scene, make_renderer
        self.views = tuple((view.camera, view.mask) for view in views)

    def _step(self, done, total):
        if self.isInterruptionRequested():
            raise _Cancelled()
        self.progress.emit(done, total, "Refining")

    def run(self):
        try:
            self.progress.emit(0, 1, "Rendering the scene")
            targets = targets_from(self.scene_renderer, self.views)
            if self.isInterruptionRequested():
                raise _Cancelled()
            result = refine(self.make_renderer(self.object_scene), targets, on_step=self._step)
        except _Cancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
