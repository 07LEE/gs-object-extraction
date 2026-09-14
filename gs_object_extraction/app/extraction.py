"""Background mask lifting and cleanup with cooperative cancellation."""

from PySide6.QtCore import QThread, Signal
from ..extract import extract


class _ExtractionCancelled(Exception):
    pass


class ExtractionJob(QThread):
    progress = Signal(int, int, str)  # completed views, total views, stage
    succeeded = Signal(object)  # {"selected": boolean array, "cleaned": boolean array}
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, renderer, views, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self.views = tuple((view.camera, view.mask) for view in views)

    def _check_cancelled(self):
        if self.isInterruptionRequested():
            raise _ExtractionCancelled()

    def _report_progress(self, stage, done, total):
        self._check_cancelled()
        self.progress.emit(done, total, stage)

    def run(self):
        try:
            self._check_cancelled()
            stages = extract(self.renderer, self.views, progress=self._report_progress)
            self._check_cancelled()
        except _ExtractionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(stages)
