import os
from threading import Event

import numpy as np
import pytest

from gs_object_extraction.extract import extract
from gs_object_extraction.renderer import Lifted


class FakeRenderer:
    """A background fragment becomes visible after the scene is removed."""

    n = 3

    def __init__(self):
        self.calls = []

    def lift(self, camera, labels, *, active=None):
        self.calls.append((camera, labels.copy(), active))
        if active is None:
            return Lifted(np.array([1., .8, 0.]), np.array([0., .2, 1.]), np.ones(3))
        return Lifted(np.array([1., .1, 0.]), np.array([0., .9, 0.]), np.ones(3))


def test_progress_tracks_all_views_and_generator_is_reused_for_cleanup():
    renderer, updates = FakeRenderer(), []
    mask = np.array([[True, False]])
    views = ((camera, mask) for camera in ("front", "back"))
    stages = extract(renderer, views, band=0, progress=lambda *args: updates.append(args))
    assert updates == [
        ("Selecting object", 1, 6), ("Selecting object", 2, 6),
        ("Cleaning 1/2", 3, 6), ("Cleaning 1/2", 4, 6),
        ("Cleaning 2/2", 5, 6), ("Cleaning 2/2", 6, 6),
    ]
    assert [camera for camera, _, _ in renderer.calls] == ["front", "back"] * 3
    np.testing.assert_array_equal(stages["selected"], [True, True, False])
    np.testing.assert_array_equal(stages["cleaned"], [True, False, False])


def test_progress_exception_stops_before_the_next_lift():
    renderer = FakeRenderer()

    def stop(stage, done, total):
        assert len(renderer.calls) == done == 1
        raise InterruptedError("stop now")

    with pytest.raises(InterruptedError, match="stop now"):
        extract(renderer, [(None, np.ones((1, 2), bool))] * 2, progress=stop)
    assert len(renderer.calls) == 1


def test_no_cleanup_reports_only_selection():
    updates = []
    stages = extract(FakeRenderer(), [(None, np.ones((1, 2), bool))], rounds=0,
                     progress=lambda *args: updates.append(args))
    assert updates == [("Selecting object", 1, 1)]
    np.testing.assert_array_equal(stages["cleaned"], stages["selected"])


@pytest.mark.parametrize("views, rounds, message", [([], 2, "masked view"),
                                                  ([(None, None)], -1, "nonnegative")])
def test_invalid_extraction_input_fails_before_lifting(views, rounds, message):
    renderer = FakeRenderer()
    with pytest.raises(ValueError, match=message):
        extract(renderer, views, rounds=rounds)
    assert not renderer.calls


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def masked_views():
    from gs_object_extraction.app.orbit import Orbit
    from gs_object_extraction.app.views import MaskedView
    camera = Orbit(np.zeros(3), 2.).camera(4, 3)
    return [MaskedView(camera, np.ones((3, 4), bool), (), ())]


def record_signals(job):
    from PySide6.QtCore import Qt
    events = {"progress": [], "succeeded": [], "failed": [], "cancelled": []}
    for name, values in events.items():
        getattr(job, name).connect(lambda *args, values=values: values.append(args), Qt.DirectConnection)
    return events


def test_worker_emits_progress_and_cleaned_result(app):
    from gs_object_extraction.app.extraction import ExtractionJob
    renderer = FakeRenderer()
    job = ExtractionJob(renderer, masked_views())
    events = record_signals(job)
    job.start()
    assert job.wait(5000)
    assert events["progress"] == [(1, 3, "Selecting object"), (2, 3, "Cleaning 1/2"),
                                  (3, 3, "Cleaning 2/2")]
    assert len(events["succeeded"]) == 1
    np.testing.assert_array_equal(events["succeeded"][0][0]["cleaned"], [True, False, False])
    assert not events["failed"] and not events["cancelled"]


def test_worker_reports_renderer_error(app):
    from gs_object_extraction.app.extraction import ExtractionJob

    class BrokenRenderer(FakeRenderer):
        def lift(self, *args, **kwargs):
            raise RuntimeError("lifting failed")

    job = ExtractionJob(BrokenRenderer(), masked_views())
    events = record_signals(job)
    job.start()
    assert job.wait(5000)
    assert events["failed"] == [("lifting failed",)]
    assert not events["progress"] and not events["succeeded"] and not events["cancelled"]


def test_worker_cancels_after_current_lift_without_publishing_result(app):
    from gs_object_extraction.app.extraction import ExtractionJob
    entered, release = Event(), Event()

    class BlockingRenderer(FakeRenderer):
        def lift(self, *args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release the renderer")
            return super().lift(*args, **kwargs)

    renderer = BlockingRenderer()
    job = ExtractionJob(renderer, masked_views())
    events = record_signals(job)
    job.start()
    try:
        assert entered.wait(5)
        job.requestInterruption()
    finally:
        release.set()
        assert job.wait(5000)
    assert len(renderer.calls) == 1
    assert events["cancelled"] == [()]
    assert not events["progress"] and not events["succeeded"] and not events["failed"]


def test_worker_honours_cancellation_before_work(app, monkeypatch):
    from gs_object_extraction.app.extraction import ExtractionJob
    renderer = FakeRenderer()
    job = ExtractionJob(renderer, masked_views())
    monkeypatch.setattr(job, "isInterruptionRequested", lambda: True)
    events = record_signals(job)
    job.start()
    assert job.wait(5000)
    assert events["cancelled"] == [()] and not renderer.calls
    assert not events["succeeded"] and not events["failed"]


def test_worker_checks_cancellation_after_last_progress(app):
    from PySide6.QtCore import Qt
    from gs_object_extraction.app.extraction import ExtractionJob
    job = ExtractionJob(FakeRenderer(), masked_views())
    events = record_signals(job)
    job.progress.connect(lambda done, total, stage: job.requestInterruption() if done == total else None,
                         Qt.DirectConnection)
    job.start()
    assert job.wait(5000)
    assert len(events["progress"]) == 3 and events["cancelled"] == [()]
    assert not events["succeeded"] and not events["failed"]
