"""Small real-CUDA check of the GUI worker through an exported object PLY."""

import os

import numpy as np
import pytest

from gs_object_extraction.camera import Camera
from gs_object_extraction.ply import load_ply, save_ply
from gs_object_extraction.scene import GaussianScene


@pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1",
                    reason="set GS_OBJECT_EXTRACTION_TEST_CUDA=1 in the GPU environment")
def test_real_worker_extracts_two_views_and_exports_the_foreground(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from gs_object_extraction.app.extraction import ExtractionJob
    from gs_object_extraction.app.views import MaskedView
    from gs_object_extraction.renderer import GraphdecoRenderer

    app = QApplication.instance() or QApplication([])
    # The centre group is the object; the two separated side groups are background.
    offsets = np.array([[-.12, -.08, .02], [.12, -.08, -.02], [0., .12, .01]])
    means = np.concatenate([offsets + centre for centre in
                            ((0., 0., 0.), (-1.25, .2, -.15), (1.25, -.2, .15))])
    n = len(means)
    base = GaussianScene.from_colors(means, np.tile([.10, .08, .06], (n, 1)),
                                      np.tile([.85, .35, .15], (n, 1)), np.linspace(.92, .99, n))
    angles = np.linspace(.1, .7, n)
    rotations = np.column_stack([np.cos(angles / 2), np.zeros((n, 2)), np.sin(angles / 2)])
    sh = np.zeros((n, 4, 3))
    sh[:, 0] = base.sh[:, 0]
    sh[:, 1:] = np.random.default_rng(13).normal(0, .03, (n, 3, 3))
    scene = GaussianScene(base.means, base.scales, rotations, base.opacities, sh,
                          ids=np.array([7001, 41, 900003, 100, 200, 300, 400, 500, 600]),
                          extras={"confidence": np.linspace(.2, .9, n, dtype=np.float32),
                                  "source_weight": np.linspace(.123456789, .987654321, n)})
    foreground = np.arange(n) < len(offsets)
    renderer = GraphdecoRenderer(scene)
    cameras = [Camera.look_at(eye, (0., 0., 0.), width=96, height=72)
               for eye in ((0., 0., 4.), (.7, .3, 4.))]
    views = []
    for camera in cameras:
        mask = renderer.render(camera, active=foreground).alpha > .02
        assert 0 < mask.sum() < mask.size / 4
        views.append(MaskedView(camera, mask, (), ()))

    job = ExtractionJob(renderer, views)
    progress, results, errors, cancelled = [], [], [], []
    job.progress.connect(lambda done, total, stage: progress.append((done, total, stage)))
    job.succeeded.connect(results.append)
    job.failed.connect(errors.append)
    job.cancelled.connect(lambda: cancelled.append(True))
    loop, timeout = QEventLoop(), QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    job.finished.connect(loop.quit)
    try:
        timeout.start(15000)
        job.start()
        loop.exec()  # Deliver the worker's queued signals on the GUI thread.
        assert job.isFinished(), "CUDA extraction exceeded 15 seconds"
    finally:
        timeout.stop()
        job.requestInterruption()
        assert job.wait(15000), "CUDA extraction did not stop after interruption"
    app.processEvents()

    assert errors == [] and cancelled == [] and len(results) == 1
    assert progress == [(1, 6, "Selecting object"), (2, 6, "Selecting object"),
                        (3, 6, "Cleaning 1/2"), (4, 6, "Cleaning 1/2"),
                        (5, 6, "Cleaning 2/2"), (6, 6, "Cleaning 2/2")]
    for stage in ("selected", "cleaned"):
        np.testing.assert_array_equal(results[0][stage], foreground)

    path = tmp_path / "foreground.ply"
    save_ply(scene.subset(results[0]["cleaned"]), path)
    exported = load_ply(path)
    assert len(exported) == len(offsets)
    np.testing.assert_array_equal(exported.ids, scene.ids[foreground])
    for name in ("means", "scales", "quaternions", "opacities", "sh"):
        np.testing.assert_allclose(getattr(exported, name), getattr(scene, name)[foreground],
                                   rtol=2e-6, atol=1e-8)
    assert exported.extras.keys() == scene.extras.keys()
    for name, values in scene.extras.items():
        np.testing.assert_array_equal(exported.extras[name], values[foreground])
