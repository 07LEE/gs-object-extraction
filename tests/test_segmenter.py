import os
import numpy as np
import pytest
from gs_object_extraction.app.segmenter import DEFAULT_CHECKPOINT, Segmenter


class FakePredictor:
    def __init__(self):
        self.images, self.calls = [], []

    def set_image(self, image):
        self.images.append(image.shape)

    def predict(self, point_coords, point_labels, multimask_output):
        self.calls.append((point_coords.copy(), point_labels.copy(), multimask_output))
        masks = np.zeros((3 if multimask_output else 1, 4, 5), bool)
        masks[:, 1, 2] = True
        if multimask_output:
            masks[1, 0, 0] = True  # only the best-scoring candidate, the middle one, marks this pixel
        scores = np.array([.2, .9, .5]) if multimask_output else np.array([.7])
        return masks, scores, None


def test_single_point_keeps_the_best_candidate_and_later_points_refine():
    fake = FakePredictor()
    seg = Segmenter(predictor=fake)
    image = np.zeros((4, 5, 3), np.uint8)
    mask, score = seg.predict(image, key=1, points=[(2, 1)], labels=[1])
    assert score == .9 and mask[0, 0] and mask.dtype == bool
    mask, score = seg.predict(image, key=1, points=[(2, 1), (4, 3)], labels=[1, 0])
    assert score == .7 and not mask[0, 0]
    assert [c[2] for c in fake.calls] == [True, False]
    np.testing.assert_array_equal(fake.calls[1][1], [1, 0])


def test_embedding_is_computed_once_per_view():
    fake = FakePredictor()
    seg = Segmenter(predictor=fake)
    image = np.zeros((4, 5, 3), np.uint8)
    for key in (1, 1, 1, 2, 2):
        seg.predict(image, key=key, points=[(0, 0)], labels=[1])
    assert len(fake.images) == 2
    seg.forget_view()
    seg.predict(image, key=2, points=[(0, 0)], labels=[1])
    assert len(fake.images) == 3


@pytest.mark.parametrize("points, labels", [([], []), ([(1, 1)], [1, 0]), ([(1, 1)], [2])])
def test_bad_prompts_are_rejected(points, labels):
    with pytest.raises(ValueError):
        Segmenter(predictor=FakePredictor()).predict(np.zeros((4, 5, 3), np.uint8), 1, points, labels)


def test_missing_checkpoint_says_how_to_get_it(tmp_path):
    seg = Segmenter(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="setup_env.sh"):
        seg.predict(np.zeros((4, 5, 3), np.uint8), 1, [(1, 1)], [1])
    assert not seg.loaded


def test_racing_callers_build_one_model(tmp_path, monkeypatch):
    """A click while the background load runs waits for it instead of building a second model."""
    import sys
    import threading
    import time
    import types
    checkpoint = tmp_path / "sam2.1_hiera_tiny.pt"
    checkpoint.write_bytes(b"")
    built = []

    def build_sam2(config, path, device):
        time.sleep(.05)  # long enough for the other threads to arrive
        built.append(config)
        return object()

    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", types.SimpleNamespace(build_sam2=build_sam2))
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor",
                        types.SimpleNamespace(SAM2ImagePredictor=lambda model: model))
    seg = Segmenter(checkpoint)
    results = []
    threads = [threading.Thread(target=lambda: results.append(seg.load())) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(built) == 1 and len(results) == 4 and all(r is results[0] for r in results)


@pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1" or not DEFAULT_CHECKPOINT.exists(),
                    reason="needs the GPU environment and the SAM2 checkpoint")
def test_sam2_segments_a_rendered_blob_from_one_click():
    from gs_object_extraction.app.orbit import Orbit
    from gs_object_extraction.renderer import GraphdecoRenderer
    from gs_object_extraction.scene import GaussianScene
    rng = np.random.default_rng(0)
    n = 600
    blob = GaussianScene.from_colors(rng.normal(0, .15, (n, 3)), np.full((n, 3), .04),
                                     np.tile([.9, .5, .1], (n, 1)), np.full(n, .95))
    renderer = GraphdecoRenderer(blob)
    camera = Orbit(np.zeros(3), 2.).camera(320, 240)
    image = renderer.render_image(camera, background=(.1, .1, .12))
    _, alpha = renderer.depth_image(camera)
    mask, score = Segmenter().predict(image, key=1, points=[(160, 120)], labels=[1])
    solid = alpha > .5
    assert score > .5 and (mask & solid).sum() / (mask | solid).sum() > .8


@pytest.mark.parametrize("name, config", [
    ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    ("sam2_hiera_large.pt", "configs/sam2/sam2_hiera_l.yaml"),
    ("/weights/sam2_hiera_small.pt", "configs/sam2/sam2_hiera_s.yaml"),
])
def test_config_follows_the_checkpoint_name(name, config):
    from gs_object_extraction.app.segmenter import config_for
    assert config_for(name) == config


def test_unknown_checkpoint_name_is_refused():
    from gs_object_extraction.app.segmenter import config_for
    with pytest.raises(ValueError, match="base_plus"):
        config_for("my_weights.pt")
