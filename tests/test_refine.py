import os

import numpy as np
import pytest

from gs_object_extraction.camera import Camera
from gs_object_extraction.masks import band_labels
from gs_object_extraction.refine import refine
from gs_object_extraction.scene import GaussianScene


def test_refining_without_a_marked_view_is_refused():
    with pytest.raises(ValueError, match="at least one marked view"):
        refine(object(), [])


@pytest.mark.skipif(os.environ.get("GS_OBJECT_EXTRACTION_TEST_CUDA") != "1",
                    reason="set GS_OBJECT_EXTRACTION_TEST_CUDA=1 in the GPU environment")
def test_a_see_through_object_is_refit_to_fill_its_mask_and_the_renderer_is_left_as_it_was():
    from gs_object_extraction.renderer import GsplatRenderer
    grid = np.array([[x, y, 2.] for x in np.linspace(-.5, .5, 6) for y in np.linspace(-.5, .5, 6)])
    n = len(grid)
    scene = GaussianScene.from_colors(grid, np.full((n, 3), .18), np.tile([.85, .35, .15], (n, 1)), np.full(n, .25))
    renderer = GsplatRenderer(scene)
    camera = Camera.look_at((0, 0, 0), (0, 0, 2), width=64, height=64, fov_y_degrees=50)
    shown = renderer.render(camera).alpha
    labels = band_labels(shown > .1, 2)
    target = np.zeros((64, 64, 3), np.float32)
    target[:] = [.85, .35, .15]  # the colour the object had in the scene, solid
    before = renderer.opacities.clone(), renderer.sh.clone(), renderer.scales.clone()
    result = refine(renderer, [(camera, target, labels)], steps=300)
    assert result["after"] > result["before"] + .2 and result["after"] > .7
    assert result["opacities"].shape == (n,) and result["sh"].shape == scene.sh.shape and result["scale_factors"].shape == (n, 3)
    assert result["scale_factors"].min() >= .5 - 1e-6 and result["scale_factors"].max() <= 2 + 1e-6
    assert result["opacities"].mean() > .6  # what was see-through is now solid
    assert renderer.opacities.equal(before[0]) and renderer.sh.equal(before[1]) and renderer.scales.equal(before[2])
    same = refine(renderer, [(camera, target, labels)], steps=20, fit_scales=False)
    np.testing.assert_array_equal(same["scale_factors"], 1)  # sizes are left alone when asked
