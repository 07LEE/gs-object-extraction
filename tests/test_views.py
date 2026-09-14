import numpy as np
import pytest
from gs_object_extraction.app.orbit import Orbit
from gs_object_extraction.app.views import MaskedView


def test_masked_view_checks_the_mask_matches_its_camera():
    camera = Orbit(np.zeros(3), 2.).camera(40, 30)
    mask = np.zeros((30, 40), bool)
    mask[5, 5] = True
    assert MaskedView(camera, mask, ((5, 5),), (1,)).mask.sum() == 1
    with pytest.raises(ValueError, match="size"):
        MaskedView(camera, np.ones((40, 30), bool), (), ())
    with pytest.raises(ValueError, match="empty"):
        MaskedView(camera, np.zeros((30, 40), bool), (), ())
