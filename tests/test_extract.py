from types import SimpleNamespace
import numpy as np
from gs_object_extraction.extract import extract, prune_off_mask, score, select
from gs_object_extraction.masks import band_labels
from gs_object_extraction.renderer import Lifted

# A 1 x 6 image; the object mask covers pixels 2-4.
MASK = np.array([[0, 0, 1, 1, 1, 0]], bool)
# Pixel footprints: 0, 1 are object; 2 is floor; 3 is junk under the floor, seen on
# pixel 3 but hidden on pixel 5 by floor Gaussian 4 until the floor is left out.
FOOTPRINT = [{2: 1., 3: 1.}, {4: 1.}, {0: 1., 1: 1.}, {3: .5, 5: 1.}, {5: 1.}]
HIDDEN_BY = {(3, 5): 4}


class FakeRenderer:
    n = len(FOOTPRINT)

    def weights(self, active):
        active = np.ones(self.n, bool) if active is None else active
        w = np.zeros((self.n, 6))
        for i, cover in enumerate(FOOTPRINT):
            for px, v in cover.items():
                blocker = HIDDEN_BY.get((i, px))
                if active[i] and not (blocker is not None and active[blocker]):
                    w[i, px] = v
        return w

    def lift(self, camera, labels, *, active=None):
        w, labels = self.weights(active), labels[0]
        return Lifted(w[:, labels == 1].sum(1), w[:, labels == 0].sum(1), w.sum(1))

    def render(self, camera, *, active=None):
        return SimpleNamespace(alpha=np.minimum(self.weights(active).sum(0), 1)[None])


def test_select_uses_the_inside_share_and_support():
    lifted = Lifted(np.array([.7, .6, .04, 0.]), np.array([.3, .4, 0., 0.]), np.zeros(4))
    np.testing.assert_array_equal(select(lifted), [True, False, False, False])


def test_prune_off_mask_drops_mostly_off_mask_and_keeps_unseen():
    selected = np.array([True, True, True, False, True])
    inside = np.array([5., 1., 2., 0., 0.])
    outside = np.array([1., 3., 2., 9., 0.])
    np.testing.assert_array_equal(prune_off_mask(selected, inside, outside, .5), [True, False, True, False, True])
    # the default is stricter: an even split counts as off-mask
    np.testing.assert_array_equal(prune_off_mask(selected, inside, outside), [True, False, False, False, True])


def test_extract_prunes_junk_that_only_shows_once_the_scene_is_left_out():
    renderer, views = FakeRenderer(), [(None, MASK)]
    stages = extract(renderer, views, band=0)
    np.testing.assert_array_equal(stages["selected"], [True, True, False, True, False])
    np.testing.assert_array_equal(stages["cleaned"], [True, True, False, False, False])
    before, after = score(renderer, views, stages["selected"], band=0), score(renderer, views, stages["cleaned"], band=0)
    assert before == {"gaussians": 3, "dirt": 1 / 3, "missing": 0.}
    assert after == {"gaussians": 2, "dirt": 0., "missing": 0.}


def test_band_labels_ignore_pixels_within_band_of_the_other_side_only():
    mask = np.zeros((12, 14), bool)
    mask[3:9, 4:10] = True
    np.testing.assert_array_equal(band_labels(mask, 0), mask.astype(np.int8))
    ys, xs = np.mgrid[:12, :14]
    for band in (1, 2, 3):
        labels = band_labels(mask, band)
        for y, x in zip(ys.ravel(), xs.ravel()):
            near = ((ys - y) ** 2 + (xs - x) ** 2 <= band * band) & (mask != mask[y, x])
            assert labels[y, x] == (-1 if near.any() else int(mask[y, x]))
    assert (band_labels(np.ones((5, 5), bool), 2) == 1).all()  # the image border is not a boundary


# A 1 x 12 image whose inside pixels 5 and 6 are more than 2 px from the boundary.
WIDE_MASK = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]], bool)


class ChainRenderer:
    """Footprints per Gaussian; a pixel is hidden while any of its listed blockers is drawn."""

    def __init__(self, footprint, hidden_by):
        self.footprint, self.hidden_by, self.n = footprint, hidden_by, len(footprint)

    def lift(self, camera, labels, *, active=None):
        active = np.ones(self.n, bool) if active is None else active
        w = np.zeros((self.n, labels.shape[1]))
        for i, cover in enumerate(self.footprint):
            for px, v in cover.items():
                if active[i] and not any(active[b] for b in self.hidden_by.get((i, px), ())):
                    w[i, px] = v
        labels = labels[0]
        return Lifted(w[:, labels == 1].sum(1), w[:, labels == 0].sum(1), w.sum(1))


def test_default_band_keeps_edge_gaussians_whose_spill_is_near_the_mask():
    # 0 object core, 1 object edge that spills 1-2 px past the mask behind floor 2, 3 background across the edge.
    renderer = ChainRenderer([{5: 1., 6: 1.}, {8: 1., 9: 1., 10: 1.}, {9: 1., 10: 1., 11: 1.}, {1: 1., 2: 1., 3: 1.}],
                             {(1, 9): (2,), (1, 10): (2,)})
    views = [(None, WIDE_MASK)]
    default = extract(renderer, views)
    np.testing.assert_array_equal(default["selected"], [True, True, False, False])
    np.testing.assert_array_equal(default["cleaned"], [True, True, False, False])
    np.testing.assert_array_equal(extract(renderer, views, band=0)["cleaned"], [True, False, False, False])


def test_pruning_repeats_because_dropping_junk_can_expose_more_junk():
    # Junk 1 and 2 sit behind floor 3 at pixel 11, and junk 2 also behind junk 1.
    renderer = ChainRenderer([{5: 1., 6: 1.}, {6: .5, 11: 1.}, {5: .5, 11: 1.}, {11: 1.}],
                             {(1, 11): (3,), (2, 11): (3, 1)})
    views = [(None, WIDE_MASK)]
    np.testing.assert_array_equal(extract(renderer, views, band=0)["selected"], [True, True, True, False])
    np.testing.assert_array_equal(extract(renderer, views, band=0, rounds=1)["cleaned"], [True, False, True, False])
    np.testing.assert_array_equal(extract(renderer, views, band=0)["cleaned"], [True, False, False, False])


def test_off_threshold_decides_how_much_spill_a_gaussian_may_keep():
    # Gaussian 1 draws 1.2 inside the mask and 1.0 outside it, hidden behind background 2 until it is left out.
    renderer = ChainRenderer([{5: 1., 6: 1.}, {6: 1.2, 0: 1.}, {0: 1.}], {(1, 0): (2,)})
    views = [(None, WIDE_MASK)]
    np.testing.assert_array_equal(extract(renderer, views)["selected"], [True, True, False])
    np.testing.assert_array_equal(extract(renderer, views)["cleaned"], [True, False, False])
    np.testing.assert_array_equal(extract(renderer, views, off_threshold=.5)["cleaned"], [True, True, False])
