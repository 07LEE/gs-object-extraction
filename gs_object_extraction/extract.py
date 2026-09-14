"""Clean object extraction: lift object masks to Gaussians, then prune what the selection draws off-mask on its own.

Mask lifting keeps Gaussians that are hidden in the full scene (under the
floor, behind other background) because nothing there says they are not
object. Rendered without the rest of the scene they land outside the object
masks; ``prune_off_mask`` drops those whose object-only contribution is mostly
off-mask. It repeats, because dropping one can expose another.
"""

import numpy as np
from .masks import band_labels
from .renderer import Lifted

THRESHOLD, MIN_SUPPORT, ROUNDS, BAND = .65, .05, 2, 2


def lift_masks(renderer, views, *, band=0, active=None):
    """Contributions summed over ``(camera, mask)`` views."""
    lifted = Lifted.zeros(renderer.n)
    for camera, mask in views:
        lifted += renderer.lift(camera, band_labels(mask, band), active=active)
    return lifted


def select(lifted, threshold=THRESHOLD, min_support=MIN_SUPPORT):
    """Gaussians whose labelled contribution is at least ``threshold`` inside the masks."""
    support = lifted.inside + lifted.outside
    share = np.divide(lifted.inside, support, out=np.zeros_like(support), where=support > 0)
    return (support >= min_support) & (share >= threshold)


def prune_off_mask(selected, inside, outside, threshold=.5):
    """Keep selected Gaussians unless most of their object-only contribution is off-mask."""
    total = inside + outside
    off = np.divide(outside, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    return selected & ~(off > threshold)


def extract(renderer, views, *, threshold=THRESHOLD, min_support=MIN_SUPPORT, rounds=ROUNDS, band=BAND):
    """``{"selected": mask-lifted selection, "cleaned": after pruning}`` as boolean arrays over the scene."""
    selected = select(lift_masks(renderer, views), threshold, min_support)
    stages = {"selected": selected}
    for _ in range(rounds):
        own = lift_masks(renderer, views, band=band, active=selected)
        selected = prune_off_mask(selected, own.inside, own.outside)
    stages["cleaned"] = selected
    return stages


def score(renderer, views, selected, *, band=BAND):
    """Object-only alpha against masks, per mask pixel, band ignored.

    ``dirt`` is alpha outside the mask and ``missing`` the alpha deficit inside.
    """
    dirt, missing = [], []
    for camera, mask in views:
        labels = band_labels(mask, band)
        alpha = np.clip(renderer.render(camera, active=selected).alpha, 0, 1)
        area = max(int(np.count_nonzero(mask)), 1)
        dirt.append(float(alpha[labels == 0].sum() / area))
        missing.append(float((1 - alpha)[labels == 1].sum() / area))
    return {"gaussians": int(selected.sum()), "dirt": float(np.mean(dirt)) if dirt else None,
            "missing": float(np.mean(missing)) if missing else None}
