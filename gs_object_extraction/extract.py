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
OFF_MASK = .35  # drop a Gaussian once this much of its object-only contribution lands off-mask
EDGE, TRIM = .05, .7  # a Gaussian reaching this far past the masks is shrunk by this much


def lift_masks(renderer, views, *, band=0, active=None, on_view=None):
    """Contributions summed over views, calling ``on_view()`` after each lift."""
    lifted = Lifted.zeros(renderer.n)
    for camera, mask in views:
        lifted += renderer.lift(camera, band_labels(mask, band), active=active)
        if on_view is not None:
            on_view()
    return lifted


def select(lifted, threshold=THRESHOLD, min_support=MIN_SUPPORT):
    """Gaussians whose labelled contribution is at least ``threshold`` inside the masks."""
    support = lifted.inside + lifted.outside
    share = np.divide(lifted.inside, support, out=np.zeros_like(support), where=support > 0)
    return (support >= min_support) & (share >= threshold)


def prune_off_mask(selected, inside, outside, threshold=OFF_MASK):
    """Keep selected Gaussians unless most of their object-only contribution is off-mask."""
    return selected & ~(off_share(inside, outside) > threshold)


def off_share(inside, outside):
    """Fraction of a Gaussian's object-only contribution that lands off-mask."""
    total = inside + outside
    return np.divide(outside, total, out=np.zeros_like(total, dtype=float), where=total > 0)


def trim_scales(off, *, threshold=EDGE, factor=TRIM):
    """Scale factor per Gaussian: the ones reaching past the masks are shrunk, the rest kept.

    The haze an object trails into empty space, the skirt under where it meets the
    floor above all, is the tails of the Gaussians on its outline rather than any
    Gaussian of its own. Pulling those tails in cleans the edge; the object's own
    surface is left alone, which shrinking everything would thin.
    """
    if not 0 < factor <= 1:
        raise ValueError("the trim factor must be above zero and at most one")
    return np.where(np.asarray(off) > threshold, factor, 1.)


def extract(renderer, views, *, threshold=THRESHOLD, min_support=MIN_SUPPORT, rounds=ROUNDS, band=BAND, off_threshold=OFF_MASK,
            progress=None):
    """Return ``selected`` and ``cleaned`` boolean arrays over the scene, and the off-mask shares.

    ``progress(stage, done, total)`` runs after every lifted view; raising from
    it stops extraction. Views are retained so iterators work for every round.
    ``off`` comes from the last cleaning round and is all zeros without one.
    """
    views = tuple(views)
    if not views:
        raise ValueError("extraction requires at least one masked view")
    if rounds < 0:
        raise ValueError("cleaning rounds must be nonnegative")
    done, total = 0, len(views) * (rounds + 1)
    stage = "Selecting object"

    def report_view():
        nonlocal done
        done += 1
        if progress is not None:
            progress(stage, done, total)

    selected = select(lift_masks(renderer, views, on_view=report_view), threshold, min_support)
    stages = {"selected": selected, "off": np.zeros(renderer.n)}
    for round_index in range(rounds):
        stage = f"Cleaning {round_index + 1}/{rounds}"
        own = lift_masks(renderer, views, band=band, active=selected, on_view=report_view)
        stages["off"] = off_share(own.inside, own.outside)
        selected = prune_off_mask(selected, own.inside, own.outside, off_threshold)
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
