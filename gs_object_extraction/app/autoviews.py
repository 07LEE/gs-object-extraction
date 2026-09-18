"""Mark the object all the way around without a click in every view.

One marked view already gives a rough selection. Rendering that selection on its
own in a new view says where the object lands on screen, and that point is the
prompt SAM2 needs, so the tool clicks for the user. The ring sits low: views from
high up are worse, because the ground behind the object falls inside its outline.
"""

import numpy as np
from PySide6.QtCore import QThread, Signal
from ..extract import lift_masks, select
from ..masks import disk_dilate
from .orbit import Orbit
from .viewport import BACKGROUND
from .views import MaskedView

VIEWS = 16
PITCH = np.deg2rad(12.)
DISTANCE = 3.2  # times the object's radius
MIN_PIXELS = 200  # a silhouette smaller than this is not worth marking
MIN_AGREE = .25  # this much of a mask has to sit on the object for the view to be believed
MAX_FRAME = .6  # ... and must not swallow the frame
POINTS = 3  # clicks spread over the silhouette
NEAR = 1.5  # keep the selection within this many object radii when looking for the object on screen
DEEP = 8  # click this far inside the outline where the silhouette is wide enough
MAX_FILL = .2  # back the ring off until the object takes at most this much of the frame


def surface_points(renderer, views, step=4):
    """Points on the object's visible surface, unprojected from the masks the user marked.

    Lifting from a single view also takes in what stands behind the object, so
    the selection is no guide to where the object is. What the user marked is:
    every masked pixel shows a surface at the depth the scene renders there.
    """
    points = []
    for camera, mask in views:
        depth, alpha = renderer.depth_image(camera)
        ys, xs = np.nonzero(mask[::step, ::step] & (alpha[::step, ::step] > .5))
        if not len(ys):
            continue
        ys, xs = ys * step, xs * step
        z = depth[ys, xs]
        local = np.stack([(xs - camera.cx) / camera.fx * z, (ys - camera.cy) / camera.fy * z, z], axis=1)
        rotation, translation = camera.world_to_camera[:3, :3], camera.world_to_camera[:3, 3]
        points.append((local - translation) @ rotation)
    if not points:
        raise ValueError("The marked views show nothing solid inside the mask.")
    return np.concatenate(points)


def object_frame(points):
    """Centre and radius of a point cloud, ignoring a few strays."""
    points = np.asarray(points, dtype=float)
    centre = np.median(points, axis=0)
    radius = float(np.percentile(np.linalg.norm(points - centre, axis=1), 95))
    return centre, max(radius, 1e-6)


def ring(centre, radius, up, *, count=VIEWS, pitch=PITCH, distance=DISTANCE):
    """Evenly spaced orbits around the object at one low elevation."""
    if count < 1:
        raise ValueError("a ring needs at least one view")
    return [Orbit(centre, distance * radius, yaw=float(yaw), pitch=float(pitch), up=up)
            for yaw in np.linspace(0, 2 * np.pi, count, endpoint=False)]


def inner_part(coverage, depth=DEEP):
    """The silhouette away from its outline, as deep as it survives being eaten in."""
    for radius in (depth, depth // 2, depth // 4):
        if radius >= 1:
            eroded = coverage & ~disk_dilate(~coverage, radius)
            if eroded.sum() >= MIN_PIXELS:
                return eroded
    return coverage


def prompt_points(coverage, count=POINTS):
    """Where to click on a silhouette: its middle, then the points furthest from what is chosen.

    The points come from well inside the outline, since a click on the edge reads
    as much as the background behind it. Spreading them matters for an object SAM2
    can read as parts, a plant's leaves or a box's faces: one click returns one part.
    """
    inner = inner_part(coverage)
    ys, xs = np.nonzero(inner)
    y, x = int(np.median(ys)), int(np.median(xs))
    if not inner[y, x]:  # the middle can fall in a hole, e.g. between leaves
        nearest = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        y, x = int(ys[nearest]), int(xs[nearest])
    chosen = [(x, y)]
    far = np.full(len(xs), np.inf)
    for _ in range(min(count, len(xs)) - 1):
        far = np.minimum(far, (xs - chosen[-1][0]) ** 2 + (ys - chosen[-1][1]) ** 2)
        chosen.append((int(xs[far.argmax()]), int(ys[far.argmax()])))
    return chosen


def best_candidate(masks, coverage):
    """The candidate that agrees most with where the object landed, and how squarely it sits on it.

    The candidate is chosen by overlap both ways, so a mask of one leaf does not win
    over a mask of the plant. What comes back is how much of the chosen mask lands on
    the object: a mask of part of the object still says everything around it is not
    the object, and is worth keeping, while a mask of the sky beside it is not.
    """
    overlaps = [(mask & coverage).sum() / max((mask | coverage).sum(), 1) for mask in masks]
    best = int(np.argmax(overlaps))
    mask = masks[best]
    return mask, float((mask & coverage).sum() / max(mask.sum(), 1))


def usable(mask, sits_on):
    """Believe a mask that sits on the object, and does not swallow the frame.

    A ring passes through places the capture never covered and behind other objects.
    The renders there are a smear, and a promptable segmenter answers a click in them
    with the sky or the ground, which is what this rejects. A mask of only part of the
    object is kept: it still says everything around it is not the object, and dropping
    such views measurably cost more than it saved.
    """
    return mask is not None and mask.any() and sits_on >= MIN_AGREE and mask.mean() <= MAX_FRAME


class AutoMarkJob(QThread):
    """Select from the views marked so far, then mark a ring of views around the object."""

    progress = Signal(int, int)  # views visited, views planned
    succeeded = Signal(object, int)  # marked views, views SAM2 could not mark
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, renderer, segmenter, means, views, up, size, parent=None, *, count=VIEWS, pitch=PITCH):
        super().__init__(parent)
        self.renderer, self.segmenter = renderer, segmenter
        self.means = means
        self.views = tuple((view.camera, view.mask) for view in views)
        self.up, self.size = up, size
        self.count, self.pitch = count, pitch

    def run(self):
        try:
            marked, skipped = [], 0
            selection = select(lift_masks(self.renderer, self.views))
            if not selection.any():
                raise ValueError("The marked views do not select any Gaussian yet. Mark the object more closely first.")
            centre, radius = object_frame(surface_points(self.renderer, self.views))
            # A selection lifted from one view trails off behind the object; only what sits
            # around the object says where it lands on screen.
            near = selection & (np.linalg.norm(np.asarray(self.means) - centre, axis=1) <= NEAR * radius)
            if not near.any():
                raise ValueError("The marked views select nothing near the object.")
            width, height = self.size
            for _ in range(3):  # one view sees only the near side, so the radius can come out short
                first = ring(centre, radius, self.up, count=1, pitch=self.pitch)[0].camera(width, height)
                if (self.renderer.alpha_image(first, active=near) > .5).mean() <= MAX_FILL:
                    break
                radius *= 1.4
            for index, orbit in enumerate(ring(centre, radius, self.up, count=self.count, pitch=self.pitch)):
                if self.isInterruptionRequested():
                    self.cancelled.emit()
                    return
                camera = orbit.camera(width, height)
                silhouette = np.clip(self.renderer.alpha_image(camera, active=near), 0, 1) > .5
                if silhouette.sum() < MIN_PIXELS:
                    skipped += 1
                else:
                    points = prompt_points(silhouette)
                    masks, _ = self.segmenter.candidates(self.renderer.render_image(camera, background=BACKGROUND),
                                                         ("auto", index), points, [1] * len(points), several=True)
                    mask, sits_on = best_candidate(masks, silhouette)
                    if usable(mask, sits_on):
                        marked.append(MaskedView(camera, mask, tuple(points), (1,) * len(points)))
                    else:
                        skipped += 1
                self.progress.emit(index + 1, self.count)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(marked, skipped)
