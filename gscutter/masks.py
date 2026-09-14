"""Mask labels for lifting: inside, outside, and an ignored band along the boundary."""

import numpy as np


def disk_dilate(mask, radius):
    """Binary dilation by a Euclidean disk; the image border is not part of either side."""
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dy or dx) and dy * dy + dx * dx <= radius * radius:
                src = mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
                out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] |= src
    return out


def band_labels(mask, band):
    """1 inside, 0 outside, -1 within ``band`` px of a pixel of the other side."""
    mask = np.asarray(mask, dtype=bool)
    labels = mask.astype(np.int8)
    if band > 0:
        labels[(disk_dilate(mask, band) & ~mask) | (disk_dilate(~mask, band) & mask)] = -1
    return labels
