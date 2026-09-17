"""Click-to-mask on rendered views with SAM2.

The model loads on first use. The image embedding is computed once per view
(identified by a key) and reused while points are added to that view.
"""

import contextlib
from pathlib import Path
import re
import threading
import numpy as np

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "sam2.1_hiera_base_plus.pt"
_SIZES = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}


def config_for(checkpoint):
    """SAM2 model config matching a released checkpoint name, e.g. sam2.1_hiera_base_plus.pt."""
    match = re.fullmatch(r"(sam2(?:\.1)?)_hiera_(tiny|small|base_plus|large)", Path(checkpoint).stem)
    if match is None:
        raise ValueError(f"cannot tell the SAM2 model from {Path(checkpoint).name}; use a released checkpoint "
                         "name such as sam2.1_hiera_base_plus.pt (tiny, small, base_plus or large)")
    family, size = match.groups()
    return f"configs/{family}/{family}_hiera_{_SIZES[size]}.yaml"


class Segmenter:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, predictor=None):
        self.checkpoint = Path(checkpoint)
        self._predictor = predictor
        self._key = None
        self._lock = threading.Lock()  # a click during a background load waits instead of building a second model

    @property
    def loaded(self):
        return self._predictor is not None

    def load(self):
        """Build the model, from this thread or another; the first caller does the work."""
        with self._lock:
            if self._predictor is None:
                if not self.checkpoint.exists():
                    raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint} (run scripts/setup_env.sh)")
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                self._predictor = SAM2ImagePredictor(build_sam2(config_for(self.checkpoint), str(self.checkpoint), device="cuda"))
            return self._predictor

    @staticmethod
    def _inference():
        try:
            import torch
        except ImportError:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if torch.cuda.is_available():
            stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
        return stack

    def predict(self, image, key, points, labels):
        """Mask (bool H x W) and score for (x, y) pixel points labelled 1 (object) or 0 (background).

        ``image`` is the uint8 RGB view; ``key`` names it, so the embedding is
        recomputed only for a new view. A single point asks SAM2 for several
        candidates and keeps the best, as SAM2 recommends for ambiguous prompts.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if len(points) == 0 or len(points) != len(labels) or not np.isin(labels, (0, 1)).all():
            raise ValueError("need one 0/1 label per (x, y) point")
        predictor = self.load()
        single = len(points) == 1
        with self._inference():
            if key != self._key:
                predictor.set_image(np.asarray(image))
                self._key = key
            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, multimask_output=single)
        best = int(np.argmax(scores)) if single else 0
        return np.asarray(masks[best]) > 0, float(scores[best])

    def forget_view(self):
        """Drop the cached view so the next prediction recomputes the embedding."""
        self._key = None
