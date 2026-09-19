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
WARMUP_SIZE = 64  # SAM2 resizes every view to its own input size, so a small one warms the same kernels
_WARMUP_VIEW = object()  # a view key of its own, so the warmup embeds instead of reusing a cached view


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
        # One caller at a time, so a click during a background load waits for the model instead of
        # building a second one, and does not reach the predictor while the warmup is still using it.
        self._lock = threading.Lock()

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

    def warm(self):
        """Build the model, then spend one throwaway prompt on it so the first real click does not.

        The first embedding after a build pays for kernel autotuning, several times
        what the ones after it cost. Doing it on the loading thread, where the window
        is already waiting, is what the first click is spared. A warmup that fails
        costs only the time it would have saved, so it does not stop the model being
        used.
        """
        predictor = self.load()
        middle = WARMUP_SIZE // 2
        try:
            self.candidates(np.zeros((WARMUP_SIZE, WARMUP_SIZE, 3), np.uint8), _WARMUP_VIEW, [(middle, middle)], [1])
        except Exception:
            pass  # the model is built and usable; only the saving is lost
        finally:
            self.forget_view()  # the throwaway view is not one the user can click on
        return predictor

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

    def candidates(self, image, key, points, labels, *, several=None):
        """Masks SAM2 offers for the prompt, with their scores.

        ``image`` is the uint8 RGB view; ``key`` names it, so the embedding is
        recomputed only for a new view. ``several`` asks for the set of
        candidates SAM2 proposes for an ambiguous prompt, which it recommends
        for a lone point; by default that is what a lone point gets.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        labels = np.asarray(labels, dtype=int).reshape(-1)
        if len(points) == 0 or len(points) != len(labels) or not np.isin(labels, (0, 1)).all():
            raise ValueError("need one 0/1 label per (x, y) point")
        predictor = self.load()
        with self._lock, self._inference():
            if key != self._key:
                predictor.set_image(np.asarray(image))
                self._key = key
            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels,
                                                 multimask_output=len(points) == 1 if several is None else several)
        return np.asarray(masks) > 0, np.asarray(scores, dtype=float).reshape(-1)

    def predict(self, image, key, points, labels):
        """Mask (bool H x W) and score for (x, y) pixel points labelled 1 (object) or 0 (background)."""
        masks, scores = self.candidates(image, key, points, labels)
        best = int(np.argmax(scores))
        return masks[best], float(scores[best])

    def forget_view(self):
        """Drop the cached view so the next prediction recomputes the embedding."""
        self._key = None
