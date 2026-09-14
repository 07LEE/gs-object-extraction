"""360-USID scene index: cameras and masks for evaluation, never photos for selection.

A scene registers with-object training photos (``images/``, ``object_masks/``,
``unseen_masks/``) and object-removed test photos (``test_images/``,
``test_object_masks/``) in one COLMAP model. Per the dataset README the test
object masks are rendered by Masked-GS, not hand-annotated. Photos are exposed
only as paths: the evaluator may read them, the selection process must not.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .colmap import camera_from_colmap, principal_point_offset, read_cameras_binary, read_images_binary
from .camera import Camera


@dataclass(frozen=True)
class UsidView:
    name: str
    split: str  # "train" (object present) or "test" (object removed)
    camera: Camera
    photo: Path
    object_mask: Path
    unseen_mask: Path | None


def load_mask(path, *, soft=False):
    """Boolean mask at >= 128 of 255 (JPEG masks carry compression noise), or [0, 1] floats."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("reading 360-USID masks needs Pillow (pip install -e '.[images]')") from exc
    values = np.asarray(Image.open(path).convert("L"))
    return values / 255 if soft else values >= 128


def _image_size(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.size


class UsidScene:
    def __init__(self, root, *, width=None):
        """Cameras match each photo's pixel size, or ``width`` with the photo aspect ratio."""
        self.root = Path(root)
        cameras = read_cameras_binary(self.root / "sparse/0/cameras.bin")
        images = read_images_binary(self.root / "sparse/0/images.bin")
        train = sorted(p.name for p in (self.root / "images").iterdir())
        test = sorted(p.name for p in (self.root / "test_images").iterdir())
        missing = sorted(set(train + test) - set(images))
        if missing:
            raise ValueError(f"{self.root.name}: photos without COLMAP poses: {missing[:3]}")
        self.views = []
        for split, names in (("train", train), ("test", test)):
            for name in names:
                folder = "images" if split == "train" else "test_images"
                photo = self.root / folder / name
                w, h = _image_size(photo)
                if width:
                    w, h = int(width), max(1, round(h * width / w))
                image = images[name]
                stem = Path(name).stem
                self.views.append(UsidView(
                    stem, split, camera_from_colmap(cameras[image.camera_id], image, w, h), photo,
                    self.root / ("object_masks/" + name if split == "train" else f"test_object_masks/{stem}.png"),
                    self.root / "unseen_masks" / name if split == "train" else None))
        offsets = [principal_point_offset(cameras[images[n].camera_id]) for n in train + test]
        self.max_principal_offset = float(np.abs(offsets).max())

    def split(self, name):
        return [view for view in self.views if view.split == name]

    def graphdeco_holdout(self, llffhold=8):
        """Training views Graphdeco ``--eval`` held out: every llffhold-th sorted name."""
        names = sorted(view.name for view in self.split("train"))
        return set(names[::llffhold])
