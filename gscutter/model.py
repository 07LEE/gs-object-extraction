"""Trained Graphdeco 3DGS model folders: the point cloud and which photos training used."""

import json
from pathlib import Path
import re


def point_cloud(model_dir, iteration=None):
    """``point_cloud/iteration_<N>/point_cloud.ply``, the latest N unless given."""
    root = Path(model_dir) / "point_cloud"
    if iteration is None:
        done = sorted(int(p.name[len("iteration_"):]) for p in root.glob("iteration_*") if p.name[len("iteration_"):].isdigit())
        if not done:
            raise FileNotFoundError(f"no point_cloud/iteration_* in {model_dir}")
        iteration = done[-1]
    path = root / f"iteration_{iteration}" / "point_cloud.ply"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def training_roles(model_dir, *, llffhold=8):
    """``{photo stem: "train" | "test"}`` for a Graphdeco model folder.

    With ``--eval`` (``eval=True`` in cfg_args) Graphdeco holds out every
    ``llffhold``-th photo in sorted name order and writes those cameras first in
    cameras.json. Anything that does not match is refused rather than guessed.
    """
    records = json.loads((Path(model_dir) / "cameras.json").read_text())
    names = [Path(r["img_name"]).stem for r in records]
    if len(set(names)) != len(names):
        raise ValueError("cameras.json has duplicate image names")
    flag = re.search(r"\beval=(True|False)\b", (Path(model_dir) / "cfg_args").read_text())
    if flag is None:
        raise ValueError("cfg_args has no eval flag")
    if flag.group(1) == "False":
        return {name: "train" for name in names}
    held = set(sorted(names)[::llffhold])
    if set(names[:len(held)]) != held:
        raise ValueError("cameras.json does not list the held-out cameras first; not a Graphdeco --eval model")
    return {name: "test" if name in held else "train" for name in names}
