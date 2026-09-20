"""360-USID evaluation: extract one object from a trained scene and score the result.

Repo-internal, not an installed command. The viewer (``gs-object-extraction-gui``) is
how the project is used; this module is what ``scripts/extract_360usid.py`` drives to
reproduce the dataset numbers.

The scene folder uses the 360-USID layout (COLMAP ``sparse/0``, ``images/``,
``object_masks/``); the model folder is a Graphdeco training output. Masks of
the photos the model trained on drive the extraction; masks of the photos it
held out only score it. The photos themselves are never read.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from . import extract as ex
from .model import point_cloud, training_roles
from .ply import load_ply, save_ply
from .usid import UsidScene, load_mask


def usid_views(scene_dir, model_dir):
    """``(camera, mask)`` pairs for with-object photos the model trained on, and for those it held out."""
    roles = training_roles(model_dir)
    views = UsidScene(scene_dir).split("train")
    unknown = sorted(v.name for v in views if v.name not in roles)
    if unknown:
        raise ValueError(f"photos missing from the model's cameras.json: {unknown[:3]}")
    pick = lambda role: [(v.camera, load_mask(v.object_mask)) for v in views if roles[v.name] == role]
    return pick("train"), pick("test")


def crop_box(mask, pad=.25):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    py, px = int(pad * (ys.max() - ys.min())) + 8, int(pad * (xs.max() - xs.min())) + 8
    return slice(max(0, ys.min() - py), min(h, ys.max() + py)), slice(max(0, xs.min() - px), min(w, xs.max() + px))


def preview(renderer, camera, mask, before, after, path):
    """Selection on white | cleaned on white | cleaned on black, cropped around the mask."""
    from PIL import Image
    box = crop_box(mask)
    tiles = [np.clip(renderer.render(camera, active=s, background=bg).rgb, 0, 1)[box]
             for s, bg in ((before, (1, 1, 1)), (after, (1, 1, 1)), (after, (0, 0, 0)))]
    gap = np.full((tiles[0].shape[0], 6, 3), .5)
    row = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], axis=1)
    Image.fromarray((row * 255).round().astype(np.uint8)).save(path)


def extract_scene(scene_dir, model_dir, output, **options):
    """Write ``object.ply``, ``summary.json`` and ``preview.png`` for one scene; return the summary."""
    from .renderer import GsplatRenderer
    train, held = usid_views(scene_dir, model_dir)
    scene = load_ply(point_cloud(model_dir))
    renderer = GsplatRenderer(scene)
    stages = ex.extract(renderer, train, **options)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_ply(scene.subset(stages["cleaned"]), output / "object.ply")
    summary = {"scene_dir": str(scene_dir), "model_dir": str(model_dir), "options": options,
               "extraction_views": len(train), "scoring_views": len(held),
               "scores": {name: ex.score(renderer, held, s, band=options.get("band", ex.BAND)) for name, s in stages.items()}
               if held else None}
    if held:
        camera, mask = max(held, key=lambda view: np.count_nonzero(view[1]))
        preview(renderer, camera, mask, stages["selected"], stages["cleaned"], output / "preview.png")
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="extract_360usid", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract", help="cut a clean object out of a trained 3DGS")
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=ex.THRESHOLD, help="inside share needed to select a Gaussian")
    p.add_argument("--rounds", type=int, default=ex.ROUNDS, help="off-mask pruning rounds")
    p.add_argument("--band", type=int, default=ex.BAND, help="ignored boundary band in pixels while pruning and scoring")
    p.add_argument("--off-threshold", type=float, default=ex.OFF_MASK,
                   help="drop a Gaussian once this much of its object-only contribution lands off-mask")
    args = parser.parse_args(argv)
    summary = extract_scene(args.scene_dir, args.model_dir, args.output, threshold=args.threshold,
                            rounds=args.rounds, band=args.band, off_threshold=args.off_threshold)
    print(json.dumps(summary["scores"], indent=1))
