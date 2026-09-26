#!/usr/bin/env python3
"""Render the README's turntable: the scene, the extracted object on white, and on black, turning together.

    python scripts/make_hero.py SCENE.ply OBJECT.ply docs/images/scene-and-object.webp

``OBJECT.ply`` is an export of the viewer (or ``extract_360usid.py``) taken from ``SCENE.ply``, so
it sits in the scene's own coordinates and the same camera frames both.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gs_object_extraction.app.orbit import Orbit, estimate_up
from gs_object_extraction.ply import load_ply
from gs_object_extraction.renderer import GsplatRenderer

PANEL = (325, 386)  # width, height of each panel
FRAMES, MS = 48, 50
PITCH = 12  # degrees above the horizon
FILL = .8  # how much of a panel's height the object takes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path)
    parser.add_argument("object", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()
    scene, thing = load_ply(args.scene), load_ply(args.object)
    scene_renderer, object_renderer = GsplatRenderer(scene), GsplatRenderer(thing)
    up = estimate_up(scene.means)
    centre = np.median(thing.means, axis=0)
    radius = float(np.percentile(np.linalg.norm(thing.means - centre, axis=1), 99))
    fov = 50.
    distance = radius / FILL / np.tan(np.deg2rad(fov / 2))  # the object's radius spans FILL of half the height
    frames = []
    for i in range(FRAMES):
        camera = Orbit(centre, distance, yaw=2 * np.pi * i / FRAMES, pitch=np.deg2rad(PITCH), up=up, fov_y=fov).camera(*PANEL)
        panels = [scene_renderer.render_image(camera),
                  object_renderer.render_image(camera, background=(1, 1, 1)),
                  object_renderer.render_image(camera, background=(0, 0, 0))]
        frames.append(Image.fromarray(np.concatenate(panels, axis=1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=MS, loop=0, quality=args.quality, method=6)
    print(f"{args.output}: {frames[0].width}x{frames[0].height}, {FRAMES} frames, {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
