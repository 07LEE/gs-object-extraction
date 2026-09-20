#!/usr/bin/env python3
"""Extract the object from every 360-USID scene with a trained model and build a contact sheet.

    python scripts/extract_360usid.py --data-root DATA --output outputs/360usid

``--data-root`` holds ``360-USID/<scene>`` (dataset) and ``gs/<scene>`` (Graphdeco models).
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gs_object_extraction.cli import extract_scene


def contact_sheet(paths, out, width=1200):
    from PIL import Image
    tiles = [Image.open(p) for p in paths]
    tiles = [t.resize((width, round(t.height * width / t.width))) for t in tiles]
    sheet = Image.new("RGB", (width, sum(t.height for t in tiles) + 8 * len(tiles)), (128, 128, 128))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + 8
    sheet.save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()
    scenes = sorted(p.name for p in (args.data_root / "gs").iterdir() if (args.data_root / "360-USID" / p.name).is_dir())
    summaries = {}
    for name in scenes:
        if args.scenes and name not in args.scenes:
            continue
        s = extract_scene(args.data_root / "360-USID" / name, args.data_root / "gs" / name, args.output / name)
        summaries[name] = s
        a, b = s["scores"]["selected"], s["scores"]["cleaned"]
        print(f"{name}: n {a['gaussians']} -> {b['gaussians']} | dirt {a['dirt']:.4f} -> {b['dirt']:.4f} | "
              f"missing {a['missing']:.4f} -> {b['missing']:.4f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=1) + "\n")
    previews = [args.output / name / "preview.png" for name in summaries if (args.output / name / "preview.png").exists()]
    if previews:
        contact_sheet(previews, args.output / "sheet.png")


if __name__ == "__main__":
    main()
