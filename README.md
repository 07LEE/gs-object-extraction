# 3D Gaussian Splatting Object Extraction

Extract an object from a trained 3D Gaussian Splatting scene and export it as a standalone PLY. SAM2 segments rendered views to identify the object's Gaussians. No source photos are needed.

![A scene and the object extracted from it](docs/images/scene-and-object.jpg)

## How it works

1. Open a scene PLY and position the camera.
2. Press **S** to enter select mode, then click the object. Review the SAM2 mask and press **Enter** to add the view.
3. Repeat from several angles, then press **Ctrl+E** to extract the object.
4. Review the preview and press **Ctrl+Shift+S** to export a PLY.

## Installation

Requires a CUDA GPU and a Python environment with PyTorch, torchvision and `diff_gaussian_rasterization`, such as the one used to train Graphdeco 3DGS.

Run from the repository root. The setup script creates `.venv-gpu` using that environment's packages, installs this tool and its dependencies, and downloads the SAM2 checkpoint.

```bash
GS_PYTHON=/path/to/gs_train/bin/python scripts/setup_env.sh
source .venv-gpu/bin/activate
```

## Viewer

Open a `point_cloud.ply` produced by Graphdeco 3DGS:

```bash
gs-object-extraction-gui path/to/point_cloud.ply
```

![The viewer with one click on the object](docs/images/viewer.jpg)

| Input | Action |
| --- | --- |
| Left drag, right drag, wheel | Orbit, pan, zoom |
| Double-click (outside select mode) | Centre rotation on the point under the cursor |
| S | Toggle select mode |
| Left click, right click (select mode) | Add an object point or background point |
| Backspace, Esc, Enter | Undo a point, clear the points, add the view |
| Ctrl+E, Ctrl+Shift+S | Extract the object, export it as a PLY |

- Mark views around the object; about 16 is a useful starting point. Keep the camera low to reduce ground included in the masks.
- **Mark around the object (experimental)** adds views automatically from your existing selections. Review the results and rerun if parts of the object are missing.
- **Edge trim** shrinks Gaussians that extend beyond the masks to reduce halos. Set it to `1.00` to preserve their size.

Surfaces missing from the source scene cannot be recovered. Thin structures such as leaves may lose detail at the edges.

## License

[Apache 2.0](LICENSE)
