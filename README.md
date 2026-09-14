# 3D Gaussian Splatting Object Extraction

Tool for cutting a single object out of a trained 3D Gaussian Splatting scene and saving it as a standalone PLY. Works from the trained model and per-view object masks only; source photos are never read.

## 1. Installation

### Requirements

- CUDA GPU
- Python 3.10+ with PyTorch and diff_gaussian_rasterization (the Graphdeco 3DGS training environment)
- numpy, Pillow, PySide6, SAM2 (installed by the setup script)

### From a local clone (editable, for development)

Run from the repository root. The setup script builds .venv-gpu on top of the 3DGS training environment, installs requirements.txt and downloads the SAM2 checkpoint.

```bash
GS_PYTHON=/path/to/gs_train/bin/python scripts/setup_env.sh
source .venv-gpu/bin/activate
```

requirements.txt lists what is installed on top of the training environment (PySide6, SAM2 and their dependencies). The SAM2 checkpoint is saved under checkpoints. If PYTHONPATH points at another Python installation (e.g. ROS), unset it before running.

## 2. Quick Start

### Open the viewer

```bash
gs-object-extraction-gui path/to/point_cloud.ply
```

Left drag orbits, right or middle drag pans, the wheel zooms and a double-click sets the rotation centre. The up axis is estimated from the floor (Auto) and can be set to one of the six axes in the Scene panel.

Press S to mark the object: a left click adds an object point, a right click a background point, and SAM2 draws the mask on the current view. Enter adds the view to the list, Backspace undoes the last point and Esc clears them. Moving the view drops points that were not added. Mark the object from several directions.

### Extract and export in the viewer

After adding views, click Extract object (Ctrl+E). The tool selects Gaussians from the confirmed masks, then renders the selection on its own to remove pieces outside the masks. Progress is shown while it runs; Cancel extraction stops after the current view finishes.

The preview switches to Object only when extraction finishes. Orbit and zoom to inspect it, use Reset view to fit the object, and choose White or Black under Object background. Choose Scene to return to marking views. Adding or removing a confirmed view clears the previous result; extract again after making changes.

Click Export object PLY (Ctrl+Shift+S) to save the cleaned object. The PLY retains its Gaussian parameters, spherical harmonic coefficients, IDs and extra attributes. Choose a different file from the source scene.

### Extract one 360-USID object from dataset masks

```bash
gs-object-extraction extract --scene-dir DATA/360-USID/cone --model-dir DATA/gs/cone --output outputs/cone
```

### Extract every 360-USID scene

```bash
python scripts/extract_360usid.py --data-root DATA --output outputs/360usid
```

DATA holds the dataset scenes (360-USID/scene_name) and the trained models (gs/scene_name).

## 3. Core Features

- Desktop Viewer: Open a Gaussian PLY and orbit, pan and zoom with GPU rendering; the view starts upright near the scene centre
- Click-to-mask: SAM2 turns object and background clicks on the rendered view into a mask; added views are kept for extraction
- Mask Lifting: Lift per-view object masks to Gaussians by their rendered contribution and keep the Gaussians drawn mostly inside the masks
- Off-mask Pruning: Render the selection on its own and remove Gaussians drawn mostly outside the masks, such as floor pieces hidden in the full scene
- Source-photo Free: Reads the trained PLY, camera poses and masks only
- Graphdeco Models: Reads cameras.json and cfg_args to separate training views from held-out views; held-out views are used only for checking
- 360-USID Scenes: Reads COLMAP sparse models, object masks and view splits in the 360-USID layout
- Outputs: object.ply (extracted object), summary.json (Gaussian counts and check metrics per stage), preview.png (before and after, on white and black backgrounds)

## 4. Tests

```bash
python3 -m pytest -q
GS_OBJECT_EXTRACTION_TEST_CUDA=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The second command runs the GPU tests inside the activated .venv-gpu.
