# Third-party licenses

None of these is bundled in this repository. They are installed with pip (see [requirements.txt](../requirements.txt)) or downloaded on first use, and each stays under its own license.

| Component | Version | License |
| --- | --- | --- |
| [SAM 2](https://github.com/facebookresearch/sam2) | commit pinned in requirements.txt | [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| SAM 2.1 checkpoint (downloaded on first use) | sam2.1_hiera_base_plus | [Apache-2.0](https://github.com/facebookresearch/sam2#license) |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | 1.5.3 | [Apache-2.0](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) and torchvision | unpinned | [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [NumPy](https://github.com/numpy/numpy) | >=1.24 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [Qt for Python (PySide6)](https://doc.qt.io/qtforpython-6/) | >=6.7 | [LGPL-3.0, GPL-2.0 or GPL-3.0](https://doc.qt.io/qtforpython-6/licenses.html) |

PySide6 is used as an installed library and is not modified. A build that bundles it, such as a frozen executable, has to meet the LGPL's terms for redistribution.
