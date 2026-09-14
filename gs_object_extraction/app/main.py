"""Start the desktop viewer.

    gs-object-extraction-gui [scene.ply]
"""

import argparse
import sys
from PySide6.QtWidgets import QApplication
from .window import MainWindow


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gs-object-extraction-gui", description="Open a Gaussian PLY in the viewer.")
    parser.add_argument("ply", nargs="?", help="Gaussian PLY file to open")
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    if args.ply:
        window.open_ply(args.ply)
    return app.exec()
