"""What the viewer remembers between runs: the folder a scene was last opened from and the export options."""

from pathlib import Path
from PySide6.QtCore import QSettings
from .export import ExportOptions


def make_settings():
    return QSettings("gs-object-extraction", "gs-object-extraction")


def load_folder(settings):
    """The folder the last scene came from, or an empty string if there is none or it is gone."""
    folder = settings.value("folder", "", type=str)
    return folder if folder and Path(folder).is_dir() else ""


def save_folder(settings, folder):
    settings.setValue("folder", str(folder))
    settings.sync()


def load_export_options(settings):
    degree = settings.value("export/sh_degree", 3, type=int)
    return ExportOptions(upright=settings.value("export/upright", False, type=bool),
                         ground=settings.value("export/ground", False, type=bool),
                         sh_degree=min(max(degree, 0), 3))


def save_export_options(settings, options):
    settings.setValue("export/upright", options.upright)
    settings.setValue("export/ground", options.ground)
    settings.setValue("export/sh_degree", options.sh_degree)
    settings.sync()
