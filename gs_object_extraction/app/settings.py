"""What the viewer remembers between runs: the folder a scene was last opened from."""

from pathlib import Path
from PySide6.QtCore import QSettings


def make_settings():
    return QSettings("gs-object-extraction", "gs-object-extraction")


def load_folder(settings):
    """The folder the last scene came from, or an empty string if there is none or it is gone."""
    folder = settings.value("folder", "", type=str)
    return folder if folder and Path(folder).is_dir() else ""


def save_folder(settings, folder):
    settings.setValue("folder", str(folder))
    settings.sync()
