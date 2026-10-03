"""What the viewer remembers between runs."""

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from gs_object_extraction.app.settings import load_folder, save_folder
from gs_object_extraction.app.window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


def test_no_folder_is_remembered_to_begin_with(settings):
    assert load_folder(settings) == ""


def test_the_folder_comes_back_while_it_is_still_there(settings, tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    save_folder(settings, scenes)
    assert load_folder(settings) == str(scenes)
    scenes.rmdir()
    assert load_folder(settings) == ""  # a folder that no longer exists is not offered


def test_the_open_dialog_starts_in_the_remembered_folder(app, settings, tmp_path, monkeypatch):
    save_folder(settings, tmp_path)
    window = MainWindow(settings=settings)
    asked = []
    monkeypatch.setattr("gs_object_extraction.app.window.QFileDialog.getOpenFileName",
                        lambda parent, title, folder, filters: asked.append(folder) or ("", ""))
    window.choose_ply()
    assert asked == [str(tmp_path)]
    window.close()
