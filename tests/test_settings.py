"""What the viewer remembers between runs."""

import os
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from gs_object_extraction.app.export import ExportDialog, ExportOptions
from gs_object_extraction.app.settings import (load_export_options, load_folder, save_export_options, save_folder)
from gs_object_extraction.app.window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)


def test_nothing_is_remembered_to_begin_with(settings):
    assert load_folder(settings) == ""
    assert load_export_options(settings) == ExportOptions()


def test_the_folder_comes_back_while_it_is_still_there(settings, tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    save_folder(settings, scenes)
    assert load_folder(settings) == str(scenes)
    scenes.rmdir()
    assert load_folder(settings) == ""  # a folder that no longer exists is not offered


def test_the_export_options_come_back_as_saved(settings):
    options = ExportOptions(upright=True, ground=True, sh_degree=1)
    save_export_options(settings, options)
    assert load_export_options(settings) == options


def test_a_degree_outside_the_range_is_brought_back_in(settings):
    settings.setValue("export/sh_degree", 9)
    assert load_export_options(settings).sh_degree == 3
    settings.setValue("export/sh_degree", -2)
    assert load_export_options(settings).sh_degree == 0


def test_a_new_window_starts_with_what_was_remembered(app, settings):
    save_export_options(settings, ExportOptions(ground=True, sh_degree=2))
    window = MainWindow(settings=settings)
    assert window.export_options == ExportOptions(ground=True, sh_degree=2)
    window.close()


def test_confirming_the_export_options_remembers_them(app, settings, monkeypatch):
    window = MainWindow(settings=settings)
    window.export_action.setEnabled(True)

    def accept(self):
        self.upright_box.setChecked(True)
        self.compression_box.setCurrentIndex(3)
        return 1

    monkeypatch.setattr(ExportDialog, "exec", accept)
    monkeypatch.setattr("gs_object_extraction.app.window.QFileDialog.exec", lambda self: 0)
    window.choose_export()
    assert load_export_options(settings) == ExportOptions(upright=True, sh_degree=0)
    window.close()


def test_the_open_dialog_starts_in_the_remembered_folder(app, settings, tmp_path, monkeypatch):
    save_folder(settings, tmp_path)
    window = MainWindow(settings=settings)
    asked = []
    monkeypatch.setattr("gs_object_extraction.app.window.QFileDialog.getOpenFileName",
                        lambda parent, title, folder, filters: asked.append(folder) or ("", ""))
    window.choose_ply()
    assert asked == [str(tmp_path)]
    window.close()
