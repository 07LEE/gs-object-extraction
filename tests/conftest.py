import os
import shutil
from pathlib import Path

# Windows made by the tests remember into a folder of their own, not the user's settings. Qt reads
# XDG_CONFIG_HOME when it first looks for its settings, so this is set before any window is made, and
# the folder is emptied first so one run does not start with what the last one remembered.
_CONFIG = Path(__file__).resolve().parents[1] / ".pytest_cache" / "config"
shutil.rmtree(_CONFIG, ignore_errors=True)
os.environ["XDG_CONFIG_HOME"] = str(_CONFIG)
