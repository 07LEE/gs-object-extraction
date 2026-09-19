"""Exercise installation control flow without downloading GPU packages."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def installer(tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / "setup_env.sh"
    shutil.copy2(source, root / "scripts" / "setup_env.sh")
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "sam2.1_hiera_base_plus.pt").write_text("existing checkpoint")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    python = binaries / "python"
    python.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, shutil, sys
args = sys.argv[1:]
with open(os.environ["INSTALL_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[:3] == ["-I", "-m", "venv"]:
    dest = pathlib.Path(args[3])
    (dest / "bin").mkdir(parents=True)
    shutil.copy2(__file__, dest / "bin" / "python")
    (dest / "pyvenv.cfg").write_text("include-system-site-packages = false\\n")
elif args[:2] == ["-I", "-c"] and "sys.prefix" in args[2]:
    cfg = pathlib.Path(__file__).parent.parent / "pyvenv.cfg"
    sys.exit(not cfg.exists() or "= false" not in cfg.read_text())
elif args[:1] == ["-c"] and "assert torch.cuda.is_available()" in args[1]:
    sys.exit(int(os.environ.get("FAIL_GPU", "0")))
''')
    python.chmod(0o755)
    nvcc = binaries / "nvcc"
    nvcc.write_text('#!/bin/sh\necho "Cuda compilation tools, release ${TEST_CUDA:-12.4}, V12.4"\n')
    nvcc.chmod(0o755)
    env = os.environ.copy()
    env.pop("CUDA_HOME", None)
    env.pop("GS_PYTHON", None)
    env.update(PYTHON=str(python), VENV=".venv-gpu", INSTALL_LOG=str(tmp_path / "calls"),
               PATH=f"{binaries}:{env['PATH']}")
    return root, env


def run_setup(installer):
    root, env = installer
    return subprocess.run(["bash", str(root / "scripts" / "setup_env.sh")],
                          cwd=root.parent, env=env, text=True, capture_output=True)


def test_fresh_install_and_rerun(installer):
    import json
    root, env = installer
    result = run_setup(installer)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["INSTALL_LOG"]).read_text().splitlines()]
    assert ["-I", "-m", "venv", ".venv-gpu"] in calls
    torch = next(i for i, c in enumerate(calls) if "torch==2.6.0" in c)
    renderer = next(i for i, c in enumerate(calls) if any("git+https://github.com/graphdeco" in a for a in c))
    sam = next(i for i, c in enumerate(calls) if "requirements.txt" in c)
    assert torch < renderer < sam
    assert "--system-site-packages" not in Path(env["INSTALL_LOG"]).read_text()
    sentinel = root / ".venv-gpu" / "preserve"
    sentinel.write_text("keep")
    assert run_setup(installer).returncode == 0
    assert sentinel.read_text() == "keep"
    assert not list(root.glob(".venv-gpu.backup.*"))


def test_legacy_environment_is_preserved_and_replaced(installer):
    root, env = installer
    old = root / ".venv-gpu"
    (old / "bin").mkdir(parents=True)
    shutil.copy2(env["PYTHON"], old / "bin" / "python")
    (old / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
    (old / "preserve").write_text("old packages")
    result = run_setup(installer)
    assert result.returncode == 0, result.stderr
    backup, = root.glob(".venv-gpu.backup.*/environment")
    assert (backup / "preserve").read_text() == "old packages"
    assert "= false" in (old / "pyvenv.cfg").read_text()
    assert not (old / "preserve").exists()


def test_unsupported_toolkit_does_not_change_environment(installer):
    root, env = installer
    env["TEST_CUDA"] = "10.2"
    result = run_setup(installer)
    assert result.returncode != 0
    assert "Unsupported CUDA Toolkit" in result.stderr
    assert not (root / ".venv-gpu").exists()


def test_gpu_failure_stops_before_renderer_build(installer):
    root, env = installer
    env["FAIL_GPU"] = "1"
    assert run_setup(installer).returncode != 0
    calls = Path(env["INSTALL_LOG"]).read_text()
    assert "torch==2.6.0" in calls
    assert "git+https://github.com/graphdeco" not in calls
