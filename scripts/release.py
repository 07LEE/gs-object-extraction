"""Prepare a release: bump the version in pyproject.toml and stage it.

Usage: python scripts/release.py 0.2.1

Run it on a clean main. It edits and stages pyproject.toml, shows the notes the
release will carry, then prints the commit message and the commands that
tag and push. It never commits or pushes; pushing the tag starts the Release
workflow.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from release_notes import notes, repo_slug

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
VERSION_LINE = re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.MULTILINE)


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def parse(version):
    return tuple(int(part) for part in version.split("."))


def fail(message):
    sys.exit(f"error: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version", help="new version such as 0.2.1, without the v")
    version = parser.parse_args().version

    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        fail(f"{version!r} is not X.Y.Z")
    if git("branch", "--show-current") != "main":
        fail("switch to main first")
    if git("status", "--porcelain"):
        fail("the working tree is not clean")

    text = PYPROJECT.read_text()
    match = VERSION_LINE.search(text)
    if not match:
        fail("no version line found in pyproject.toml")
    current = match[1]
    if parse(version) <= parse(current):
        fail(f"{version} is not newer than the current {current}")
    if git("tag", "--list", f"v{version}"):
        fail(f"tag v{version} already exists")

    PYPROJECT.write_text(VERSION_LINE.sub(f'version = "{version}"', text, count=1))
    git("add", str(PYPROJECT))

    print(f"pyproject.toml: {current} -> {version} (staged)\n")
    print("Release notes preview:\n")
    print(notes(f"v{version}", repo_slug(), ref="HEAD"))
    print(
        f"\nNext:\n"
        f'  git commit -m "chore: bump version to {version}"\n'
        f"  git tag v{version}\n"
        f"  git push origin main v{version}"
    )


if __name__ == "__main__":
    main()
