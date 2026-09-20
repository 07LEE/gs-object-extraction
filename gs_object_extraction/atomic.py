"""Durable atomic file replacement, so an interrupted write cannot damage its target.

``replace_atomically`` writes through a private temporary file in the target's own
directory and renames it into place, flushing both the data and the directory entry
to the disk. A target therefore always holds either its previous content or the
complete new one, through a failed write, a crash of this process and a power loss.
"""

import os
from pathlib import Path
import stat
import tempfile
import time


_PREFIX = "."
_SUFFIX = ".tmp"
# A temporary this old belongs to a process that was killed before it could clean up;
# no write runs for an hour. Sweeping only well past any live write leaves litter
# behind rather than ever removing a temporary another process is still filling.
STALE_SECONDS = 3600


def target_mode(path) -> int:
    """Mode for a written file: keep an existing target's mode, otherwise the umask default."""
    try:
        return stat.S_IMODE(Path(path).stat().st_mode)  # one call, so no exists()/stat() race
    except (FileNotFoundError, NotADirectoryError):
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _clear_stale(path) -> None:
    """Remove temporaries a killed process left behind for this target."""
    prefix = f"{_PREFIX}{path.name}."
    try:
        entries = list(os.scandir(path.parent))
    except OSError:
        return  # a missing or unreadable directory fails loudly when the write starts
    cutoff = time.time() - STALE_SECONDS
    for entry in entries:
        # Exact prefix and suffix rather than a glob: a target name may hold [ or *.
        if not entry.name.startswith(prefix) or not entry.name.endswith(_SUFFIX):
            continue
        try:
            if entry.is_file(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                os.unlink(entry.path)
        except OSError:
            continue  # already gone, or not ours to remove


def _sync_directory(directory) -> None:
    """Make the rename itself durable, not only the bytes it points at."""
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return  # platforms that cannot open a directory keep only the rename's own ordering
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_atomically(path, write, *, mode=None) -> None:
    """Replace ``path`` with whatever ``write`` puts on the binary stream it is given.

    The stream is a new file in the target's directory, readable only by this user
    until it holds the finished content. Anything ``write`` raises leaves the previous
    ``path`` untouched and no temporary behind.
    """
    path = Path(path)
    _clear_stale(path)
    if mode is None:
        mode = target_mode(path)
    descriptor, name = tempfile.mkstemp(prefix=f"{_PREFIX}{path.name}.", suffix=_SUFFIX, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())  # the rename orders the data, it does not store it
        os.chmod(temporary, mode)  # private (0600) until now
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _sync_directory(path.parent)
