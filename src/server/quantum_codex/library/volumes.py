"""Telling an absent volume from a deleted model.

Models live on external SSDs. Unplugging one is routine, and it must never look
like the model was deleted — the difference decides whether the right response
is "plug the drive back in" or "download it again", and getting it wrong is how
a registry ends up quietly discarding entries for 60 GB of weights that are
perfectly intact on a disk sitting on the desk (cahier 43).

The distinction is structural rather than heuristic: on macOS an external volume
is mounted at ``/Volumes/<name>``, so a path under a ``/Volumes`` entry that is
not currently a mount point is unreachable *because the volume is away*, not
because anything was removed.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Where macOS mounts everything that is not the startup disk.
VOLUMES_ROOT = Path("/Volumes")


@dataclass(frozen=True)
class VolumeInfo:
    """The volume a path belongs to, and whether it is currently there."""

    #: Mount point, e.g. ``/Volumes/Models``. ``None`` for the startup disk.
    mount_point: Path | None
    #: Volume name as it appears in Finder, when it is external.
    name: str | None
    mounted: bool
    total_bytes: int | None = None
    free_bytes: int | None = None

    @property
    def is_external(self) -> bool:
        return self.mount_point is not None


def volume_for(path: str | Path) -> VolumeInfo:
    """Describe the volume ``path`` lives on.

    Works on a path that does not exist: that is the whole point — the question
    is asked precisely when the path cannot be reached.
    """
    resolved = Path(path).expanduser()
    parts = resolved.parts

    # Anything not under /Volumes is on the startup disk, which is always there.
    if len(parts) < 3 or Path(parts[0], parts[1]) != VOLUMES_ROOT:
        return VolumeInfo(mount_point=None, name=None, mounted=True, **_space(Path("/")))

    mount_point = VOLUMES_ROOT / parts[2]
    # `ismount` is the load-bearing check: the directory can exist as an empty
    # placeholder after an unclean eject, so existence alone would report a
    # volume that is not actually there.
    mounted = mount_point.is_mount()

    return VolumeInfo(
        mount_point=mount_point,
        name=parts[2],
        mounted=mounted,
        **(_space(mount_point) if mounted else {"total_bytes": None, "free_bytes": None}),
    )


def _space(path: Path) -> dict[str, int | None]:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return {"total_bytes": None, "free_bytes": None}
    return {"total_bytes": usage.total, "free_bytes": usage.free}


def free_bytes_for(path: str | Path) -> int | None:
    """Free space on the volume that would hold ``path``.

    Used before a download: promising to fetch 60 GB onto a disk with 8 GB left
    wastes an hour before failing (cahier 24).
    """
    target = Path(path).expanduser()
    # Walk up to the nearest existing ancestor: the target directory itself is
    # usually the thing about to be created.
    for candidate in [target, *target.parents]:
        if candidate.exists():
            try:
                return shutil.disk_usage(candidate).free
            except OSError:
                return None
    return None


def mounted_volumes() -> list[VolumeInfo]:
    """Every external volume currently attached."""
    if not VOLUMES_ROOT.is_dir():
        return []
    volumes: list[VolumeInfo] = []
    for entry in sorted(VOLUMES_ROOT.iterdir()):
        if entry.is_mount():
            volumes.append(volume_for(entry / "placeholder"))
    return volumes


def is_reachable(path: str | Path) -> bool:
    """Whether ``path`` could be read right now, volume included."""
    return volume_for(path).mounted and os.path.exists(Path(path).expanduser())
