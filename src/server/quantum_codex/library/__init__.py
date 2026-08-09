"""The model library: what is installed on disk, where, and whether it works.

Distinct from ``models.py``, which describes the capabilities of the model this
server is *currently serving*. One is a catalogue of files; the other is a
contract with Codex. Conflating them is how a library entry ends up deciding
what the API advertises.
"""

from .downloads import Download, DownloadError, DownloadProgress, DownloadState, repository_size
from .manager import MANAGER, DownloadManager
from .registry import (
    ModelEntry,
    ModelRegistry,
    ModelReport,
    ModelState,
    default_root,
    evaluate,
    load_registry,
    save_registry,
)
from .volumes import VolumeInfo, free_bytes_for, is_reachable, mounted_volumes, volume_for

__all__ = [
    "MANAGER",
    "Download",
    "DownloadError",
    "DownloadManager",
    "DownloadProgress",
    "DownloadState",
    "ModelEntry",
    "ModelRegistry",
    "ModelReport",
    "ModelState",
    "VolumeInfo",
    "default_root",
    "evaluate",
    "free_bytes_for",
    "is_reachable",
    "load_registry",
    "mounted_volumes",
    "save_registry",
    "repository_size",
    "volume_for",
]
