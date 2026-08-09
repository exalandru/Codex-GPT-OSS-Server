"""Fetching GPT-OSS weights from Hugging Face.

A 120B download is sixty gigabytes over an hour or more. That length shapes
every decision here:

**Preflight before starting.** Free space is checked against the repository's
declared size first. Discovering the disk is full after fifty minutes wastes the
fifty minutes and leaves a partial tree behind (cahier 24).

**Resume rather than restart.** ``huggingface_hub`` writes ``.incomplete`` files
and continues from them, so an interrupted download costs the bytes not yet
fetched and nothing more. That is also what makes the registry's
``PARTIAL_DOWNLOAD`` state actionable instead of merely descriptive.

**Cancellation is cooperative and honest.** The transfer runs on a worker
thread, which cannot be killed; cancelling sets a flag that the progress
callback observes between files. So cancelling a download stops it after the
current file, and the partial tree is left in place for a later resume — not
deleted, because deleting fifty gigabytes to honour a click is worse than
keeping them.

**Never guess a size.** Everything reported comes from the repository metadata
or from bytes actually on disk. A progress bar that invents a denominator is
worse than no progress bar.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .volumes import free_bytes_for, volume_for

logger = logging.getLogger(__name__)

#: Only the files MLX needs. Pulling a repository wholesale drags in PyTorch
#: weights that double the download for nothing.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
]

#: Headroom kept free beyond the download itself, so completing one does not
#: leave a machine with no room to breathe.
SPACE_MARGIN_BYTES = 2 * 1024**3


class DownloadState(StrEnum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    #: Cancellation accepted, transfer not yet stopped.
    #:
    #: A separate state because the two are separate facts. Cancellation is
    #: cooperative -- the worker checks between files -- so claiming CANCELLED
    #: the moment the button is pressed would assert something that has not
    #: happened yet. The interface can stop offering Cancel immediately while
    #: still telling the truth about the transfer.
    CANCELLING = "CANCELLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def finished(self) -> bool:
        return self in (DownloadState.COMPLETED, DownloadState.FAILED, DownloadState.CANCELLED)

    @property
    def stopping(self) -> bool:
        """Cancellation has been accepted; nothing more will be asked of it."""
        return self is DownloadState.CANCELLING


class DownloadError(Exception):
    """A download that cannot be started, with a reason worth showing."""


@dataclass
class DownloadProgress:
    """What is known about one download, right now."""

    repo: str
    destination: str
    state: DownloadState = DownloadState.PENDING
    total_bytes: int | None = None
    downloaded_bytes: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    detail: str | None = None

    @property
    def fraction(self) -> float | None:
        """Completed share, or ``None`` when the total is genuinely unknown."""
        if not self.total_bytes:
            return None
        return min(1.0, self.downloaded_bytes / self.total_bytes)

    @property
    def bytes_per_second(self) -> float | None:
        elapsed = (self.finished_at or time.time()) - self.started_at
        if elapsed <= 0 or self.downloaded_bytes <= 0:
            return None
        return self.downloaded_bytes / elapsed

    @property
    def eta_seconds(self) -> float | None:
        """Only when both a total and a rate are actually known."""
        rate = self.bytes_per_second
        if rate is None or not self.total_bytes:
            return None
        remaining = max(0, self.total_bytes - self.downloaded_bytes)
        return remaining / rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "destination": self.destination,
            "state": self.state.value,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "fraction": self.fraction,
            "bytes_per_second": self.bytes_per_second,
            "eta_seconds": self.eta_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "detail": self.detail,
        }


def repository_size(repo: str) -> int | None:
    """Total bytes of the files that would be fetched.

    ``None`` when the repository does not publish sizes, which is reported as
    an unknown total rather than filled in with a guess.
    """
    from fnmatch import fnmatch

    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception as exc:  # noqa: BLE001 - any failure is "size unknown"
        logger.debug("cannot read size of %s: %s", repo, exc)
        return None

    total = 0
    known = False
    for sibling in info.siblings or []:
        if not any(fnmatch(sibling.rfilename, pattern) for pattern in ALLOW_PATTERNS):
            continue
        if sibling.size is None:
            continue
        known = True
        total += sibling.size
    return total if known else None


def directory_bytes(path: Path) -> int:
    """Bytes on disk, including partial files, so resume reports honestly."""
    try:
        return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())
    except OSError:
        return 0


def preflight(repo: str, destination: Path, *, required_bytes: int | None) -> None:
    """Refuse a download that cannot succeed, before it starts."""
    volume = volume_for(destination)
    if not volume.mounted:
        raise DownloadError(
            f"the volume {volume.name!r} is not mounted, so {destination} cannot be written"
        )

    if required_bytes is None:
        # Unknown size is not a reason to refuse — it is a reason not to promise.
        logger.info("%s does not publish file sizes; proceeding without a space check", repo)
        return

    # Bytes already present count against the requirement: resuming a download
    # needs only what is left.
    already = directory_bytes(destination) if destination.exists() else 0
    needed = max(0, required_bytes - already) + SPACE_MARGIN_BYTES
    free = free_bytes_for(destination)
    if free is not None and free < needed:
        raise DownloadError(
            f"{destination} has {free / 1024**3:.1f} GiB free but this download needs about "
            f"{needed / 1024**3:.1f} GiB. Free space, or choose a different location."
        )


class Download:
    """One transfer, observable while it runs."""

    def __init__(self, repo: str, destination: Path) -> None:
        self.progress = DownloadProgress(repo=repo, destination=str(destination))
        self._destination = destination
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Ask the transfer to stop after the current file.

        Cooperative because the worker thread cannot be killed. The partial tree
        survives: throwing away fifty gigabytes to honour a click would be worse
        than keeping them for a resume.
        """
        self._cancelled.set()
        # Reported at once, so the interface can acknowledge the click without
        # pretending the bytes stopped moving.
        if not self.progress.state.finished:
            self.progress.state = DownloadState.CANCELLING

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def run(self, *, on_change: Callable[[DownloadProgress], None] | None = None) -> DownloadProgress:
        """Fetch the repository. Blocking; intended for a worker thread.

        Files are fetched one at a time rather than through ``snapshot_download``
        precisely so cancellation can mean something: that helper offers no way
        to interrupt it, so a cancel flag beside it would be decoration. Between
        files the flag is honoured, which is the granularity this actually has —
        a single 4 GB shard still has to finish.
        """
        from fnmatch import fnmatch

        from huggingface_hub import HfApi, hf_hub_download

        progress = self.progress

        try:
            files = [
                sibling.rfilename
                for sibling in HfApi().model_info(progress.repo, files_metadata=True).siblings or []
                if any(fnmatch(sibling.rfilename, pattern) for pattern in ALLOW_PATTERNS)
            ]
        except Exception as exc:  # noqa: BLE001 - a bad repo id lands here
            return self._finish(
                DownloadState.FAILED, f"cannot read {progress.repo}: {exc}", on_change
            )

        if not files:
            return self._finish(
                DownloadState.FAILED,
                f"{progress.repo} contains no files this server can use.",
                on_change,
            )

        progress.total_bytes = repository_size(progress.repo)
        try:
            preflight(progress.repo, self._destination, required_bytes=progress.total_bytes)
        except DownloadError as exc:
            return self._finish(DownloadState.FAILED, str(exc), on_change)

        progress.state = DownloadState.DOWNLOADING
        if on_change:
            on_change(progress)

        stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch, args=(stop, on_change), daemon=True, name="download-progress"
        )
        watcher.start()

        try:
            for filename in files:
                if self.cancelled:
                    break
                hf_hub_download(
                    repo_id=progress.repo,
                    filename=filename,
                    local_dir=str(self._destination),
                )
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
            return self._stop_watching(
                stop, watcher, DownloadState.FAILED, str(exc), on_change
            )
        finally:
            stop.set()
            watcher.join(timeout=2)

        if self.cancelled:
            return self._finish(
                DownloadState.CANCELLED,
                "Cancelled. The partial download was kept and can be resumed.",
                on_change,
            )
        return self._finish(DownloadState.COMPLETED, None, on_change)

    def _stop_watching(
        self,
        stop: threading.Event,
        watcher: threading.Thread,
        state: DownloadState,
        detail: str | None,
        on_change: Callable[[DownloadProgress], None] | None,
    ) -> DownloadProgress:
        stop.set()
        watcher.join(timeout=2)
        if self.cancelled:
            return self._finish(
                DownloadState.CANCELLED,
                "Cancelled. The partial download was kept and can be resumed.",
                on_change,
            )
        return self._finish(state, detail, on_change)

    def _watch(self, stop: threading.Event, on_change: Callable[[DownloadProgress], None] | None):
        """Report bytes actually on disk, once a second.

        Measured rather than accumulated from callbacks: a resumed download
        starts with gigabytes already present, and a counter that began at zero
        would report a fraction that runs backwards.
        """
        while not stop.wait(1.0):
            with self._lock:
                self.progress.downloaded_bytes = directory_bytes(self._destination)
            if on_change:
                on_change(self.progress)

    def _finish(
        self,
        state: DownloadState,
        detail: str | None,
        on_change: Callable[[DownloadProgress], None] | None,
    ) -> DownloadProgress:
        with self._lock:
            self.progress.state = state
            self.progress.detail = detail
            self.progress.finished_at = time.time()
            self.progress.downloaded_bytes = directory_bytes(self._destination)
        if on_change:
            on_change(self.progress)
        logger.info("download %s: %s%s", self.progress.repo, state.value, f" ({detail})" if detail else "")
        return self.progress
