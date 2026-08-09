"""Running downloads, and registering what they produce.

One download at a time. Two concurrent sixty-gigabyte transfers to the same disk
finish no sooner than one after the other and make both progress reports
meaningless, so a second request is refused with the reason rather than queued
invisibly.

The manager owns the *worker*; the registry owns the *record*. A completed
download is registered here so the library shows it without a rescan, and a
cancelled one is not — its partial tree is already reported as
``PARTIAL_DOWNLOAD`` by the registry if a root covers it.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from ..config import ConfigError
from .downloads import Download, DownloadError, DownloadProgress, DownloadState
from .registry import ModelRegistry, default_root, load_registry, save_registry

logger = logging.getLogger(__name__)

#: `owner/name`, the only shape Hugging Face repository ids take.
REPO_PATTERN = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class DownloadManager:
    """The single active download, if any, and the last one's outcome."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Download | None = None
        self._thread: threading.Thread | None = None
        self._last: DownloadProgress | None = None

    # -- state ---------------------------------------------------------------

    @property
    def active(self) -> DownloadProgress | None:
        with self._lock:
            if self._current is None or self._current.progress.state.finished:
                return None
            return self._current.progress

    @property
    def last(self) -> DownloadProgress | None:
        with self._lock:
            if self._current is not None and self._current.progress.state.finished:
                return self._current.progress
            return self._last

    # -- control -------------------------------------------------------------

    def start(self, repo: str, *, destination: Path | None = None) -> DownloadProgress:
        """Begin fetching ``repo``. Returns immediately.

        The destination defaults to the first configured root, so a download
        lands where the library already looks and appears without a rescan.
        """
        if not REPO_PATTERN.match(repo):
            raise DownloadError(
                f"{repo!r} is not a Hugging Face repository id. Expected owner/name, "
                "for example mlx-community/gpt-oss-20b-MXFP4-Q8."
            )

        with self._lock:
            if self._current is not None and not self._current.progress.state.finished:
                raise DownloadError(
                    f"{self._current.progress.repo} is already downloading. "
                    "Wait for it, or cancel it first."
                )

            target = Path(destination) if destination else _default_destination(repo)

            # A chosen location that is not there is a stopped drive, not a
            # reason to write somewhere else. Falling back to another disk would
            # put sixty gigabytes where the user did not ask for them, and they
            # would find out when it filled up.
            root = target.parent
            if not root.exists():
                try:
                    root.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise DownloadError(
                        f"the download location {root} is not available ({exc.strerror}). "
                        f"Reattach the volume, or choose another location in Models."
                    ) from exc

            download = Download(repo, target)
            self._current = download
            self._thread = threading.Thread(
                target=self._run, args=(download,), daemon=True, name=f"download-{repo}"
            )
            self._thread.start()
            return download.progress

    def cancel(self) -> DownloadProgress:
        """Ask the running download to stop. Idempotent.

        Cancelling something already cancelling, or already finished, is a
        no-op that reports the current state rather than an error. A user who
        clicks twice because the first click has not visibly landed has done
        nothing wrong, and answering the second click with a 400 is the
        interface blaming them for its own latency.
        """
        with self._lock:
            if self._current is None:
                raise DownloadError("no download has been started")
            # Already stopping, or already over: say where it is.
            self._current.cancel()
            return self._current.progress

    def _run(self, download: Download) -> None:
        progress = download.run()
        if progress.state is not DownloadState.COMPLETED:
            return

        # Registered here so the library shows it immediately. A failure to
        # register is logged rather than raised: the weights are on disk either
        # way, and a rescan would find them.
        try:
            registry: ModelRegistry = load_registry()
            registry.add(progress.destination, source="downloaded", repo=progress.repo)
            save_registry(registry)
        except ConfigError as refusal:
            # The library refused it on purpose -- most often a repository that
            # is not a usable GPT-OSS model. That is this server's
            # specialisation working, not a fault, so it gets one line with the
            # reason instead of a stack trace that reads like a crash.
            logger.info("downloaded %s but it is not usable here: %s", progress.repo, refusal)
        except Exception:  # noqa: BLE001
            logger.exception("downloaded %s but could not add it to the library", progress.repo)


def download_root() -> Path:
    """Where downloads are written.

    The user's chosen location if there is one, else the first scan root, else
    the app-owned directory. Chosen once for the installation, not per profile
    or per model.
    """
    from ..config import load_settings

    chosen = load_settings().download_root
    if chosen:
        return Path(chosen).expanduser()
    roots = load_registry().roots
    return Path(roots[0]).expanduser() if roots else default_root()


def _default_destination(repo: str) -> Path:
    """Where a repository lands by default.

    Named after the repository rather than its owner, so a library listing
    reads as model names.
    """
    return download_root() / repo.split("/", 1)[1]


#: One manager per process. Downloads are a property of the running server, not
#: of a request, so the state has to outlive the call that started it.
MANAGER = DownloadManager()
