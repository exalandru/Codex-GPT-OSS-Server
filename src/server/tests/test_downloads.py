"""Download preflight, progress honesty and cancellation.

A 120B fetch is an hour of network. The assertions here are about the decisions
made *before* and *around* that hour — refusing what cannot succeed, never
inventing a number, and keeping a partial tree that can be resumed — because
those are what make the hour survivable. The transfer itself is
``huggingface_hub``'s.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantum_codex.library.downloads import (
    SPACE_MARGIN_BYTES,
    Download,
    DownloadError,
    DownloadProgress,
    DownloadState,
    directory_bytes,
    preflight,
)
from quantum_codex.library.manager import DownloadManager


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path / "home"))


# -- preflight ---------------------------------------------------------------


def test_a_download_to_an_absent_volume_is_refused_immediately() -> None:
    with pytest.raises(DownloadError, match="not mounted"):
        preflight(
            "mlx-community/gpt-oss-20b",
            __import__("pathlib").Path("/Volumes/DefinitelyNotAttached/models/x"),
            required_bytes=1024,
        )


def test_a_download_larger_than_the_disk_is_refused_before_it_starts(tmp_path) -> None:
    """Fifty minutes then "disk full" wastes the fifty minutes."""
    with pytest.raises(DownloadError, match="free"):
        preflight("owner/huge", tmp_path / "target", required_bytes=1 << 60)


def test_an_unknown_size_does_not_block_the_download(tmp_path) -> None:
    # Not knowing the size is a reason not to promise, not a reason to refuse.
    preflight("owner/model", tmp_path / "target", required_bytes=None)


def test_bytes_already_present_count_towards_the_requirement(tmp_path) -> None:
    """Resuming needs only what is left, not the whole repository again."""
    target = tmp_path / "partial"
    target.mkdir()
    (target / "shard.safetensors").write_bytes(b"x" * 4096)

    # Requiring slightly more than what is present must pass, because only the
    # difference is still needed.
    preflight("owner/model", target, required_bytes=4096 + 1024)

    assert directory_bytes(target) == 4096


def test_the_margin_is_kept_beyond_the_download(tmp_path) -> None:
    # Completing a download must not leave a machine with nothing free.
    free = __import__("shutil").disk_usage(tmp_path).free
    with pytest.raises(DownloadError):
        preflight("owner/model", tmp_path / "t", required_bytes=free - SPACE_MARGIN_BYTES // 2)


# -- progress honesty --------------------------------------------------------


def test_an_unknown_total_reports_no_fraction_and_no_eta() -> None:
    """A progress bar that invents a denominator is worse than none."""
    progress = DownloadProgress(repo="owner/model", destination="/tmp/x", downloaded_bytes=1024)

    assert progress.total_bytes is None
    assert progress.fraction is None
    assert progress.eta_seconds is None


def test_a_known_total_gives_a_bounded_fraction() -> None:
    progress = DownloadProgress(
        repo="owner/model", destination="/tmp/x", total_bytes=1000, downloaded_bytes=250
    )

    assert progress.fraction == 0.25


def test_the_fraction_never_exceeds_one() -> None:
    # A resumed tree can hold more bytes than the allow-list total, and 130%
    # would read as a bug rather than as rounding.
    progress = DownloadProgress(
        repo="owner/model", destination="/tmp/x", total_bytes=1000, downloaded_bytes=1300
    )

    assert progress.fraction == 1.0


def test_a_finished_download_is_recognised_as_such() -> None:
    assert DownloadState.COMPLETED.finished
    assert DownloadState.CANCELLED.finished
    assert DownloadState.FAILED.finished
    assert not DownloadState.DOWNLOADING.finished


# -- the manager -------------------------------------------------------------


def test_a_malformed_repository_id_is_refused_with_an_example() -> None:
    with pytest.raises(DownloadError, match="owner/name"):
        DownloadManager().start("not-a-repo")


def test_a_bare_owner_is_not_a_repository() -> None:
    with pytest.raises(DownloadError, match="owner/name"):
        DownloadManager().start("mlx-community")


def test_cancelling_with_nothing_running_says_so() -> None:
    with pytest.raises(DownloadError, match="no download"):
        DownloadManager().cancel()


def test_nothing_is_active_before_a_download_starts() -> None:
    manager = DownloadManager()

    assert manager.active is None
    assert manager.last is None


# -- cancellation is idempotent ------------------------------------------------
#
# A user who clicks Cancel twice because the first click has not visibly landed
# has done nothing wrong. These pin that the second click is a no-op reporting
# the current state, never an error the interface would have to explain away.


def _running(manager: DownloadManager, repo: str = "owner/model") -> Download:
    """A manager with a download in flight, without any network.

    The transfer itself belongs to ``huggingface_hub``; what is under test is
    the manager's own state machine around it, so the worker is never started.
    """
    download = Download(repo, Path("/tmp/nowhere"))
    download.progress.state = DownloadState.DOWNLOADING
    manager._current = download  # noqa: SLF001 - the state a started download leaves
    return download


def test_cancelling_a_running_download_reports_cancelling_not_cancelled() -> None:
    """Two separate facts: the request was accepted, the bytes have not stopped.

    Claiming CANCELLED here would assert something the cooperative worker has
    not done yet.
    """
    manager = DownloadManager()
    _running(manager)

    assert manager.cancel().state is DownloadState.CANCELLING


def test_cancelling_twice_is_a_no_op_that_reports_the_same_state() -> None:
    manager = DownloadManager()
    _running(manager)

    first = manager.cancel()
    second = manager.cancel()

    assert first.state is DownloadState.CANCELLING
    assert second.state is DownloadState.CANCELLING
    assert second is first


def test_cancelling_many_times_never_raises() -> None:
    """What a double-click, or an impatient user, actually produces."""
    manager = DownloadManager()
    _running(manager)

    for _ in range(5):
        manager.cancel()

    assert manager.active.state is DownloadState.CANCELLING


@pytest.mark.parametrize(
    "state", [DownloadState.COMPLETED, DownloadState.CANCELLED, DownloadState.FAILED]
)
def test_cancelling_something_already_over_reports_it_rather_than_failing(
    state: DownloadState,
) -> None:
    """A cancel that raced a finishing transfer must not become an error.

    The poll runs every two seconds, so the button can still be on screen when
    the download ends. Answering that click with a refusal would blame the user
    for the interface's latency.
    """
    manager = DownloadManager()
    download = _running(manager)
    download.progress.state = state

    assert manager.cancel().state is state


def test_a_cancelled_download_keeps_its_partial_tree() -> None:
    """Deleting fifty gigabytes to honour a click is worse than keeping them."""
    manager = DownloadManager()
    download = _running(manager)
    manager.cancel()

    assert download.cancelled is True
    # Nothing here removes anything: the registry reports the tree as
    # PARTIAL_DOWNLOAD and a later start resumes into it.
    assert manager.active.destination == "/tmp/nowhere"
