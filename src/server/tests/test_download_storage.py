"""Where downloads go, and cancelling them.

Two separate concerns that both surfaced in the same smoke: a download location
the user cannot choose, and a Cancel that answered a second click with an error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantum_codex.config import Settings, load_settings, save_settings
from quantum_codex.library.downloads import DownloadError, DownloadState
from quantum_codex.library.manager import DownloadManager, download_root
from quantum_codex.library.registry import ModelRegistry, default_root, save_registry


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path))
    return tmp_path


# -- the download location ---------------------------------------------------


def test_a_clean_install_downloads_into_the_app_owned_directory(home) -> None:
    assert download_root() == default_root()


def test_a_chosen_location_is_used(home) -> None:
    save_settings(Settings(download_root="/tmp/somewhere"))

    assert download_root().as_posix() == "/tmp/somewhere"


def test_the_choice_survives_a_reload(home) -> None:
    save_settings(Settings(download_root="/Volumes/External/models"))

    assert load_settings().download_root == "/Volumes/External/models"
    assert download_root().as_posix() == "/Volumes/External/models"


def test_an_external_volume_is_remembered_even_while_unmounted(home) -> None:
    """A stopped drive is not a reason to silently pick another disk."""
    save_settings(Settings(download_root="/Volumes/NotMounted/models"))

    assert download_root().as_posix() == "/Volumes/NotMounted/models"


def test_without_a_choice_downloads_follow_the_librarys_first_root(home) -> None:
    """Which, on a clean install, is the app-owned directory the registry seeds."""
    from quantum_codex.library.registry import load_registry

    assert download_root() == Path(load_registry().roots[0]).expanduser()


def test_an_explicit_choice_beats_a_scan_root(home) -> None:
    scanned = home / "scanned"
    scanned.mkdir()
    registry = ModelRegistry()
    registry.add_root(scanned)
    save_registry(registry)
    save_settings(Settings(download_root="/tmp/chosen"))

    assert download_root().as_posix() == "/tmp/chosen"


def test_changing_the_location_leaves_the_library_untouched(home) -> None:
    """Nothing is moved and no entry is rewritten: a model's identity has never
    been its path, so a new download root says nothing about existing ones."""
    from quantum_codex.library.registry import load_registry

    before = [r.entry.path for r in load_registry().report()]

    save_settings(Settings(download_root="/somewhere/else"))

    assert [r.entry.path for r in load_registry().report()] == before


def test_an_unavailable_location_fails_rather_than_falling_back(home) -> None:
    """Sixty gigabytes on the wrong disk is worse than a clear refusal."""
    save_settings(Settings(download_root="/Volumes/NotMounted/models"))
    manager = DownloadManager()

    with pytest.raises(DownloadError, match="not available|No such file"):
        manager.start("mlx-community/gpt-oss-20b-MXFP4-Q8")


def test_the_destination_is_named_after_the_repository(home) -> None:
    save_settings(Settings(download_root=str(home / "weights")))

    from quantum_codex.library.manager import _default_destination

    assert _default_destination("mlx-community/gpt-oss-20b-MXFP4-Q8").name == (
        "gpt-oss-20b-MXFP4-Q8"
    )


# -- cancellation ------------------------------------------------------------


def test_cancelling_is_reported_before_the_transfer_has_stopped(home) -> None:
    """The two are different facts. Claiming CANCELLED at the click would be a
    lie about bytes still moving."""
    from quantum_codex.library.downloads import Download

    download = Download("owner/name", home / "dest")
    download.progress.state = DownloadState.DOWNLOADING

    download.cancel()

    assert download.progress.state is DownloadState.CANCELLING
    assert download.cancelled is True
    assert DownloadState.CANCELLING.finished is False


def test_cancelling_twice_is_not_an_error(home) -> None:
    """A user clicking again because nothing visibly happened has done nothing
    wrong; answering with a 400 blames them for our latency."""
    from quantum_codex.library.downloads import Download

    download = Download("owner/name", home / "dest")
    download.progress.state = DownloadState.DOWNLOADING

    download.cancel()
    download.cancel()

    assert download.progress.state is DownloadState.CANCELLING


def test_cancelling_a_finished_download_does_not_rewrite_its_outcome(home) -> None:
    from quantum_codex.library.downloads import Download

    download = Download("owner/name", home / "dest")
    download.progress.state = DownloadState.COMPLETED

    download.cancel()

    assert download.progress.state is DownloadState.COMPLETED


def test_cancelling_with_nothing_ever_started_is_still_an_error(home) -> None:
    """Distinct from cancelling twice: there is nothing to report on."""
    manager = DownloadManager()

    with pytest.raises(DownloadError, match="no download"):
        manager.cancel()


def test_a_cancelling_state_is_not_treated_as_finished() -> None:
    assert DownloadState.CANCELLING.stopping is True
    assert DownloadState.DOWNLOADING.stopping is False
    assert DownloadState.CANCELLED.finished is True
