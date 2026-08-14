"""The model library: what is installed, where, and whether it can be used.

## The invariant that shapes this file

**A scan never removes an entry.** Registered models are remembered by absolute
path; discovery only ever *adds*. Unplugging an external SSD makes 60 GB of
perfectly intact weights unreadable for a while, and a registry that pruned
whatever it could not see would erase the record of them — leaving the user to
re-download what is sitting on the desk beside them (cahier 23, 43).

So an entry that cannot be read is *reported* as unreachable, never forgotten.
Removal is an explicit act.

## States

``READY``             present, validated, usable
``INCOMPATIBLE``      present, but not a GPT-OSS model this server can run
``INVALID``           present, but malformed — missing weights or tokenizer
``PARTIAL_DOWNLOAD``  an interrupted download; resumable, not usable
``MISSING_VOLUME``    its external volume is not mounted
``MISSING``           the volume is there and the directory is not

The last two are deliberately distinct. They lead to different actions —
"plug the drive in" against "download it again" — and collapsing them would
routinely give the wrong advice.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..config import ConfigError, app_support_dir, read_json, write_json
from ..inspect_model import Verdict, inspect_model
from .volumes import VolumeInfo, volume_for

logger = logging.getLogger(__name__)

MODELS_VERSION = 2

#: Marker huggingface_hub leaves while a download is in flight.
INCOMPLETE_SUFFIXES = (".incomplete", ".part", ".tmp")


class ModelState(StrEnum):
    READY = "READY"
    INCOMPATIBLE = "INCOMPATIBLE"
    INVALID = "INVALID"
    PARTIAL_DOWNLOAD = "PARTIAL_DOWNLOAD"
    MISSING_VOLUME = "MISSING_VOLUME"
    MISSING = "MISSING"

    @property
    def usable(self) -> bool:
        return self is ModelState.READY

    @property
    def recoverable_by_reattaching(self) -> bool:
        """Whether the fix is hardware rather than a download."""
        return self is ModelState.MISSING_VOLUME


def models_path() -> Path:
    return app_support_dir() / "models.json"


def default_root() -> Path:
    """Where downloads land when nothing else is configured."""
    return app_support_dir() / "models"


@dataclass
class ModelEntry:
    """A model the user has told us about.

    ``id`` is the immutable library identity. The path locates the weights; the
    display and served names are mutable metadata owned elsewhere. Keeping those
    concepts separate is what lets either user-facing name change without
    orphaning this record or its settings.
    """

    path: str
    id: str = ""
    #: How it got here, which decides whether deleting it is recoverable.
    source: str = "imported"  # "imported" | "downloaded"
    #: Hugging Face repository, when it was downloaded from one.
    repo: str | None = None
    added_at: float = field(default_factory=time.time)
    last_used_at: float | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name


def _base_model_id(path: str | Path) -> str:
    """A readable deterministic id for a newly registered library entry.

    Deliberately `slug_for` and not the catalogue's `catalog_slug_for`. This id
    is *persisted* -- it is written to `models.json` and it keys
    `model-settings.json` -- so it must depend only on the directory, never on a
    catalogue that ships new entries with new versions. Resolving it through the
    catalogue would give one unchanged directory two different ids across two
    releases, and the user's per-model settings would quietly stop applying.
    Which catalogue entry a directory belongs to is a separate question, asked
    separately, by `catalog_slug_for`.
    """
    from ..models import slug_for

    base = slug_for(Path(path).name)
    if base:
        return base
    digest = hashlib.sha256(str(Path(path).expanduser()).encode()).hexdigest()[:12]
    return f"model-{digest}"


def _allocate_model_id(path: str | Path, claimed: set[str]) -> str:
    """Allocate one stable id, disambiguating equal directory names by path."""
    base = _base_model_id(path)
    if base not in claimed:
        return base
    digest = hashlib.sha256(str(Path(path).expanduser()).encode()).hexdigest()[:8]
    candidate = f"{base}-{digest}"
    suffix = 2
    while candidate in claimed:
        candidate = f"{base}-{digest}-{suffix}"
        suffix += 1
    return candidate


@dataclass(frozen=True)
class ModelReport:
    """An entry plus everything established about it right now.

    Recomputed on every listing rather than cached: a volume can be attached or
    removed between two calls, and a stale "READY" is exactly the lie this
    module exists to avoid.
    """

    entry: ModelEntry
    state: ModelState
    detail: str
    volume: VolumeInfo
    quantization: str | None = None
    context_length: int | None = None
    layers: int | None = None
    experts: int | None = None
    disk_bytes: int = 0
    shards: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.entry),
            "name": self.entry.name,
            "state": self.state.value,
            "detail": self.detail,
            "usable": self.state.usable,
            "volume": {
                "name": self.volume.name,
                "mount_point": str(self.volume.mount_point) if self.volume.mount_point else None,
                "external": self.volume.is_external,
                "mounted": self.volume.mounted,
                "free_bytes": self.volume.free_bytes,
                "total_bytes": self.volume.total_bytes,
            },
            "quantization": self.quantization,
            "context_length": self.context_length,
            "layers": self.layers,
            "experts": self.experts,
            "disk_bytes": self.disk_bytes,
            "shards": self.shards,
        }


def _has_incomplete_files(directory: Path) -> bool:
    try:
        return any(
            entry.name.endswith(INCOMPLETE_SUFFIXES)
            for entry in directory.rglob("*")
            if entry.is_file()
        )
    except OSError:
        return False


def evaluate(entry: ModelEntry) -> ModelReport:
    """Establish the current state of one entry. Never mutates anything."""
    path = Path(entry.path).expanduser()
    volume = volume_for(path)

    if not volume.mounted:
        return ModelReport(
            entry=entry,
            state=ModelState.MISSING_VOLUME,
            detail=(
                f"The volume {volume.name!r} is not mounted. The model is intact; "
                "reattach the drive to use it."
            ),
            volume=volume,
        )

    if not path.exists():
        return ModelReport(
            entry=entry,
            state=ModelState.MISSING,
            detail="The volume is available but this directory no longer exists.",
            volume=volume,
        )

    # Checked before validation: a half-downloaded model has a config.json and
    # would otherwise be reported as merely invalid, which hides the fact that
    # resuming would fix it.
    if _has_incomplete_files(path):
        return ModelReport(
            entry=entry,
            state=ModelState.PARTIAL_DOWNLOAD,
            detail="A download was interrupted. Resuming will complete it.",
            volume=volume,
        )

    report = inspect_model(path)
    state = {
        Verdict.SUPPORTED: ModelState.READY,
        Verdict.SUPPORTED_WITH_WARNING: ModelState.READY,
    }.get(report.verdict)

    if state is None:
        # Distinguish "not our family" from "our family, but broken": the first
        # is permanent, the second is worth re-fetching.
        state = (
            ModelState.INCOMPATIBLE
            if report.model_type not in (None, "gpt_oss")
            else ModelState.INVALID
        )

    return ModelReport(
        entry=entry,
        state=state,
        detail=report.reasons[0] if report.reasons else "",
        volume=volume,
        quantization=report.quantization,
        context_length=report.context_length,
        layers=report.layers,
        experts=report.experts,
        disk_bytes=report.disk_bytes,
        shards=report.shards,
    )


class ModelRegistry:
    """The persisted library, plus discovery over configured roots."""

    def __init__(self, entries: list[ModelEntry] | None = None, roots: list[str] | None = None):
        claimed: set[str] = set()
        for entry in entries or []:
            if not entry.id or entry.id in claimed:
                entry.id = _allocate_model_id(entry.path, claimed)
            claimed.add(entry.id)
        self._entries: dict[str, ModelEntry] = {
            str(Path(e.path).expanduser()): e for e in (entries or [])
        }
        self._roots: list[str] = roots if roots is not None else [str(default_root())]

    # -- roots ---------------------------------------------------------------

    @property
    def roots(self) -> list[str]:
        return list(self._roots)

    def add_root(self, path: str | Path) -> None:
        resolved = str(Path(path).expanduser())
        if resolved not in self._roots:
            self._roots.append(resolved)

    def remove_root(self, path: str | Path) -> None:
        resolved = str(Path(path).expanduser())
        if resolved not in self._roots:
            raise ConfigError(f"{resolved} is not a configured model root")
        self._roots.remove(resolved)

    # -- entries -------------------------------------------------------------

    def add(self, path: str | Path, *, source: str = "imported", repo: str | None = None) -> ModelEntry:
        """Register a model directory.

        Validation happens here so an unusable directory is refused at the point
        the user chose it, rather than at load time when the context is gone.
        """
        resolved = Path(path).expanduser()
        volume = volume_for(resolved)
        if not volume.mounted:
            raise ConfigError(
                f"the volume {volume.name!r} is not mounted, so {resolved} cannot be inspected"
            )
        if not resolved.is_dir():
            raise ConfigError(f"{resolved} is not a directory")

        report = inspect_model(resolved)
        if report.verdict is Verdict.UNSUPPORTED:
            reason = report.reasons[0] if report.reasons else "not a usable GPT-OSS model"
            raise ConfigError(f"{resolved} cannot be used: {reason}")

        claimed = {entry.id for entry in self._entries.values()}
        entry = ModelEntry(
            path=str(resolved),
            id=_allocate_model_id(resolved, claimed),
            source=source,
            repo=repo,
        )
        self._entries[str(resolved)] = entry
        return entry

    def forget(self, path: str | Path) -> ModelEntry:
        """Remove an entry from the library, leaving the files alone.

        Deleting weights is a separate, explicit act: forgetting a model that
        happens to be on an unmounted drive must not be a way to lose it.
        """
        resolved = str(Path(path).expanduser())
        entry = self._entries.pop(resolved, None)
        if entry is None:
            raise ConfigError(f"{resolved} is not in the model library")
        return entry

    def get(self, path: str | Path) -> ModelEntry | None:
        return self._entries.get(str(Path(path).expanduser()))

    def touch(self, path: str | Path) -> None:
        entry = self.get(path)
        if entry is not None:
            entry.last_used_at = time.time()

    # -- listing -------------------------------------------------------------

    def discover(self) -> list[ModelEntry]:
        """Find model directories under the configured roots.

        Additive only. A root on an absent volume simply yields nothing, and
        that must never be mistaken for "the models there are gone".
        """
        found: list[ModelEntry] = []
        for root in self._roots:
            directory = Path(root).expanduser()
            if not volume_for(directory).mounted or not directory.is_dir():
                continue
            for candidate in sorted(directory.iterdir()):
                if not candidate.is_dir() or str(candidate) in self._entries:
                    continue
                if not (candidate / "config.json").is_file():
                    continue
                if inspect_model(candidate).verdict is Verdict.UNSUPPORTED:
                    continue
                claimed = {entry.id for entry in self._entries.values()}
                entry = ModelEntry(
                    path=str(candidate),
                    id=_allocate_model_id(candidate, claimed),
                    source="discovered",
                )
                self._entries[str(candidate)] = entry
                found.append(entry)
        return found

    def report(self) -> list[ModelReport]:
        """Every known model with its state as of now."""
        return [evaluate(entry) for entry in sorted(self._entries.values(), key=lambda e: e.path)]


# -- persistence -------------------------------------------------------------


def _migrate_registry(data: dict[str, Any], version: int, expected: int) -> dict[str, Any]:
    """Version 1 had no immutable entry id; derive the ids it already implied."""
    if version != 1 or expected != 2:
        raise ConfigError(
            f"{models_path()} has unsupported schema version {version!r}; expected {expected}"
        )
    claimed: set[str] = set()
    migrated = dict(data)
    models: list[Any] = []
    for raw in data.get("models") or []:
        if not isinstance(raw, dict):
            models.append(raw)
            continue
        entry = dict(raw)
        path = entry.get("path")
        if isinstance(path, str):
            entry["id"] = _allocate_model_id(path, claimed)
            claimed.add(entry["id"])
        models.append(entry)
    migrated["models"] = models
    migrated["version"] = expected
    return migrated


def load_registry() -> ModelRegistry:
    """Read the library, writing back any identity that had to be allocated.

    An id that is only ever derived is not a stored fact: it is a function of
    this file's contents and order, recomputed by every reader. That happens to
    agree with itself today because every QCS write persists the ids it derived
    first -- but it leaves the document at the older schema version indefinitely,
    keeps the derivation on the read path, and makes identity depend on a
    property of the file rather than on a value in it. The first read after an
    upgrade is therefore also a write.
    """
    # The migrator returns a current-schema document, so the result cannot be
    # told apart from a file that was already current. Recorded here rather than
    # inferred, because "did this need writing back" is the whole question.
    upgraded = False

    def migrator(data: dict[str, Any], version: int, expected: int) -> dict[str, Any]:
        nonlocal upgraded
        upgraded = True
        return _migrate_registry(data, version, expected)

    data = read_json(models_path(), expected_version=MODELS_VERSION, migrator=migrator)
    if data is None:
        return ModelRegistry()

    persisted = [
        raw.get("id") if isinstance(raw, dict) else None for raw in (data.get("models") or [])
    ]

    entries: list[ModelEntry] = []
    known = set(ModelEntry.__dataclass_fields__)
    for raw in data.get("models") or []:
        if not isinstance(raw, dict):
            raise ConfigError("each model entry must be an object")
        unknown = set(raw) - known
        if unknown:
            raise ConfigError(
                f"model entry has unknown field(s): {', '.join(sorted(unknown))}"
            )
        entries.append(ModelEntry(**raw))

    roots = data.get("roots")
    registry = ModelRegistry(entries=entries, roots=roots if isinstance(roots, list) else None)

    if upgraded or persisted != [entry.id for entry in registry._entries.values()]:
        try:
            save_registry(registry)
        except OSError as exc:
            # The ids are correct for this process either way. Refusing to read
            # a readable library because it could not be written would be the
            # worse failure; the next successful write settles it.
            logger.warning("Could not persist migrated model ids: %s", exc)
    return registry


def save_registry(registry: ModelRegistry) -> None:
    write_json(
        models_path(),
        {
            "version": MODELS_VERSION,
            "roots": registry.roots,
            "models": [asdict(entry) for entry in registry._entries.values()],
        },
    )
