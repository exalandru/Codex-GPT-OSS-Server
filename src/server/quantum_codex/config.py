"""Configuration, profiles and runtime state on disk.

Four separate files rather than one settings blob (cahier 39), because they have
genuinely different owners and lifetimes:

``settings.json``   user preferences, edited rarely, survives everything
``profiles.json``   named server configurations, edited deliberately
``runtime.json``    written by a running server, meaningless once it exits
``benchmarks.json`` measurement history, append-mostly (later slice)

Merging them would mean a crash while recording a benchmark could corrupt the
model paths, and it would make "reset my settings" impossible to do without also
discarding results.

Every file carries a schema ``version``. Reading an unknown future version fails
loudly rather than silently discarding the fields it does not recognise —
dropping a field that a newer build wrote is how a configuration quietly loses
what the user set.

Writes are atomic: a temporary file in the same directory, then ``os.replace``.
An interrupted write must never leave a half-written profile behind, because the
next start would read it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from . import CLI_NAME
from .inference.prompt_cache import DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES

logger = logging.getLogger(__name__)

BUNDLE_ID = "com.exalandru.qcs"

#: The identifier used before the product was named. Read once, to move a
#: developer's existing configuration into the new root; never written.
PREVIOUS_BUNDLE_ID = "com.exalandru.quantum-codex"

SETTINGS_VERSION = 1
PROFILES_VERSION = 1
RUNTIME_VERSION = 1

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8123
DEFAULT_CONTEXT_LENGTH = 131072
# GPT-OSS reasons at length before answering, and a Codex turn spends part of
# its budget on the analysis channel. 8192 was too small for real sessions.
# This is a *default* for new profiles only: an existing profile keeps whatever
# the user set (see `Profiles.load`).
DEFAULT_MAX_OUTPUT_TOKENS = 32768

#: How long a model stays resident with no inference activity, in minutes.
#:
#: Lives here rather than with the supervisor that enforces it because config is
#: this package's leaf: the supervisor, the profile schema and the desktop form
#: all read this one value, and a default declared in each of them is a default
#: that eventually disagrees with itself.
DEFAULT_IDLE_TIMEOUT_MINUTES = 10

#: Upper bound offered by the configuration form. A day is already well past the
#: point where holding weights for a session that ended is deliberate.
MAX_IDLE_TIMEOUT_MINUTES = 1440


class ConfigError(Exception):
    """A configuration file exists but cannot be used as written."""


# -- locations ---------------------------------------------------------------


def app_support_dir() -> Path:
    """Where this application keeps its state.

    ``QUANTUM_CODEX_HOME`` overrides it, which is what makes the whole
    configuration surface testable without touching the user's real files.
    """
    override = os.environ.get("QUANTUM_CODEX_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / BUNDLE_ID


def settings_path() -> Path:
    return app_support_dir() / "settings.json"


def profiles_path() -> Path:
    return app_support_dir() / "profiles.json"


def runtime_path() -> Path:
    return app_support_dir() / "runtime.json"


def logs_dir() -> Path:
    return app_support_dir() / "logs"


# -- atomic I/O ---------------------------------------------------------------


def read_json(path: Path, *, expected_version: int) -> dict[str, Any] | None:
    """Read a versioned file, or ``None`` when it does not exist yet."""
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")

    version = data.get("version")
    if version == expected_version:
        return data
    if isinstance(version, int) and version > expected_version:
        # A newer build wrote this. Reading it with older rules would silently
        # drop whatever it added, so refuse rather than quietly downgrade.
        raise ConfigError(
            f"{path} was written by a newer version (schema {version}, this build "
            f"understands {expected_version}). Upgrade, or move the file aside."
        )
    return migrate(data, path=path, expected_version=expected_version)


def migrate(data: dict[str, Any], *, path: Path, expected_version: int) -> dict[str, Any]:
    """Bring an older file up to the current schema.

    There is nothing to migrate yet -- version 1 is the first. The seam exists
    now so the first real migration is a change to this function rather than a
    change to how every reader works.
    """
    version = data.get("version")
    raise ConfigError(
        f"{path} has unsupported schema version {version!r}; expected {expected_version}"
    )


def write_json(path: Path, payload: dict[str, Any], *, private: bool = False) -> None:
    """Write atomically, so an interrupted write leaves the old file intact.

    ``private`` restricts the file to the owner. Used for anything holding a
    credential -- the management token in particular, which is what stands
    between another local process and this server's control surface.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w") as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=False)
            tmp.write("\n")
            tmp.flush()
            # The rename is atomic, but it only orders the directory entry.
            # Without the fsync the new contents may still be in flight when the
            # machine loses power, leaving a name pointing at nothing useful.
            os.fsync(tmp.fileno())
        # mkstemp already creates the file owner-only; widen it for the files
        # that are not secret, so a user can read their own configuration.
        os.chmod(temp_name, 0o600 if private else 0o644)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# -- server profiles ----------------------------------------------------------


@dataclass
class ServerProfile:
    """One named server configuration (cahier 26).

    Separating capability from default from override matters here: this is the
    *server default*. A request may still narrow it, and the model's own
    capability bounds both.
    """

    name: str
    # Which model to preload, as a *model id* rather than a filesystem path. A
    # path belongs to the model library: it changes when a volume is remounted
    # or a directory renamed, and a profile holding one would quietly stop
    # matching anything.
    #
    # `None` and `""` both mean "load nothing until a request asks" -- the form
    # sends `None` for its explicit "None — load on demand" choice, and older
    # files hold `""`. Both are valid on disk and normalised by
    # `default_model`; nothing else may assume this is a string.
    #
    # Profiles written before this held a path; `load_profiles` migrates them.
    model: str | None = None
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    cache_max_entries: int = DEFAULT_MAX_ENTRIES
    cache_max_bytes: int = DEFAULT_MAX_BYTES
    log_level: str = "INFO"

    # How long weights stay resident with nothing using them. A server-wide
    # lifecycle rule, not a model setting: it describes how this daemon treats
    # idle residency, and the answer does not change because the 120B is loaded
    # rather than the 20B. `0` never unloads.
    #
    # Absent from an older profiles.json, which is why it carries a default: a
    # profile written before this existed gets the product default rather than
    # being refused or silently disabled.
    model_idle_timeout_minutes: int = DEFAULT_IDLE_TIMEOUT_MINUTES

    #: Model-specific values found in a profile written by an older build, which
    #: could not be attributed to a model because the profile named none.
    #:
    #: Held rather than discarded: they are the user's settings, and guessing
    #: which model they belong to is exactly the mistake worth avoiding. Kept out
    #: of the schema so nothing renders or edits them; `migrate_model_settings`
    #: moves them the moment the profile names a model.
    legacy_model_settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        validate_profile_name(self.name)
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"profile {self.name!r} has an invalid port {self.port}")

    @property
    def default_model(self) -> str | None:
        """The model id this profile preloads, or ``None`` for load-on-demand.

        The one place the two spellings of "nothing" become one answer. Every
        reader goes through this rather than touching ``model`` directly, which
        is what a `TypeError: argument of type 'NoneType' is not iterable`
        taught: a nullable field with several readers needs a single accessor,
        not a convention.
        """
        model = self.model
        if not isinstance(model, str):
            return None
        return model.strip() or None



#: A profile name is also a CLI argument and a JSON object key, so it is kept to
#: characters that survive both without quoting or escaping. Rejecting early is
#: what stops a name that cannot be typed at a terminal from being created in a
#: form.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


def validate_profile_name(name: str) -> str:
    """Return the cleaned name, or raise with what is actually wrong.

    One message per distinct problem: "invalid name" leaves a user guessing
    which of several rules they broke.
    """
    clean = name.strip()
    if not clean:
        raise ConfigError("a profile needs a name")
    if len(clean) > 64:
        raise ConfigError(f"profile name is {len(clean)} characters; the limit is 64")
    if not _PROFILE_NAME.match(clean):
        raise ConfigError(
            f"profile name {clean!r} may use letters, digits, spaces, dot, dash and "
            f"underscore, and must start with a letter or digit"
        )
    return clean


#: Settings that used to live on a profile and now belong to a model.
#:
#: Named here rather than imported from the schema so reading a profile cannot
#: depend on the schema module, and so this list stays fixed at the set that
#: actually existed on disk -- a later addition to `MODEL_FIELDS` was never in an
#: old profile and must not be treated as legacy.
LEGACY_MODEL_FIELDS = (
    "served_model_name",
    "context_length",
    "max_output_tokens",
    "reasoning_effort",
    "temperature",
    "top_p",
)


def _looks_like_a_path(model: object) -> bool:
    """Whether a stored model value is a filesystem location rather than an id.

    A separator is the discriminator: model ids are single tokens like
    ``gpt-oss-20b``, and every path this ever stored was absolute.

    Takes ``object`` deliberately. This runs over whatever a file happens to
    contain, which includes ``null`` for "load on demand" and, in a hand-edited
    file, anything at all. Assuming ``str`` here is what made a saved
    "None — load on demand" render every profile unreadable.
    """
    return isinstance(model, str) and "/" in model


@dataclass
class Profiles:
    """The profile collection, with the one selected by default."""

    profiles: dict[str, ServerProfile] = field(default_factory=dict)
    default: str | None = None
    #: Set when loading rewrote something. The caller may save to make the
    #: repair durable; nothing depends on it having happened.
    migrated: bool = False

    def get(self, name: str) -> ServerProfile:
        profile = self.profiles.get(name)
        if profile is None:
            known = ", ".join(sorted(self.profiles)) or "none"
            raise ConfigError(f"no profile named {name!r}. Known profiles: {known}")
        return profile

    def resolve(self, name: str | None) -> ServerProfile:
        """The profile to use when the caller named one, or the default."""
        if name is not None:
            return self.get(name)
        if self.default is not None:
            return self.get(self.default)
        if len(self.profiles) == 1:
            return next(iter(self.profiles.values()))
        raise ConfigError(
            "no profile selected and no default set. "
            f"Use --profile, or set a default with `{CLI_NAME} profiles default <name>`."
        )

    def put(self, profile: ServerProfile) -> None:
        profile.validate()
        self.profiles[profile.name] = profile
        if self.default is None:
            self.default = profile.name

    def remove(self, name: str) -> None:
        self.get(name)
        del self.profiles[name]
        if self.default == name:
            self.default = next(iter(self.profiles), None)

    # -- lifecycle ------------------------------------------------------------
    #
    # Creating, copying and renaming all live here rather than in a caller,
    # because each one has to enforce the same two rules -- a valid name, and no
    # collision -- and a second implementation would eventually enforce only
    # one of them. The CLI and the desktop both go through these.

    def create(self, name: str) -> ServerProfile:
        """A new profile, from the schema's own defaults.

        Defaults come from :class:`ServerProfile`'s field defaults, which are
        the same values the schema publishes. Nothing is invented here, so a
        profile created from the GUI and one created from the terminal are
        identical.
        """
        clean = validate_profile_name(name)
        if clean in self.profiles:
            raise ConfigError(f"a profile named {clean!r} already exists")
        profile = ServerProfile(name=clean)
        self.put(profile)
        return profile

    def duplicate(self, source: str, name: str) -> ServerProfile:
        """Copy an existing profile under a new name.

        Copied with :func:`dataclasses.replace`, so a field added later is
        carried over without anyone remembering to update a copy routine.
        """
        clean = validate_profile_name(name)
        original = self.get(source)
        if clean in self.profiles:
            raise ConfigError(f"a profile named {clean!r} already exists")
        copy = replace(original, name=clean)
        self.put(copy)
        return copy

    def rename(self, old: str, name: str) -> ServerProfile:
        """Rename in place, preserving default-selection and ordering.

        Rebuilt rather than mutated so the renamed profile keeps its position:
        a rename that reorders the list looks, to a user, like something else
        changed too.
        """
        clean = validate_profile_name(name)
        profile = self.get(old)
        if clean == old:
            return profile
        if clean in self.profiles:
            raise ConfigError(f"a profile named {clean!r} already exists")

        renamed = replace(profile, name=clean)
        renamed.validate()
        self.profiles = {
            (clean if key == old else key): (renamed if key == old else value)
            for key, value in self.profiles.items()
        }
        if self.default == old:
            self.default = clean
        return renamed


def load_profiles() -> Profiles:
    data = read_json(profiles_path(), expected_version=PROFILES_VERSION)
    if data is None:
        return Profiles()

    profiles: dict[str, ServerProfile] = {}
    migrated = False
    for name, raw in (data.get("profiles") or {}).items():
        if not isinstance(raw, dict):
            raise ConfigError(f"profile {name!r} is not an object")
        # Model-specific settings written by an older build, lifted out first:
        # the profile no longer has anywhere to put them, and they are known
        # rather than unknown -- rejecting them as typos would refuse to open a
        # configuration this build is supposed to migrate.
        legacy = {key: raw.pop(key) for key in LEGACY_MODEL_FIELDS if key in raw}

        known = {f for f in ServerProfile.__dataclass_fields__ if f != "name"}
        unknown = set(raw) - known
        if unknown:
            # Fail closed here too: an unrecognised key usually means a typo in a
            # hand-edited file, and ignoring it would silently not apply what the
            # user thought they had configured.
            raise ConfigError(
                f"profile {name!r} has unknown field(s): {', '.join(sorted(unknown))}"
            )
        profile = ServerProfile(name=name, **raw)
        if legacy:
            profile.legacy_model_settings = legacy
            migrated = True
        if _looks_like_a_path(profile.model):  # noqa: SIM102 - see the guard's docstring
            # Written by a build that stored the weights' location. Converted to
            # the model id in memory and rewritten on the next save, so the file
            # is repaired without the user editing JSON.
            from .models import slug_for

            original = profile.model
            profile.model = slug_for(Path(original).name)
            migrated = True
            logger.info(
                "profile %r recorded a model path (%s); using the model id %r instead",
                name,
                original,
                profile.model,
            )
        profiles[name] = profile

    result = Profiles(profiles=profiles, default=data.get("default"))
    result.migrated = migrated
    for profile in profiles.values():
        profile.validate()
    if result.default is not None and result.default not in profiles:
        raise ConfigError(f"default profile {result.default!r} does not exist")
    return result


def save_profiles(profiles: Profiles) -> None:
    payload = {
        "version": PROFILES_VERSION,
        "default": profiles.default,
        "profiles": {
            name: {k: v for k, v in asdict(profile).items() if k != "name"}
            for name, profile in profiles.profiles.items()
        },
    }
    write_json(profiles_path(), payload)


# -- runtime state ------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeState:
    """What a running server publishes so the CLI and GUI can find it (D1).

    Holds the management token, so the file is owner-only. It describes a
    *process*: a stale file left by a crash is normal, and every reader must
    treat it as a hint to be verified rather than as truth.
    """

    pid: int
    host: str
    port: int
    # Optional: a daemon holding no weights is normal, and which model is
    # resident changes during the process's life. Anything that needs the
    # current model asks the management plane, which knows; this file records
    # only how to reach the server.
    model: str | None
    management_token: str
    started_at: float

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """Whether the recorded process still exists.

        Existence only — the pid may have been reused. Callers confirm by
        actually talking to the management endpoint.
        """
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def write_runtime_state(state: RuntimeState) -> None:
    write_json(
        runtime_path(),
        {"version": RUNTIME_VERSION, **asdict(state)},
        private=True,
    )


def load_runtime_state() -> RuntimeState | None:
    data = read_json(runtime_path(), expected_version=RUNTIME_VERSION)
    if data is None:
        return None
    fields = {f: data.get(f) for f in RuntimeState.__dataclass_fields__}
    # `model` may legitimately be absent. Everything else is how to reach and
    # authenticate against the server, and a file missing any of that cannot be
    # acted on.
    required = {name: value for name, value in fields.items() if name != "model"}
    if any(value is None for value in required.values()):
        raise ConfigError(f"{runtime_path()} is incomplete; remove it and restart the server")
    return RuntimeState(**fields)


def clear_runtime_state() -> None:
    runtime_path().unlink(missing_ok=True)


# -- application settings -----------------------------------------------------
#
# Preferences that belong to the installation rather than to a profile or a
# model. There is one so far: where downloads land.


@dataclass
class Settings:
    """Application-wide preferences."""

    #: Where downloaded models are written. ``None`` means the app-owned
    #: directory, which is what a clean install uses.
    #:
    #: Not a profile setting and not a model setting: a user chooses one disk
    #: for weights, and it stays chosen across every profile and every model.
    #: Stored as text so an external volume that is currently unmounted is still
    #: remembered rather than silently replaced.
    download_root: str | None = None


def load_settings() -> Settings:
    data = read_json(settings_path(), expected_version=SETTINGS_VERSION)
    if data is None:
        return Settings()
    root = data.get("download_root")
    return Settings(download_root=root if isinstance(root, str) and root.strip() else None)


def save_settings(settings: Settings) -> None:
    write_json(
        settings_path(),
        {"version": SETTINGS_VERSION, "download_root": settings.download_root},
    )


# -- moving to the named bundle identifier ------------------------------------

#: User-owned state worth carrying from the pre-rename directory.
#:
#: Deliberately a short list of *durable configuration*. Everything absent is
#: absent on purpose:
#:
#: ``runtime.json``  describes a process that is not running any more, and
#:                   adopting it would point the app at a dead endpoint.
#: ``logs/``         a record of a different installation.
#: ``env/`` ``python/`` ``server/`` ``uv-cache/``
#:                   the managed runtime. Its scripts carry absolute shebangs
#:                   pointing at the old directory, so a copy would produce an
#:                   environment that looks installed and cannot execute. It is
#:                   rebuilt from bundled resources at the new location.
MIGRATED_FILES = (
    "profiles.json",       # server profiles
    "model-settings.json", # per-model overrides
    "models.json",         # the model library and its scan roots
    "settings.json",       # application preferences, incl. the download root
)

_migration_checked = False


def migrate_app_support_root() -> list[str]:
    """Carry configuration from the pre-rename directory, once.

    Runs only when the new root holds none of these files: a location that has
    been used is never overwritten, so this cannot undo work done after the
    rename. Copies rather than moves, leaving the old directory intact for a
    user who wants to go back to an older build.

    Model files themselves are untouched -- they live wherever the user put
    them, and the library records them by path, so they remain known and usable.
    """
    global _migration_checked
    if _migration_checked or os.environ.get("QUANTUM_CODEX_HOME"):
        # An explicit home is a test or a deliberate override; migrating into it
        # would import state its caller did not ask for.
        return []
    _migration_checked = True

    new = app_support_dir()
    old = new.parent / PREVIOUS_BUNDLE_ID
    if not old.is_dir() or old == new:
        return []
    if any((new / name).exists() for name in MIGRATED_FILES):
        return []

    moved: list[str] = []
    new.mkdir(parents=True, exist_ok=True)
    for name in MIGRATED_FILES:
        source = old / name
        if not source.is_file():
            continue
        try:
            shutil.copy2(source, new / name)
            moved.append(name)
        except OSError as exc:  # noqa: PERF203 - one message per file that failed
            logger.warning("could not carry %s over from %s: %s", name, old, exc)

    if moved:
        logger.info(
            "carried %s from %s into %s; the managed runtime is rebuilt rather than copied",
            ", ".join(moved),
            old,
            new,
        )
    return moved
