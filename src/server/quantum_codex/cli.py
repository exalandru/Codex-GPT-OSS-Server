"""`quantum-codex-server` command line entry point.

Everything essential must be reachable from a terminal with no GUI involved
(cahier 40). The desktop app is a control plane over these same concepts, never
a second implementation of them: anything it can do, this can do.

Installed under two names, `quantum-codex-server` (canonical) and `qcs`
(alias), which are the same entry point. Help and error text echoes whichever
one was typed, so copy-pasting a suggested command always works.

Heavy imports (mlx, mlx_lm, openai_harmony, the app module) stay inside the
subcommand functions. Importing this module must remain fast enough that
`quantum-codex-server --help` is instant and that `doctor` can report a broken
environment rather than dying while importing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from . import CLI_ALIAS, CLI_NAME, __version__
from .config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_HOST,
    DEFAULT_IDLE_TIMEOUT_MINUTES,
    DEFAULT_PORT,
    ConfigError,
    ServerProfile,
    load_profiles,
    load_runtime_state,
    migrate_app_support_root,
    save_profiles,
)
from .inference.prompt_cache import DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES


def invoked_as() -> str:
    """The name the user actually typed, for help and suggested commands.

    Both installed names run this file, so a hard-coded ``prog`` would print
    ``quantum-codex-server …`` at someone who typed ``qcs`` — a command they can
    copy, run, and have work, but which is not the one they are using. Falls back
    to the canonical name when argv carries something unhelpful, which is what
    happens under pytest and under ``python -m``.
    """
    name = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""
    return name if name in {CLI_NAME, CLI_ALIAS} else CLI_NAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=invoked_as(),
        description="Local Codex + GPT-OSS + Harmony server on MLX (Apple Silicon).",
    )
    parser.add_argument("--version", action="version", version=f"{CLI_NAME} {__version__}")

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    # -- serve ---------------------------------------------------------------

    serve = subcommands.add_parser("serve", help="Run the inference server")
    serve.add_argument(
        "--profile", help="Named profile to serve. Without it, --model is required"
    )
    serve.add_argument(
        "--model",
        help="Path to a GPT-OSS MLX model directory, or a Hugging Face repository id. "
        "Overrides the profile's model",
    )
    serve.add_argument(
        "--served-model-name",
        help="Stable model id reported to clients. Defaults to the model directory name, "
        "so clients never see a filesystem path",
    )
    # Loopback by default (cahier 42). A LAN bind is a deliberate act, not
    # something that happens because a default was left alone.
    serve.add_argument("--host", help=f"Bind address (default: {DEFAULT_HOST})")
    serve.add_argument("--port", type=int, help=f"Bind port (default: {DEFAULT_PORT})")
    serve.add_argument(
        "--context-length",
        type=int,
        help=f"Maximum KV cache context length (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    serve.add_argument(
        "--cache-max-entries",
        type=int,
        help=f"Maximum prompt cache sessions (default: {DEFAULT_MAX_ENTRIES}); 0 disables reuse",
    )
    serve.add_argument(
        "--cache-max-bytes",
        type=int,
        help=f"Prompt cache byte budget (default: {DEFAULT_MAX_BYTES // 1024**3} GiB)",
    )
    serve.add_argument(
        "--model-idle-timeout-minutes",
        type=int,
        help=(
            f"Unload the model after this many minutes without inference activity "
            f"(default: {DEFAULT_IDLE_TIMEOUT_MINUTES}); 0 never unloads"
        ),
    )
    serve.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level"
    )

    # -- inspection ----------------------------------------------------------

    status = subcommands.add_parser("status", help="Report on the running server")
    status.add_argument("--json", action="store_true", help="Emit raw JSON")

    models = subcommands.add_parser("models", help="Inspect models").add_subparsers(
        dest="models_command", metavar="<subcommand>", required=True
    )
    models.add_parser("catalog", help="Supported GPT-OSS models, joined with what is installed")

    storage = models.add_parser("storage", help="Where downloaded models are written")
    storage.add_argument(
        "path", nargs="?", help="New download location. Omit to show the current one"
    )
    storage.add_argument("--json", action="store_true", help="Emit raw JSON")
    models.add_parser("config-schema", help="Describe every per-model setting, as JSON")
    models.add_parser(
        "unload",
        help="Release the resident model, leaving the server running",
    )

    models_config = models.add_parser("config", help="Show or change one model's settings")
    models_config.add_argument("slug", help="Stable model id, e.g. gpt-oss-20b")
    models_config.add_argument(
        "assignments",
        nargs="*",
        metavar="field=value",
        help="Settings to change. An empty value clears the override",
    )
    models_config.add_argument("--json", action="store_true", help="Emit raw JSON")
    models_list = models.add_parser("list", help="List models in the library")
    models_list.add_argument("--json", action="store_true", help="Emit raw JSON")
    models_list.add_argument(
        "--scan", action="store_true", help="Also discover models under the configured roots"
    )

    models_scan = models.add_parser(
        "scan", help="Look for models under the configured roots and register new ones"
    )
    models_scan.add_argument("--json", action="store_true", help="Emit raw JSON")

    models_import = models.add_parser("import", help="Add an existing model directory")
    models_import.add_argument(
        "--expect",
        metavar="SLUG",
        help="Refuse the import unless the directory is this model. Used when locating a "
        "specific catalog model, so a directory is never attached to the wrong entry",
    )
    models_import.add_argument("--json", action="store_true", help="Emit raw JSON")
    models_import.add_argument("path")

    download = models.add_parser("download", help="Fetch a GPT-OSS model from Hugging Face")
    download.add_argument("repo", help="Repository id, e.g. mlx-community/gpt-oss-20b-MXFP4-Q8")
    download.add_argument("--destination", help="Where to put it; defaults to the first root")
    download.add_argument(
        "--watch", action="store_true", help="Follow progress until it finishes"
    )

    models_forget = models.add_parser(
        "forget", help="Remove a model from the library, leaving its files alone"
    )
    models_forget.add_argument("path")

    roots = models.add_parser("roots", help="Where models are looked for").add_subparsers(
        dest="roots_command", metavar="<subcommand>", required=True
    )
    roots.add_parser("list", help="List model roots")
    roots_add = roots.add_parser("add", help="Add a model root")
    roots_add.add_argument("path")
    roots_remove = roots.add_parser("remove", help="Remove a model root")
    roots_remove.add_argument("path")
    inspect_cmd = models.add_parser(
        "inspect", help="Check whether a directory is a GPT-OSS model this server can run"
    )
    inspect_cmd.add_argument("path")
    inspect_cmd.add_argument("--json", action="store_true", help="Emit raw JSON")
    inspect_adapter_cmd = models.add_parser(
        "inspect-adapter",
        help="Check whether a directory is a LoRA adapter this server can apply",
    )
    inspect_adapter_cmd.add_argument("path")
    inspect_adapter_cmd.add_argument("--json", action="store_true", help="Emit raw JSON")

    requests = subcommands.add_parser("requests", help="Recent request diagnostics")
    requests.add_argument("--json", action="store_true", help="Emit raw JSON")
    requests.add_argument("--limit", type=int, default=20, help="How many to show")

    cache = subcommands.add_parser("cache", help="Prompt cache").add_subparsers(
        dest="cache_command", metavar="<subcommand>", required=True
    )
    cache_stats = cache.add_parser("stats", help="Show prompt cache counters")
    cache_stats.add_argument("--json", action="store_true", help="Emit raw JSON")
    cache.add_parser("clear", help="Drop every cached prefix")

    # -- profiles ------------------------------------------------------------

    profiles = subcommands.add_parser("profiles", help="Server profiles").add_subparsers(
        dest="profiles_command", metavar="<subcommand>", required=True
    )
    profiles_list = profiles.add_parser("list", help="List profiles")
    profiles_list.add_argument("--json", action="store_true", help="Emit raw JSON")
    profiles.add_parser("schema", help="Describe every profile setting, as JSON")

    show = profiles.add_parser("show", help="Show one profile")
    show.add_argument("name")
    show.add_argument("--json", action="store_true", help="Emit raw JSON")

    profiles_set = profiles.add_parser("set", help="Change settings on a profile")
    profiles_set.add_argument("name")
    profiles_set.add_argument(
        "assignments",
        nargs="+",
        metavar="field=value",
        help="One or more settings, e.g. reasoning_effort=high port=8124",
    )
    profiles_set.add_argument("--json", action="store_true", help="Emit raw JSON")
    new = profiles.add_parser("new", help="Create a profile from the server defaults")
    new.add_argument("name")
    new.add_argument("--json", action="store_true", help="Emit raw JSON")

    duplicate = profiles.add_parser("duplicate", help="Copy an existing profile")
    duplicate.add_argument("source")
    duplicate.add_argument("name", help="Name for the copy")
    duplicate.add_argument("--json", action="store_true", help="Emit raw JSON")

    rename = profiles.add_parser("rename", help="Rename a profile")
    rename.add_argument("name")
    rename.add_argument("new_name")
    rename.add_argument("--json", action="store_true", help="Emit raw JSON")

    add = profiles.add_parser("add", help="Create or replace a profile")
    add.add_argument("name")
    add.add_argument("--model", default="")
    add.add_argument("--served-model-name")
    add.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    add.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    add.add_argument("--host", default=DEFAULT_HOST)
    add.add_argument("--port", type=int, default=DEFAULT_PORT)
    remove = profiles.add_parser("remove", help="Delete a profile")
    remove.add_argument("name")
    remove.add_argument(
        "--force",
        action="store_true",
        help="Delete even when a server appears to be running on this profile's port",
    )
    default = profiles.add_parser("default", help="Set the default profile")
    default.add_argument("name")

    # -- codex ---------------------------------------------------------------

    codex = subcommands.add_parser("codex", help="Codex integration").add_subparsers(
        dest="codex_command", metavar="<subcommand>", required=True
    )
    launch = codex.add_parser(
        "launch", help="Print the codex command that talks to this server"
    )
    launch.add_argument("--profile", help="Profile whose endpoint and model to use")
    launch.add_argument("--model", help="Model id to put in the configuration")
    launch.add_argument(
        "--config",
        action="store_true",
        help="Emit a ~/.codex/config.toml fragment instead of a one-shot command. "
        "For the Codex CLI's global configuration and the VS Code extension",
    )
    launch.add_argument(
        "--models-json",
        action="store_true",
        help="List the models this server can be pointed at, and their effective "
        "reasoning effort, as JSON. For an interface offering a choice",
    )
    launch.add_argument("prompt", nargs="?", help="Optional prompt to include")

    subcommands.add_parser("doctor", help="Report environment, runtime and model readiness")

    return parser


# -- helpers ------------------------------------------------------------------


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _management_request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    """Call the running server's management plane (D1).

    The runtime file says where the server is and carries its token. It
    describes a process, so a stale file left by a crash is normal: the
    connection attempt is what actually establishes whether a server is there.
    """
    import urllib.error
    import urllib.request

    state = load_runtime_state()
    if state is None:
        raise ConfigError("no server is running (no runtime file). Start one with `serve`.")
    if not state.is_running:
        raise ConfigError(
            f"the runtime file names pid {state.pid}, which is gone. "
            "The server exited without cleaning up; start a new one."
        )

    headers = {"Authorization": f"Bearer {state.management_token}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{state.base_url}{path}", method=method, headers=headers, data=data
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # The server explains refusals; relaying its words is the whole point.
        try:
            message = json.load(exc)["error"]["message"]
        except Exception:  # noqa: BLE001
            message = exc.reason
        raise ConfigError(str(message)) from exc
    except urllib.error.URLError as exc:
        raise ConfigError(
            f"cannot reach the server at {state.base_url}: {exc.reason}. "
            "It may have exited without cleaning up its runtime file."
        ) from exc


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


# -- commands -----------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    from .app import serve
    from .model_settings import load_model_settings, migrate_from_profiles, save_model_settings

    profiles = load_profiles()
    # Model settings written by an older build move to the model they belong to,
    # once, before anything reads them. Nothing is discarded and nothing is
    # guessed; see `migrate_from_profiles`.
    settings = load_model_settings()
    before = dict(settings.overrides)
    migrate_from_profiles(profiles, settings)
    if settings.overrides != before:
        # Written only when something actually moved, so serving a profile does
        # not rewrite two files on every start.
        save_model_settings(settings)
        save_profiles(profiles)

    if args.profile or not args.model:
        profile = profiles.resolve(args.profile)
    else:
        profile = ServerProfile(name="(command line)", model=args.model)

    # Explicit flags win over the profile: the profile is the stored default,
    # the flag is this run's intent.
    model = args.model or profile.default_model
    profile.validate()

    from .app import ServerDefaults

    # `ServerDefaults` is now only the fallback for a model with no override of
    # its own. Reasoning effort, output budget and sampling belong to the model
    # (see `model_settings`), so the profile no longer supplies them -- taking
    # them from here again would let one profile-level value shadow every
    # model's setting, which is the shape this split exists to remove.
    return serve(
        defaults=ServerDefaults(),
        model=model,
        served_model_name=args.served_model_name,
        host=args.host or profile.host,
        port=args.port or profile.port,
        cache_max_entries=(
            args.cache_max_entries
            if args.cache_max_entries is not None
            else profile.cache_max_entries
        ),
        cache_max_bytes=(
            args.cache_max_bytes if args.cache_max_bytes is not None else profile.cache_max_bytes
        ),
        log_level=args.log_level or profile.log_level,
        idle_timeout_minutes=(
            args.model_idle_timeout_minutes
            if args.model_idle_timeout_minutes is not None
            else profile.model_idle_timeout_minutes
        ),
    )


def _cmd_status(args: argparse.Namespace) -> int:
    payload = _management_request("/internal/status")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    server = payload["server"]
    print(f"server      {server['state']}  up {server['uptime_seconds']}s  {server['endpoint']}")

    model = payload.get("model")
    if model:
        print(f"model       {model['served_name']}  {model['quantization']}")
        print(f"            {model['path']}")
        adapter = model.get("adapter")
        if adapter:
            # The applied counts, not the configured path: this line exists to
            # distinguish an adapter that took effect from one that did not.
            print(
                f"adapter     {adapter['fine_tune_type']}, "
                f"{adapter['applied_tensors']}/{adapter['adapter_tensors']} tensors applied"
            )
            print(f"            {adapter['path']}")

    lifecycle = payload.get("lifecycle") or {}
    # Printed whether or not a model is resident: "no model loaded, released
    # because it was idle" and "no model loaded, none ever asked for" are
    # different situations, and only this line tells them apart.
    print(f"residency   {lifecycle.get('state', 'unknown')}{_residency_detail(lifecycle)}")

    caps = payload.get("capabilities")
    if caps:
        print(
            f"context     {caps['context_window']} tokens "
            f"(effective {caps['effective_context_window']})"
        )
        print(
            f"reasoning   {', '.join(caps['reasoning_efforts'])} "
            f"(default {caps['default_reasoning_effort']})"
        )
        print(
            f"tools       {'yes' if caps['supports_tools'] else 'no'}, "
            f"parallel {'yes' if caps['supports_parallel_tool_calls'] else 'no'}"
        )

    inference = payload["inference"]
    print(
        f"inference   {inference['active_requests']} active, "
        f"{inference['queued_requests']} queued"
    )

    cache = payload["prompt_cache"]
    print(
        f"cache       {cache['entries']} sessions, {_human_bytes(cache['bytes'])}, "
        f"hit ratio {cache['hit_ratio']:.0%} ({cache['hits']}/{cache['hits'] + cache['misses']})"
    )
    print(
        f"            {cache['cached_tokens_total']} tokens reused, "
        f"{cache['evaluated_tokens_total']} evaluated"
    )
    return 0


def _cmd_models_config_schema(_args: argparse.Namespace) -> int:
    from .profile_schema import model_schema

    print(json.dumps(model_schema(), indent=2))
    return 0


def _cmd_models_config(args: argparse.Namespace) -> int:
    """Show or change one model's settings, keyed by its stable id.

    Coercion and validation are the schema's, exactly as for a profile: the
    interface sends `field=value` strings and the server decides what they mean.
    """
    import copy

    from .inspect_adapter import describes_the_same_model, inspect_adapter
    from .library import volume_for
    from .library.catalog import defaults_for, display_name_for
    from .library.registry import load_registry
    from .model_settings import ModelSettings, load_model_settings, save_model_settings
    from .models import resolve_served_catalogue, resolved_model_names, slug_for
    from .profile_schema import (
        MODEL_FIELDS,
        coerce_model,
        model_field_names,
        validate_model,
    )

    settings = load_model_settings()
    reports = load_registry().report()
    exact_matches = []
    alias_matches = []
    for report in reports:
        names = resolved_model_names(report, overrides=settings.overrides)
        if args.slug == names.library_id:
            exact_matches.append((report, names))
        elif args.slug in {names.served_name, slug_for(report.entry.name)}:
            alias_matches.append((report, names))
    matches = exact_matches or alias_matches
    if not matches:
        known = ", ".join(getattr(report.entry, "id", "") for report in reports) or "none"
        raise ConfigError(f"no model with id or served name {args.slug!r}. Known model ids: {known}")
    if len(matches) > 1:
        raise ConfigError(
            f"model name {args.slug!r} is ambiguous; configure by immutable library id"
        )
    report, names = matches[0]
    model_id = names.library_id

    if args.assignments:
        known = model_field_names()
        values: dict[str, object] = {}
        for assignment in args.assignments:
            if "=" not in assignment:
                raise ConfigError(f"expected field=value, got {assignment!r}")
            name, _, raw = assignment.partition("=")
            name = name.strip()
            if name not in known:
                raise ConfigError(
                    f"unknown model setting {name!r}. Known: {', '.join(sorted(known))}"
                )
            values[name] = coerce_model(name, raw) if raw.strip() else None

        # Validated together, so every problem is reported at once rather than
        # one per submission.
        problems = validate_model({k: v for k, v in values.items() if v is not None})
        if problems:
            raise ConfigError("; ".join(f"{p.field}: {p.message}" for p in problems))

        # Refuse what is provably wrong; store and report what is merely
        # currently unreachable.
        #
        # A directory the user just chose is refused here, the way an
        # unreadable model directory is refused at import, because saying why
        # is the whole value of validating at the boundary. But an adapter on
        # an unmounted volume is a normal situation -- the drive is on the desk
        # -- and refusing it would make the setting not only unstorable but
        # *unclearable*, since clearing arrives through this same path.
        #
        # Only the value being written now is inspected. Re-inspecting a stored
        # adapter would let one model's unplugged drive block an unrelated edit.
        chosen_adapter = values.get("adapter_path")
        if isinstance(chosen_adapter, str) and chosen_adapter:
            adapter = inspect_adapter(chosen_adapter)
            if not adapter.usable and volume_for(chosen_adapter).mounted:
                raise ConfigError(f"{chosen_adapter} cannot be used: {adapter.reasons[0]}")
            if not describes_the_same_model(adapter, report.entry.path):
                # A label, so it is a note and not a refusal. The authority on
                # whether an adapter fits is the engine's witness, which
                # compares tensor names against the weights themselves.
                print(
                    f"note: this adapter records {adapter.trained_against!r} as the model "
                    f"it was trained against, which is not {report.entry.name!r}. If they "
                    "really differ, the load will refuse it.",
                    file=sys.stderr,
                )

        candidate = ModelSettings(overrides=copy.deepcopy(settings.overrides))
        candidate.set(model_id, values)

        # The same construction the daemon uses is the oracle: a save that would
        # make two models answer to one name is refused while the user is still
        # looking at the form.
        #
        # What is compared is *new* ambiguity, not any ambiguity. A library can
        # already contain a contested name -- importing a second copy of a model
        # is enough, and no configuration was involved -- and refusing every
        # subsequent edit would leave the user unable to change anything,
        # including the served name that would resolve it.
        before = {
            problem.served_name
            for problem in resolve_served_catalogue(
                reports, overrides=settings.overrides
            ).problems
        }
        introduced = [
            problem
            for problem in resolve_served_catalogue(
                reports, overrides=candidate.overrides
            ).problems
            if problem.served_name not in before
        ]
        if introduced:
            raise ConfigError("; ".join(problem.message for problem in introduced))

        settings = candidate
        save_model_settings(settings)

    current = settings.for_model(model_id)

    # Three separate things, kept separate:
    #
    #   effective  what the model will actually use
    #   overrides  what this user explicitly chose
    #   defaults   what it would fall back to
    #
    # A form needs `effective` to show something useful on first open, and needs
    # to know which of those values are merely inherited so it does not save
    # them back as overrides. Collapsing them would turn opening a form into a
    # silent write of every default.
    defaults = {
        field.name: field.default
        for field in MODEL_FIELDS
        if field.default is not None
    }
    defaults.update(defaults_for(names.catalog_slug))
    defaults.update(
        {
            "display_name": display_name_for(names.catalog_slug) or report.entry.name,
            "served_model_name": defaults.get("served_model_name") or names.catalog_slug,
            "context_length": report.context_length or defaults.get("context_length"),
        }
    )
    effective = {**defaults, **current}

    # `adapter_path` is a path, so what a form needs to show beside it is what
    # is *at* that path right now -- "set, but the volume is not mounted" is the
    # state a stored string alone cannot express. A sibling key rather than part
    # of `effective`, because it is an observation about the world and not a
    # setting anyone chose.
    configured_adapter = effective.get("adapter_path")
    adapter_state = (
        inspect_adapter(configured_adapter).as_dict()
        if isinstance(configured_adapter, str) and configured_adapter
        else None
    )

    if getattr(args, "json", False):
        print(
            json.dumps(
                {
                    "model": model_id,
                    "settings": current,
                    "defaults": defaults,
                    "effective": effective,
                    "inherited": sorted(set(effective) - set(current)),
                    "adapter": adapter_state,
                },
                indent=2,
            )
        )
        return 0

    if not current:
        print(f"{model_id}: no overrides; using the server defaults")
        return 0
    for name in sorted(current):
        print(f"  {name:<20} {current[name]}")
        # Directly under the path it describes, rather than at the end: printed
        # last it would read as a continuation of whichever setting happened to
        # sort there.
        if name == "adapter_path" and adapter_state is not None:
            print(f"  {'':<20} {adapter_state['verdict'].lower()}: {adapter_state['reasons'][0]}")
    return 0


def _cmd_models_storage(args: argparse.Namespace) -> int:
    """Show or change the download location.

    Application-wide. Changing it affects future downloads only: models already
    on disk are left exactly where they are, and their library entries stay
    valid because a model's identity has never been its path.
    """
    from pathlib import Path

    from .config import load_settings, save_settings
    from .library import volume_for
    from .library.manager import download_root

    if args.path:
        chosen = Path(args.path).expanduser()
        if chosen.exists() and not chosen.is_dir():
            raise ConfigError(f"{chosen} is not a directory")
        settings = load_settings()
        settings.download_root = str(chosen)
        save_settings(settings)

    root = download_root()
    volume = volume_for(root)
    payload = {
        "download_root": str(root),
        "available": root.exists() or volume.mounted,
        "volume": {"name": volume.name, "mounted": volume.mounted, "external": volume.is_external},
        "free_bytes": volume.free_bytes,
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print(f"downloads go to {root}")
    if not payload["available"]:
        print(f"  volume {volume.name!r} is not mounted")
    return 0


def _cmd_models_catalog(args: argparse.Namespace) -> int:
    """Supported models joined with what is installed.

    One list, so the interface never has to decide which supported model an
    installed directory corresponds to.
    """
    from .library.catalog import merge
    from .library.registry import load_registry
    from .model_settings import load_model_settings

    print(
        json.dumps(
            {
                "models": merge(
                    load_registry().report(), overrides=load_model_settings().overrides
                )
            },
            indent=2,
        )
    )
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    from .library import load_registry, save_registry

    registry = load_registry()
    if getattr(args, "scan", False) and registry.discover():
        save_registry(registry)

    reports = registry.report()

    if getattr(args, "json", False):
        print(json.dumps({"roots": registry.roots, "models": [r.as_dict() for r in reports]}, indent=2))
        return 0

    if not reports:
        print(f"The library is empty. Add a model with `{invoked_as()} models import <path>`,")
        print(f"or point a root at one: roots are {', '.join(registry.roots)}")
        return 0

    for report in reports:
        print(f"{report.state.value:<17} {report.entry.name}")
        print(f"    {report.entry.path}")
        if report.state.usable:
            print(
                f"    {report.quantization}, {report.context_length} ctx, "
                f"{report.layers} layers, {_human_bytes(report.disk_bytes)} on disk"
            )
        else:
            print(f"    {report.detail}")
        if report.volume.is_external:
            free = _human_bytes(report.volume.free_bytes) if report.volume.free_bytes else "?"
            state = "mounted" if report.volume.mounted else "not mounted"
            print(f"    volume {report.volume.name}: {state}, {free} free")
    # Non-zero when nothing is usable, so a script can gate on it.
    return 0 if any(r.state.usable for r in reports) else 1


def _cmd_models_scan(args: argparse.Namespace) -> int:
    """Discover models under the roots, and report what actually changed.

    "Scanned" alone is not feedback: it cannot be told apart from "scanned and
    found nothing". The counts are what let the interface say something true.
    """
    from .library import load_registry, save_registry

    registry = load_registry()
    before = len(registry.report())
    added = registry.discover()
    if added:
        save_registry(registry)

    result = {
        "roots": registry.roots,
        "found": before + len(added),
        "added": len(added),
        "already_known": before,
        "added_paths": [entry.path for entry in added],
    }

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    print(
        f"{result['found']} model(s) found, {result['added']} added, "
        f"{result['already_known']} already registered"
    )
    for path in result["added_paths"]:
        print(f"  + {path}")
    return 0


def _cmd_models_import(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .library import load_registry, save_registry
    from .models import slug_for

    expected = getattr(args, "expect", None)
    if expected:
        # Checked before the registry is touched: an import that turns out to be
        # the wrong model must leave the library exactly as it was, rather than
        # adding an entry the caller then has to undo.
        #
        # Identity is decided here, on the same rule the catalogue reconciles
        # with, so "is this the 120B?" has one answer in the whole product.
        found = slug_for(Path(args.path).expanduser().resolve().name)
        if found != expected:
            raise ConfigError(
                f"that directory is {found!r}, not {expected!r}. Choose the directory "
                f"holding the {expected} weights."
            )

    registry = load_registry()
    entry = registry.add(args.path)
    save_registry(registry)

    if getattr(args, "json", False):
        print(
            json.dumps(
                {"id": entry.id, "imported": entry.path, "name": entry.name},
                indent=2,
            )
        )
        return 0

    print(f"added {entry.name}")
    print(f"  {entry.path}")
    return 0


def _cmd_models_forget(args: argparse.Namespace) -> int:
    from .library import load_registry, save_registry

    registry = load_registry()
    entry = registry.forget(args.path)
    save_registry(registry)
    # Said explicitly: forgetting must never be mistaken for deleting.
    print(f"removed {entry.name} from the library; its files were left in place")
    return 0


def _cmd_roots_list(_args: argparse.Namespace) -> int:
    from .library import load_registry, volume_for

    for root in load_registry().roots:
        volume = volume_for(root)
        if volume.is_external:
            state = "mounted" if volume.mounted else "NOT MOUNTED"
            print(f"  {root}  [volume {volume.name}: {state}]")
        else:
            print(f"  {root}")
    return 0


def _cmd_roots_add(args: argparse.Namespace) -> int:
    from .library import load_registry, save_registry

    registry = load_registry()
    registry.add_root(args.path)
    save_registry(registry)
    print(f"added root {args.path}")
    return 0


def _cmd_roots_remove(args: argparse.Namespace) -> int:
    from .library import load_registry, save_registry

    registry = load_registry()
    registry.remove_root(args.path)
    save_registry(registry)
    print(f"removed root {args.path}")
    return 0


def _cmd_models_inspect(args: argparse.Namespace) -> int:
    from .inspect_model import Verdict, inspect_model

    report = inspect_model(args.path)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"{report.verdict.value}  {report.path}")
        for reason in report.reasons:
            print(f"  - {reason}")
        if report.usable:
            print(
                f"  {report.model_type}, {report.quantization}, {report.layers} layers, "
                f"{report.experts} experts, {report.context_length} ctx, "
                f"{report.shards} shards, {_human_bytes(report.disk_bytes)}"
            )
    # A non-zero exit lets a script gate on this without parsing the output.
    return 0 if report.verdict is not Verdict.UNSUPPORTED else 1


def _cmd_models_inspect_adapter(args: argparse.Namespace) -> int:
    from .inspect_adapter import inspect_adapter

    report = inspect_adapter(args.path)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(f"{report.verdict.value}  {report.path}")
        for reason in report.reasons:
            print(f"  - {reason}")
        if report.trained_against:
            print(f"  trained against {report.trained_against}")
        if report.usable:
            print(f"  {report.tensor_count} tensors, {_human_bytes(report.weight_bytes)}")
    return 0 if report.usable else 1


def _cmd_requests(args: argparse.Namespace) -> int:
    payload = _management_request(f"/internal/requests?limit={args.limit}")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    lifetime = payload["lifetime"]
    window = payload["window"]
    print(
        f"lifetime    {lifetime['requests']} requests "
        f"({lifetime['completed']} completed, {lifetime['incomplete']} incomplete, "
        f"{lifetime['cancelled']} cancelled, {lifetime['failed']} failed)"
    )

    def rate(value: float | None, unit: str) -> str:
        return f"{value:.1f} {unit}" if value else "—"

    print(f"window      last {window['size']} of {window['capacity']}")
    print(f"  prefill   {rate(window['median_prefill_tokens_per_second'], 'tok/s')} median")
    print(f"  decode    {rate(window['median_decode_tokens_per_second'], 'tok/s')} median")
    print(f"  first tok {rate(window['median_time_to_first_token_seconds'], 's')} median")
    print(f"  queue     {rate(window['median_queue_wait_seconds'], 's')} median")
    ratio = window["cache_hit_ratio"]
    print(
        f"  cache     {ratio:.0%} of requests reused a prefix, "
        f"{window['tokens_reused']} tokens reused, {window['tokens_evaluated']} evaluated"
        if ratio is not None
        else "  cache     no data yet"
    )

    if payload["requests"]:
        print()
        print("recent")
        for entry in payload["requests"]:
            tools = ", ".join(call["name"] for call in entry["tool_calls"]) or "-"
            print(
                f"  {entry['request_id']}  {entry['outcome'] or 'running':<10} "
                f"in={entry['input_tokens']:<7} cached={entry['cached_tokens']:<7} "
                f"out={entry['output_tokens']:<5} "
                f"{(entry['duration_seconds'] or 0):.2f}s  tools={tools}"
            )
    return 0


def _cmd_cache_stats(args: argparse.Namespace) -> int:
    payload = _management_request("/internal/cache")
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    total = payload["hits"] + payload["misses"]
    print(f"sessions    {payload['entries']} / {payload['max_entries']}")
    print(f"memory      {_human_bytes(payload['bytes'])} / {_human_bytes(payload['max_bytes'])}")
    print(f"requests    {payload['hits']} reused, {payload['misses']} cold ({total} total)")
    print(f"hit ratio   {payload['hit_ratio']:.0%}")
    print(
        f"tokens      {payload['cached_tokens_total']} reused, "
        f"{payload['evaluated_tokens_total']} evaluated"
    )
    print(f"evictions   {payload['evictions']}")
    for model, stats in payload["by_model"].items():
        print(f"  {model}: {stats['sessions']} sessions, {_human_bytes(stats['bytes'])}")
    return 0


def _residency_detail(lifecycle: dict) -> str:
    """The idle-unload half of the residency line, when there is one to tell."""
    parts: list[str] = []
    # A missing key means the daemon did not report it; 0 means the user turned
    # the timer off. `or 0` collapsed the two and announced "disabled" for a
    # server that had simply not said, which is the one case where the claim is
    # unfounded.
    timeout = lifecycle.get("idle_timeout_seconds")
    if timeout == 0:
        parts.append("auto-unload disabled")
    elif timeout and lifecycle.get("auto_unload_armed"):
        idle = lifecycle.get("idle_seconds")
        parts.append(
            f"idle {idle:.0f}s of {timeout:.0f}s" if isinstance(idle, int | float)
            else f"auto-unload after {timeout:.0f}s idle"
        )
    if lifecycle.get("unload_reason"):
        parts.append(f"last release: {lifecycle['unload_reason']}")
    return f"  ({', '.join(parts)})" if parts else ""


def _cmd_models_unload(_args: argparse.Namespace) -> int:
    """Release the resident model without stopping the daemon.

    Here as well as on the dashboard because no essential operation may exist
    only in the desktop app (cahier 40). Both call the same endpoint, which
    calls the same supervisor operation the idle timer does.
    """
    # An explicit empty body rather than none: a POST with no `Content-Length`
    # is legal and awkward, and there is nothing to gain from sending one.
    payload = _management_request("/internal/model/unload", method="POST", body={})
    lifecycle = payload.get("lifecycle") or {}
    if not payload.get("released"):
        print("no model was resident; the server is unchanged")
        return 0
    print(f"model released; server still running ({lifecycle.get('state', 'unknown')})")
    return 0


def _cmd_cache_clear(_args: argparse.Namespace) -> int:
    payload = _management_request("/internal/cache", method="DELETE")
    print(
        f"cleared {payload['cleared_entries']} session(s), "
        f"{_human_bytes(payload['cleared_bytes'])} freed"
    )
    return 0


def _cmd_profiles_list(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    profiles = load_profiles()

    if getattr(args, "json", False):
        # The desktop app consumes this. Emitting structured output here keeps
        # the schema in one place instead of having the app parse a table.
        print(
            json.dumps(
                {
                    "default": profiles.default,
                    "profiles": [asdict(p) for p in profiles.profiles.values()],
                },
                indent=2,
            )
        )
        return 0

    if not profiles.profiles:
        print(f"No profiles configured. Create one with `{invoked_as()} profiles new <name>`.")
        return 0
    for name, profile in sorted(profiles.profiles.items()):
        marker = "*" if name == profiles.default else " "
        # A profile with no model is normal now: the daemon loads on demand.
        # "None" would read like a fault rather than a choice.
        served = profile.default_model or "(loads on demand)"
        print(
            f"{marker} {name:<24} {served:<22} "
            f"{profile.host}:{profile.port}  {profile.log_level.lower()}"
        )
    return 0


def _served_models() -> list:
    """What this server would serve right now, resolved exactly as it resolves it.

    The same construction the daemon makes -- library reports plus the per-model
    overrides -- so no interface can resolve a model, a name or an effort
    differently from the server that will answer the request.

    Uses the contained reading: a model whose served name is ambiguous is left
    out, and every other model is still offered. The strict reading belongs to
    the boundary that *stores* a configuration, not to a form asking what exists.
    """
    from .library.registry import load_registry
    from .model_settings import load_model_settings
    from .models import resolve_served_catalogue

    try:
        catalogue = resolve_served_catalogue(
            load_registry().report(), overrides=load_model_settings().overrides
        )
    except Exception:  # noqa: BLE001 - a broken library must not break the form
        return []
    return list(catalogue.models)


def _installed_slugs() -> list[str]:
    """Stable library ids, which is what a profile stores."""
    return [model.id for model in _served_models()]


def _model_choice_labels() -> dict[str, str]:
    """How to name those ids to a human, without storing the name."""
    return {
        model.id: f"{model.display_name} — served as {model.slug}"
        for model in _served_models()
    }


def _launch_models() -> list:
    """Installed models with their **effective** reasoning effort."""
    from .codex.launch import LaunchModel

    return [
        LaunchModel(
            id=model.id,
            slug=model.slug,
            display_name=model.display_name,
            reasoning_effort=model.default_reasoning_effort.value,
        )
        for model in _served_models()
    ]


def _cmd_profiles_schema(_args: argparse.Namespace) -> int:
    """The description a configuration form is generated from.

    Emitted by the server so no client has to carry its own idea of which
    settings exist or what values they accept.
    """
    from .profile_schema import schema

    print(json.dumps(schema(_installed_slugs(), labels=_model_choice_labels()), indent=2))
    return 0


def _cmd_profiles_show(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    profile = load_profiles().get(args.name)
    print(json.dumps(asdict(profile), indent=2))
    return 0


def _cmd_profiles_set(args: argparse.Namespace) -> int:
    """Apply `field=value` assignments, validating before writing.

    Nothing is saved unless every assignment is acceptable: a half-applied
    change is harder to reason about than a rejected one.
    """
    from dataclasses import asdict, replace

    from .profile_schema import coerce, field_names, validate

    profiles = load_profiles()
    profile = profiles.get(args.name)

    updates: dict[str, object] = {}
    for assignment in args.assignments:
        if "=" not in assignment:
            raise ConfigError(f"{assignment!r} is not field=value")
        field, _, raw = assignment.partition("=")
        field = field.strip()
        if field not in field_names():
            known = ", ".join(sorted(field_names()))
            raise ConfigError(f"unknown setting {field!r}. Known settings: {known}")
        try:
            updates[field] = coerce(field, raw)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    problems = validate(updates)
    if problems:
        raise ConfigError("; ".join(f"{p.field}: {p.message}" for p in problems))

    updated = replace(profile, **updates)
    # The dataclass keeps its own invariants — a reasoning level from another
    # model family, an impossible port — beyond the declared bounds.
    updated.validate()
    profiles.profiles[args.name] = updated
    save_profiles(profiles)

    if getattr(args, "json", False):
        print(json.dumps(asdict(updated), indent=2))
        return 0

    for field, value in updates.items():
        print(f"{args.name}.{field} = {value}")
    return 0


def _cmd_profiles_add(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    profiles.put(
        ServerProfile(
            name=args.name,
            model=args.model,
            served_model_name=args.served_model_name,
            context_length=args.context_length,
            reasoning_effort=args.reasoning_effort,
            host=args.host,
            port=args.port,
        )
    )
    save_profiles(profiles)
    print(f"saved profile {args.name!r}")
    return 0


def _emit_profile(profile: ServerProfile, *, as_json: bool, what: str) -> int:
    if as_json:
        print(json.dumps(asdict(profile), indent=2))
    else:
        print(f"{what} {profile.name}")
    return 0


def _cmd_profiles_new(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    profile = profiles.create(args.name)
    save_profiles(profiles)
    return _emit_profile(profile, as_json=args.json, what="created")


def _cmd_profiles_duplicate(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    profile = profiles.duplicate(args.source, args.name)
    save_profiles(profiles)
    return _emit_profile(profile, as_json=args.json, what="created")


def _cmd_profiles_rename(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    profile = profiles.rename(args.name, args.new_name)
    save_profiles(profiles)
    return _emit_profile(profile, as_json=args.json, what="renamed to")


def _profile_in_use(profile: ServerProfile) -> str | None:
    """Whether a running daemon looks like it belongs to this profile.

    A profile is not owned by a server -- the daemon reads it once at start and
    keeps no reference. So this is a *heuristic*, deliberately named as one: a
    live runtime file on the same host and port is good enough to warn a user
    before they delete the configuration they are currently running, and never
    good enough to refuse outright without `--force`.
    """
    state = load_runtime_state()
    if state is None or not state.is_running:
        return None
    if (state.host, state.port) != (profile.host, profile.port):
        return None
    return f"{state.host}:{state.port} (pid {state.pid})"


def _cmd_profiles_remove(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    profile = profiles.get(args.name)

    in_use = _profile_in_use(profile)
    if in_use and not args.force:
        print(
            f"a server appears to be running on {in_use}, which is this profile's "
            f"endpoint. Stop it first, or pass --force to delete the profile anyway "
            f"(the running server keeps the settings it started with).",
            file=sys.stderr,
        )
        return 1

    profiles.remove(args.name)
    save_profiles(profiles)
    print(f"removed profile {args.name!r}")
    return 0


def _cmd_profiles_default(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    profiles.get(args.name)
    profiles.default = args.name
    save_profiles(profiles)
    print(f"default profile is now {args.name!r}")
    return 0


def _cmd_codex_launch(args: argparse.Namespace) -> int:
    """Print the Codex configuration that points at this server.

    `~/.codex/config.toml` is never written (cahier 30): it is the user's own
    Codex setup, shared with their cloud usage. Both forms are text to copy.
    """
    from .codex.launch import render_command, render_config, resolve

    profiles = load_profiles()
    state = load_runtime_state()

    profile = profiles.resolve(args.profile) if profiles.profiles else None
    if state is not None and state.is_running:
        host, port = state.host, state.port
    elif profile is not None:
        host, port = profile.host, profile.port
        print("# no server running; using the profile's configured endpoint", file=sys.stderr)
    else:
        host, port = DEFAULT_HOST, DEFAULT_PORT
        print("# no server running and no profile; using the defaults", file=sys.stderr)

    settings = resolve(
        host=host,
        port=port,
        default_model=profile.default_model if profile else None,
        available=tuple(_launch_models()),
        chosen=getattr(args, "model", None),
    )

    if getattr(args, "models_json", False):
        # What an interface needs to offer a choice: the models this server
        # serves, their effective efforts, and what it would pick unprompted.
        # Emitted from here so the desktop form declares none of it itself.
        print(
            json.dumps(
                {
                    "default": settings.model.library_id if settings.model else None,
                    "models": [
                        {
                            "id": m.library_id,
                            "slug": m.slug,
                            "display_name": m.display_name,
                            "reasoning_effort": m.reasoning_effort,
                        }
                        for m in settings.available
                    ],
                },
                indent=2,
            )
        )
        return 0

    if getattr(args, "config", False):
        print(render_config(settings))
        return 0

    if settings.needs_a_model:
        # Said once, on stderr, so the command itself stays copy-pasteable.
        known = ", ".join(m.slug for m in settings.available) or "none installed"
        print(
            f"# no default model set; add -c model=<id> before running. Available: {known}",
            file=sys.stderr,
        )
    print(render_command(settings, prompt=args.prompt))
    return 0


def _cmd_models_download(args: argparse.Namespace) -> int:
    """Download through the running server, so one machine has one downloader.

    Doing it in this process instead would let a CLI invocation and the desktop
    app fetch the same repository into the same directory at once.
    """
    import time

    payload: dict[str, object] = {"repo": args.repo}
    if args.destination:
        payload["destination"] = args.destination

    progress = _management_request("/internal/downloads", method="POST", body=payload)
    print(f"downloading {progress['repo']}")
    print(f"  into {progress['destination']}")

    if not args.watch:
        print(f"  follow it with `{invoked_as()} models download --watch`, or in the app")
        return 0

    while True:
        time.sleep(2)
        status = _management_request("/internal/downloads")
        current = status["active"] or status["last"]
        if current is None:
            return 0
        line = f"  {current['state']}"
        if current.get("fraction") is not None:
            line += f"  {current['fraction'] * 100:5.1f}%"
        line += f"  {_human_bytes(current['downloaded_bytes'])}"
        if current.get("total_bytes"):
            line += f" / {_human_bytes(current['total_bytes'])}"
        if current.get("bytes_per_second"):
            line += f"  {_human_bytes(int(current['bytes_per_second']))}/s"
        if current.get("eta_seconds"):
            line += f"  eta {int(current['eta_seconds'] // 60)}m"
        print(line)
        if current["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            if current.get("detail"):
                print(f"  {current['detail']}")
            return 0 if current["state"] == "COMPLETED" else 1


def _cmd_roots_dispatch(args: argparse.Namespace) -> int:
    return {
        "list": _cmd_roots_list,
        "add": _cmd_roots_add,
        "remove": _cmd_roots_remove,
    }[args.roots_command](args)


def _cmd_doctor(_args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    return run_doctor()


_COMMANDS = {
    ("serve", None): _cmd_serve,
    ("status", None): _cmd_status,
    ("doctor", None): _cmd_doctor,
    ("models", "list"): _cmd_models_list,
    ("models", "catalog"): _cmd_models_catalog,
    ("models", "storage"): _cmd_models_storage,
    ("models", "config"): _cmd_models_config,
    ("models", "config-schema"): _cmd_models_config_schema,
    ("models", "unload"): _cmd_models_unload,
    ("models", "inspect"): _cmd_models_inspect,
    ("models", "inspect-adapter"): _cmd_models_inspect_adapter,
    ("models", "import"): _cmd_models_import,
    ("models", "scan"): _cmd_models_scan,
    ("models", "forget"): _cmd_models_forget,
    ("models", "download"): _cmd_models_download,
    ("models", "roots"): _cmd_roots_dispatch,
    ("requests", None): _cmd_requests,
    ("cache", "stats"): _cmd_cache_stats,
    ("cache", "clear"): _cmd_cache_clear,
    ("profiles", "list"): _cmd_profiles_list,
    ("profiles", "show"): _cmd_profiles_show,
    ("profiles", "schema"): _cmd_profiles_schema,
    ("profiles", "set"): _cmd_profiles_set,
    ("profiles", "new"): _cmd_profiles_new,
    ("profiles", "duplicate"): _cmd_profiles_duplicate,
    ("profiles", "rename"): _cmd_profiles_rename,
    ("profiles", "add"): _cmd_profiles_add,
    ("profiles", "remove"): _cmd_profiles_remove,
    ("profiles", "default"): _cmd_profiles_default,
    ("codex", "launch"): _cmd_codex_launch,
}


def main(argv: Sequence[str] | None = None) -> int:
    # Before anything reads configuration: the product was renamed, and the
    # application-data root moved with its bundle identifier.
    migrate_app_support_root()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    subcommand = (
        getattr(args, "models_command", None)
        or getattr(args, "cache_command", None)
        or getattr(args, "profiles_command", None)
        or getattr(args, "codex_command", None)
    )

    handler = _COMMANDS[(args.command, subcommand)]
    try:
        return handler(args)
    except ConfigError as exc:
        # Configuration problems are the user's to fix, so they get a plain
        # message rather than a traceback that buries it.
        return _fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
