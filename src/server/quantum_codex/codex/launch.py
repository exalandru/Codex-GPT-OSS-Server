"""Generating the Codex configuration that points at this server.

Two presentations of one decision:

``command``   a one-shot `codex …` invocation, passing everything as overrides
``config``    a `~/.codex/config.toml` fragment, for the Codex CLI's global
              configuration and for the VS Code extension

Both are rendered from the same resolved settings. They existed as one string
built inline, and the model was interpolated straight from a profile field that
may legitimately be ``None`` -- which produced ``-c model="None"``, a model
identity Codex cannot have. Anything that renders provider configuration goes
through :func:`resolve` now.

This never writes to `~/.codex/config.toml` (cahier 30). It is the user's own
file, shared with their cloud Codex; we hand them text to paste.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The provider id used in Codex configuration, and the name Codex displays.
#:
#: One definition, so the command form and the config-file form describe the
#: same provider. `qcs` matches the product's bundle identifier and is short
#: enough to read inside a `model_providers.<id>.…` key, which is where it
#: mostly appears.
PROVIDER_ID = "qcs"
PROVIDER_NAME = "QCS"


@dataclass(frozen=True)
class LaunchModel:
    """One model Codex could be pointed at, with what this server knows about it.

    ``slug`` is the *served* identity: the id ``GET /v1/models`` publishes and
    the one ``_resolve_model`` routes on. Never a directory name, and never a
    display alias -- those are library metadata, and a client that asked for one
    would be refused.
    """

    slug: str
    #: Immutable library identity used by QCS selectors and profiles. Codex is
    #: always given ``slug`` instead.
    id: str | None = None
    display_name: str | None = None
    #: The effective reasoning effort for this model, resolved by the backend
    #: (catalogue default, then the user's per-model override). ``None`` only
    #: for a model this build knows nothing about, where inventing one would be
    #: a guess presented as configuration.
    reasoning_effort: str | None = None

    @property
    def library_id(self) -> str:
        return self.id or self.slug


@dataclass(frozen=True)
class LaunchSettings:
    """Everything needed to point Codex at this server."""

    base_url: str
    #: The model Codex should ask for, or ``None`` when nothing was chosen.
    #:
    #: ``None`` is a real state -- the daemon loads on demand -- and it is never
    #: rendered. A caller that needs a model asks the user; it does not invent
    #: one, and it certainly does not stringify the absence.
    model: LaunchModel | None = None
    #: Models the user could pick, so an interface can offer a selector instead
    #: of a blank.
    available: tuple[LaunchModel, ...] = ()

    @property
    def needs_a_model(self) -> bool:
        """Whether the output is incomplete without a choice.

        Codex does not fall back to *this server's* models when no model is
        given -- it falls back to its own cloud model selection, against a
        provider that serves neither. Measured with the real client. So an
        omitted model is not "load on demand", it is a launch that quietly does
        not use this server at all.
        """
        return self.model is None


def resolve(
    *,
    host: str,
    port: int,
    default_model: str | None,
    available: tuple[LaunchModel, ...] = (),
    chosen: str | None = None,
) -> LaunchSettings:
    """The provider settings, with the model decided once.

    Precedence: an explicit choice, then the profile's default, then -- only
    when exactly one model is installed -- that model, because there is nothing
    to choose between. Otherwise no model, and the caller must ask.

    A named model is matched against what is installed so its effort comes from
    the backend rather than from whoever is rendering.

    ``default_model`` and ``chosen`` are QCS *selectors* -- a profile stores the
    stable library id -- while what is rendered is always the served name. A
    selector that matches nothing while models are known is therefore stale, and
    rendering it would hand Codex an internal id it can never ask for; the
    caller is told to choose instead. With nothing to match against it is taken
    at face value, so the command still works with no server running.
    """
    # Two lookups rather than one merged mapping: a profile stores the stable
    # library id, and a served name that happens to equal *another* model's id
    # must not shadow it. Merging them made the answer depend on iteration
    # order, which is how a rename of one model would silently redirect a
    # profile pointing at another.
    by_id = {model.library_id: model for model in available}
    by_slug = {model.slug: model for model in available}
    name = chosen or default_model
    if name is None and len(available) == 1:
        name = available[0].library_id

    model = (by_id.get(name) or by_slug.get(name)) if name else None
    if model is None and name and not available:
        # Nothing to match against -- no server running, or a library that could
        # not be read. The name is honoured as a served name, which is what it
        # is: refusing here would only move the error somewhere less clear.
        model = LaunchModel(slug=name)
    return LaunchSettings(
        base_url=f"http://{host}:{port}/v1",
        model=model,
        available=tuple(available),
    )


def _auth_line(prefix: str) -> str:
    """The command-backed auth Codex 0.147 requires to refresh model metadata.

    Not authentication: this server has none and binds loopback. Codex only
    calls `/v1/models` for a provider that declares command auth, so without
    this it never learns what this server serves and falls back to generic
    defaults.
    """
    return f'{prefix}auth={{command="echo", args=["local"]}}'


def render_command(settings: LaunchSettings, *, prompt: str | None = None) -> str:
    """The one-shot invocation, with nothing written to disk."""
    import json

    parts = [
        "codex exec" if prompt else "codex",
        f'-c model_provider="{PROVIDER_ID}"',
        f'-c model_providers.{PROVIDER_ID}.name="{PROVIDER_NAME}"',
        f'-c model_providers.{PROVIDER_ID}.base_url="{settings.base_url}"',
        f'-c model_providers.{PROVIDER_ID}.wire_api="responses"',
        "-c '" + _auth_line(f"model_providers.{PROVIDER_ID}.") + "'",
    ]
    # Emitted only when there is one. `model="None"` is not a model.
    if settings.model:
        parts.insert(1, f'-c model="{settings.model.slug}"')
        # Codex offers no way to choose reasoning effort for a custom provider's
        # model after launch -- the picker that would is for its own models. So
        # if it is not set here it cannot be set at all, and the model runs at
        # whatever Codex assumes rather than what this server resolved for it.
        if settings.model.reasoning_effort:
            parts.insert(2, f'-c model_reasoning_effort="{settings.model.reasoning_effort}"')
    if prompt:
        parts.append(json.dumps(prompt))
    return " \\\n  ".join(parts)


def render_config(settings: LaunchSettings) -> str:
    """A `~/.codex/config.toml` fragment.

    For the Codex CLI's persistent configuration and for the VS Code extension,
    neither of which takes `-c` overrides.
    """
    lines = [
        "# Quantum Codex GPT-OSS Server",
        "# Append to ~/.codex/config.toml. Your existing settings are untouched.",
        "",
    ]
    if settings.model:
        lines.append(f'model = "{settings.model.slug}"')
        # Same reason as the command form, and from the same resolution: this is
        # the only place a legacy-provider model's effort can be set.
        if settings.model.reasoning_effort:
            lines.append(f'model_reasoning_effort = "{settings.model.reasoning_effort}"')
    else:
        lines.append("# No default model chosen. Set one here, for example:")
        for candidate in settings.available or (LaunchModel(slug="gpt-oss-20b"),):
            lines.append(f'#   model = "{candidate.slug}"')
    lines += [
        f'model_provider = "{PROVIDER_ID}"',
        "",
        f"[model_providers.{PROVIDER_ID}]",
        f'name = "{PROVIDER_NAME}"',
        f'base_url = "{settings.base_url}"',
        'wire_api = "responses"',
        "",
        "# Codex 0.147 only refreshes model metadata for a provider declaring",
        "# command-backed auth. This server has no authentication and binds",
        "# loopback; without this line Codex never reads /v1/models.",
        f"[model_providers.{PROVIDER_ID}.auth]",
        'command = "echo"',
        'args = ["local"]',
    ]
    return "\n".join(lines)
