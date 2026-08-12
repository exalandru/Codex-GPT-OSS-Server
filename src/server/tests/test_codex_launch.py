"""The Codex configuration this server generates.

One resolution behind two presentations: a one-shot command and a
`config.toml` fragment. They were built separately, and the model was
interpolated from a field that may legitimately be `None`, which produced
`-c model="None"` -- a model identity Codex cannot have.
"""

from __future__ import annotations

import pytest

from quantum_codex.codex.launch import (
    LaunchModel,
    LaunchSettings,
    render_command,
    render_config,
    resolve,
)

#: The two presets as the backend resolves them, with their effective efforts.
TWENTY = LaunchModel(slug="gpt-oss-20b", reasoning_effort="medium")
ONE_TWENTY = LaunchModel(slug="gpt-oss-120b", reasoning_effort="high")
BOTH = (TWENTY, ONE_TWENTY)


def settings(**kwargs) -> LaunchSettings:
    base = {"host": "127.0.0.1", "port": 8123, "default_model": None}
    return resolve(**{**base, **kwargs})


# -- the absent model --------------------------------------------------------


def test_no_default_model_never_renders_the_word_none() -> None:
    """The regression, in both forms."""
    chosen = settings(default_model=None, available=BOTH)

    for rendered in (render_command(chosen), render_config(chosen)):
        assert "None" not in rendered
        assert 'model="None"' not in rendered


def test_no_default_model_omits_the_setting_rather_than_inventing_one() -> None:
    chosen = settings(available=BOTH)

    command = render_command(chosen)

    assert "-c model=" not in command
    assert chosen.needs_a_model is True


def test_the_config_form_names_the_models_a_user_could_choose() -> None:
    """A blank is not actionable; the available ids are."""
    rendered = render_config(settings(available=BOTH))

    assert 'model = "gpt-oss-20b"' in rendered
    assert 'model = "gpt-oss-120b"' in rendered


def test_a_single_installed_model_needs_no_choice() -> None:
    """Nothing to choose between, so nothing is invented by choosing it."""
    chosen = settings(available=(TWENTY,))

    assert chosen.model is not None
    assert chosen.model.slug == "gpt-oss-20b"
    assert chosen.needs_a_model is False


def test_nothing_installed_and_no_default_still_renders_usable_provider_config() -> None:
    rendered = render_config(settings())

    assert "model_provider" in rendered
    assert "None" not in rendered


# -- the chosen model --------------------------------------------------------


def test_the_profile_default_is_used() -> None:
    assert settings(default_model="gpt-oss-120b", available=BOTH).model == ONE_TWENTY


def test_an_explicit_choice_beats_the_profile_default() -> None:
    chosen = settings(default_model="gpt-oss-120b", chosen="gpt-oss-20b", available=BOTH)

    assert chosen.model == TWENTY
    assert '-c model="gpt-oss-20b"' in render_command(chosen)
    # And it brings that model's effort, not the default model's.
    assert '-c model_reasoning_effort="medium"' in render_command(chosen)


# -- both forms describe the same provider -----------------------------------


@pytest.mark.parametrize("render", [render_command, render_config])
def test_every_form_carries_what_codex_0_147_needs(render) -> None:
    """Command auth is why this generator exists: without it Codex never reads
    /v1/models and falls back to defaults this server does not serve."""
    rendered = render(settings(default_model="gpt-oss-20b"))

    assert "qcs" in rendered
    assert "responses" in rendered
    assert "http://127.0.0.1:8123/v1" in rendered
    assert "echo" in rendered


def test_the_two_forms_agree_on_the_model() -> None:
    chosen = settings(default_model="gpt-oss-20b")

    assert 'model="gpt-oss-20b"' in render_command(chosen)
    assert 'model = "gpt-oss-20b"' in render_config(chosen)


def test_the_endpoint_follows_the_profile() -> None:
    chosen = settings(host="127.0.0.1", port=9999, default_model="gpt-oss-20b")

    assert "http://127.0.0.1:9999/v1" in render_command(chosen)
    assert "http://127.0.0.1:9999/v1" in render_config(chosen)


def test_a_prompt_produces_an_exec_invocation() -> None:
    rendered = render_command(settings(default_model="gpt-oss-20b"), prompt="say hi")

    assert rendered.startswith("codex exec")
    assert '"say hi"' in rendered


def test_the_config_form_never_writes_anything() -> None:
    """It is text to paste. The user's own Codex config is theirs (cahier 30)."""
    rendered = render_config(settings(default_model="gpt-oss-20b"))

    assert "~/.codex/config.toml" in rendered
    assert "untouched" in rendered


# -- the documented snippet --------------------------------------------------
#
# The README prints a `codex …` invocation for a user to copy. It named
# `model_provider="quantum-codex"` long after the generator had settled on
# `qcs`: still a working configuration, and a *different* provider from the one
# the app and the CLI hand out, so a user following the README ended up with two
# entries in `~/.codex/config.toml` that could not be told apart by name.
#
# Documentation is not usually worth a test. This is, because the failure is
# silent on both sides: nothing errors, and the two configurations only diverge
# once someone tries to reconcile them.


def _readme() -> str:
    from pathlib import Path

    # tests/ -> server/ -> src/ -> repository root
    readme = Path(__file__).resolve().parents[3] / "README.md"
    assert readme.is_file(), f"expected the repository README at {readme}"
    return readme.read_text()


def test_the_readme_quotes_the_provider_id_this_build_emits() -> None:
    from quantum_codex.codex.launch import PROVIDER_ID, PROVIDER_NAME

    readme = _readme()
    rendered = render_command(settings(default_model="gpt-oss-20b"))

    # Every provider key in the rendered command appears verbatim in the README.
    for line in rendered.splitlines():
        setting = line.strip().rstrip(" \\").removeprefix("-c ").strip("'")
        if setting.startswith(("model_provider=", f"model_providers.{PROVIDER_ID}.")):
            assert setting in readme, f"the README does not document `{setting}`"

    assert f'name="{PROVIDER_NAME}"' in readme


def test_the_readme_does_not_still_document_a_provider_id_nothing_emits() -> None:
    # The other direction, which is the half that actually broke: the README can
    # name every current key and still carry the previous ones beside them.
    from quantum_codex.codex.launch import PROVIDER_ID

    readme = _readme()
    quoted = {
        line.split("model_providers.", 1)[1].split(".", 1)[0]
        for line in readme.splitlines()
        if "model_providers." in line
    }

    assert quoted == {PROVIDER_ID}, f"the README documents provider ids {quoted - {PROVIDER_ID}}"


# -- the model Codex is actually told to use ----------------------------------
#
# Measured with the real client: given no `model`, Codex does not fall back to
# a model this provider serves. It falls back to its own cloud model selection,
# against a provider that has none of them. So "the profile has no default" is a
# valid server state and an *invalid* launch configuration, and the two must not
# be conflated.
#
# The same applies to reasoning effort. Codex exposes no way to choose it for a
# custom provider's model after launch, so whatever is emitted here is what the
# model runs at for the whole session.


def test_the_profile_default_is_emitted_as_the_served_id() -> None:
    """Item 1: default 20B -> the id `/v1/models` publishes, not a directory."""
    rendered = render_command(settings(default_model="gpt-oss-20b", available=BOTH))

    assert '-c model="gpt-oss-20b"' in rendered
    # Not the directory the weights happen to live in.
    assert "mxfp4" not in rendered


def test_the_profile_default_carries_its_effective_reasoning_effort() -> None:
    rendered = render_command(settings(default_model="gpt-oss-20b", available=BOTH))

    assert '-c model_reasoning_effort="medium"' in rendered


def test_the_other_model_carries_its_own_effort_not_the_first_ones() -> None:
    """Item 2: the two models resolve independently."""
    rendered = render_command(settings(default_model="gpt-oss-120b", available=BOTH))

    assert '-c model="gpt-oss-120b"' in rendered
    assert '-c model_reasoning_effort="high"' in rendered
    assert "medium" not in rendered


def test_no_default_and_several_installed_produces_no_runnable_command() -> None:
    """Item 3: silence is not a safe default, and neither is a guess.

    With nothing chosen the command must not name a model — an interface has to
    ask. What it must *never* do is emit a placeholder, which reads as runnable
    and is not.
    """
    resolved = settings(default_model=None, available=BOTH)

    assert resolved.needs_a_model is True
    for rendered in (render_command(resolved), render_config(resolved)):
        assert "-c model=" not in rendered.replace("-c model_provider", "")
        assert "<model>" not in rendered
        assert "model_reasoning_effort" not in rendered


def test_choosing_a_model_does_not_touch_the_profile_default() -> None:
    """Item 4: the choice is for this configuration, not a settings change.

    `resolve` is a pure function of what it is given: it returns settings and
    writes nothing. Pinning that here is what stops a later convenience —
    "remember the last choice" — from quietly becoming a profile edit.
    """
    import inspect

    from quantum_codex.codex import launch

    chosen = settings(default_model=None, available=BOTH, chosen="gpt-oss-20b")
    assert chosen.model is not None
    assert chosen.model.slug == "gpt-oss-20b"

    # Still no default, and still asking for one when nothing is chosen.
    assert settings(default_model=None, available=BOTH).needs_a_model is True
    source = inspect.getsource(launch)
    for writer in ("save_profiles", "save_model_settings", "open(", "write_text"):
        assert writer not in source, f"the generator must not persist anything ({writer})"


@pytest.mark.parametrize("render", [render_command, render_config])
def test_no_path_can_render_the_word_none_as_a_model(render) -> None:
    """Item 5, over every combination that could produce one."""
    for resolved in (
        settings(default_model=None, available=()),
        settings(default_model=None, available=BOTH),
        settings(default_model="gpt-oss-20b", available=BOTH),
        settings(default_model=None, available=BOTH, chosen="gpt-oss-120b"),
    ):
        rendered = render(resolved)
        assert 'model="None"' not in rendered
        assert "model = \"None\"" not in rendered


def test_the_two_forms_resolve_the_same_model_and_effort() -> None:
    """Item 6: one resolution, two presentations."""
    resolved = settings(default_model="gpt-oss-120b", available=BOTH)

    command, config = render_command(resolved), render_config(resolved)

    assert '-c model="gpt-oss-120b"' in command
    assert 'model = "gpt-oss-120b"' in config
    assert '-c model_reasoning_effort="high"' in command
    assert 'model_reasoning_effort = "high"' in config


def test_a_model_with_no_known_effort_omits_it_rather_than_inventing_one() -> None:
    """A custom import this build ships no opinion about.

    Emitting a made-up effort would be a guess presented as configuration; the
    model is still named, which is the part Codex cannot do without.
    """
    custom = LaunchModel(slug="my-own-gpt-oss", reasoning_effort=None)

    rendered = render_command(settings(default_model="my-own-gpt-oss", available=(custom,)))

    assert '-c model="my-own-gpt-oss"' in rendered
    assert "model_reasoning_effort" not in rendered


# -- two namespaces, one lookup ----------------------------------------------


def test_a_stale_selector_is_not_rendered_as_a_model_codex_could_ask_for() -> None:
    """`default_model` is a QCS selector; what is rendered is a served name.

    A profile stores the stable library id. Rendering an id that matches nothing
    would hand Codex a name this server never published, and the launch would
    quietly not use this server at all.
    """
    chosen = settings(default_model="library-gone", available=BOTH)

    assert chosen.model is None
    assert chosen.needs_a_model is True


def test_a_served_name_cannot_shadow_another_model_s_library_id() -> None:
    """The ambiguity a single merged lookup introduces.

    One model's served name may equal another's library id -- nothing forbids
    it, since the two are different namespaces. Resolving both from one mapping
    made the answer depend on iteration order, so renaming one model could
    redirect a profile that points at another.
    """
    renamed = LaunchModel(id="library-a", slug="library-b", reasoning_effort="low")
    other = LaunchModel(id="library-b", slug="codex-local", reasoning_effort="high")

    # Both orderings, because order is exactly what a merged mapping made the
    # answer depend on.
    for available in ((renamed, other), (other, renamed)):
        resolved = settings(default_model="library-b", available=available)

        assert resolved.model is other, available
        assert resolved.model.slug == "codex-local"
        assert 'model = "codex-local"' in render_config(resolved)
