"""The Codex-native ``/v1/models`` payload.

Every assertion here is checked against the Codex 0.147.0 source, because this
schema's failure mode is *silence*: a malformed entry does not make Codex raise,
it makes Codex fall back to generic defaults and print a warning. A test that
only checked "the endpoint returns some JSON" would pass in exactly the
situation this endpoint exists to prevent.

Source of truth: ``codex-rs/protocol/src/openai_models.rs`` (``ModelInfo``,
``ModelsResponse``, ``deserialize_model_infos_with_legacy_base``).
"""

from __future__ import annotations

import pytest

from quantum_codex.canonical import CanonicalTurn, ReasoningEffort
from quantum_codex.codex.model_metadata import (
    PARITY_CLAUSES,
    WITHHELD_CLAUSES,
    WRITE_BOUNDARY_CLAUSE,
    build_model_info,
    build_models_response,
)
from quantum_codex.harmony.render import HarmonyRenderer
from quantum_codex.models import ServedModel

# Fields declared in Rust as `Option<T>` *without* `#[serde(default)]`. Serde
# still requires the key to be present; null is fine, absent fails the decode of
# the entire response.
REQUIRED_NULLABLE_KEYS = (
    "description",
    "availability_nux",
    "upgrade",
    "default_verbosity",
    "apply_patch_tool_type",
)

# Fields with no default at all.
REQUIRED_KEYS = (
    "slug",
    "display_name",
    "supported_reasoning_levels",
    "shell_type",
    "visibility",
    "supported_in_api",
    "priority",
    "support_verbosity",
    "truncation_policy",
    "supports_parallel_tool_calls",
    "experimental_supported_tools",
)


@pytest.fixture
def model() -> ServedModel:
    return ServedModel(
        slug="gpt-oss-20b",
        display_name="GPT-OSS 20B (MLX)",
        context_window=131072,
    )


def test_envelope_is_models_not_the_openai_list_shape(model: ServedModel) -> None:
    payload = build_models_response((model,))

    # Codex deserializes `ModelsResponse { models: Vec<ModelInfo> }`. The OpenAI
    # `{"object":"list","data":[...]}` shape is what produced
    # `failed to refresh available models`.
    assert set(payload) == {"models"}
    assert isinstance(payload["models"], list)
    assert payload["models"][0]["slug"] == "gpt-oss-20b"


def test_every_required_key_is_present(model: ServedModel) -> None:
    info = build_model_info(model)

    for key in REQUIRED_KEYS:
        assert key in info, f"missing required key: {key}"


@pytest.mark.parametrize("key", REQUIRED_NULLABLE_KEYS)
def test_nullable_keys_are_present_not_omitted(key: str, model: ServedModel) -> None:
    """`Option<T>` without a serde default is a required key, not an optional one."""
    assert key in build_model_info(model)


def test_an_entry_carries_its_own_instructions(model: ServedModel) -> None:
    # Codex rejects an entry with neither `base_instructions` nor
    # `model_messages.instructions_template`, with an explicit decode error.
    info = build_model_info(model)

    assert info["base_instructions"]
    assert len(info["base_instructions"]) > 200


def test_input_modalities_are_stated_because_the_default_allows_images(
    model: ServedModel,
) -> None:
    # Omitting the field means `["text", "image"]` -- Codex would then send
    # images this server rejects.
    assert build_model_info(model)["input_modalities"] == ["text"]


def test_only_gpt_oss_reasoning_levels_are_advertised(model: ServedModel) -> None:
    info = build_model_info(model)
    efforts = [level["effort"] for level in info["supported_reasoning_levels"]]

    assert efforts == ["low", "medium", "high"]
    # Codex's own enum also has xhigh/max/ultra; those belong to other model
    # families and Harmony has no such levels.
    for absent in ("xhigh", "max", "ultra", "minimal", "none"):
        assert absent not in efforts
    assert all(level["description"] for level in info["supported_reasoning_levels"])


def test_sequential_tool_calling_is_reported_honestly(model: ServedModel) -> None:
    # Harmony's `<|call|>` ends the assistant turn, so one call per turn.
    assert build_model_info(model)["supports_parallel_tool_calls"] is False


def test_shell_type_follows_what_the_server_can_route(model: ServedModel) -> None:
    """The advertised surface and the accepted surface must agree (D5).

    This governs what *this server* claims, not what Codex sends: on macOS the
    `unified_exec` feature makes Codex expose a shell tool regardless of
    `shell_type`. Agreement here is still
    required, so the two never contradict each other.
    """
    assert model.supports_tools is True
    assert build_model_info(model)["shell_type"] == "default"

    without_tools = ServedModel(
        slug="x", display_name="x", context_window=1000, supports_tools=False
    )
    assert build_model_info(without_tools)["shell_type"] == "disabled"


def test_capabilities_this_server_lacks_are_not_advertised(model: ServedModel) -> None:
    info = build_model_info(model)

    assert info["supports_search_tool"] is False
    assert info["use_responses_lite"] is False
    assert info["include_apps_usage_instructions"] is False
    # No summariser exists here; raw reasoning items are emitted instead.
    assert info["supports_reasoning_summary_parameter"] is False
    assert info["default_reasoning_summary"] == "none"
    # Never round-tripped against this backend, so not claimed.
    assert info["apply_patch_tool_type"] is None


def test_context_window_is_reported_at_the_real_kv_limit(model: ServedModel) -> None:
    info = build_model_info(model)

    assert info["context_window"] == 131072
    assert info["max_context_window"] == 131072
    # Codex's default is 95, which would advertise a smaller window than the
    # server actually enforces.
    assert info["effective_context_window_percent"] == 100


def test_default_reasoning_level_matches_the_served_model(model: ServedModel) -> None:
    assert build_model_info(model)["default_reasoning_level"] == "medium"

    low = ServedModel(
        slug="x",
        display_name="x",
        context_window=1000,
        default_reasoning_effort=ReasoningEffort.LOW,
    )
    assert build_model_info(low)["default_reasoning_level"] == "low"


def test_metadata_follows_the_served_model_rather_than_a_hard_coded_slug() -> None:
    """D5: one definition drives the endpoint, so a second model needs no new code."""
    models = (
        ServedModel(slug="gpt-oss-20b", display_name="20B", context_window=131072),
        ServedModel(slug="gpt-oss-120b", display_name="120B", context_window=131072),
    )

    payload = build_models_response(models)

    assert [entry["slug"] for entry in payload["models"]] == ["gpt-oss-20b", "gpt-oss-120b"]


# ---------------------------------------------------------------------------
# Known-good prompt parity
# ---------------------------------------------------------------------------
#
# These clauses were present in the predecessor server's prompt -- the exact text
# recorded in the `session_meta` of a verified 79-minute autonomous 120B run --
# and were lost when this prompt was rewritten. Nothing caught that, because
# every existing test above asks whether the payload *decodes*, not what it says.
#
# The assertion is deliberately made against the **rendered** prompt rather than
# the constant: what steers the model is the Harmony developer message, and a
# clause that never reaches it is not restored no matter what the source says.
#
# Read this as regression protection only. The A/B experiment recorded next to
# `DEFAULT_BASE_INSTRUCTIONS` found these clauses do NOT fix exhaustive-audit
# scope closure, so a green test here is not evidence of that property.


def _rendered_instructions(model: ServedModel) -> str:
    """The developer-message text the model actually receives."""
    renderer = HarmonyRenderer()
    turn = CanonicalTurn(
        instructions=build_model_info(model)["base_instructions"],
        items=(),
        reasoning_effort=ReasoningEffort.MEDIUM,
    )
    return renderer.encoding.decode(renderer.render(turn))


@pytest.mark.parametrize("clause_id", sorted(PARITY_CLAUSES))
def test_restored_parity_clause_reaches_the_rendered_prompt(
    clause_id: str, model: ServedModel
) -> None:
    assert PARITY_CLAUSES[clause_id] in _rendered_instructions(model), (
        f"parity clause {clause_id} was dropped from the rendered instructions"
    )


@pytest.mark.parametrize("clause_id", sorted(WITHHELD_CLAUSES))
def test_withheld_clause_stays_out_until_someone_decides_otherwise(
    clause_id: str, model: ServedModel
) -> None:
    # M3 pushes against deliberate coverage work and no experiment justifies it.
    # Pinned so that adding it back is a decision, not an accident.
    assert WITHHELD_CLAUSES[clause_id] not in _rendered_instructions(model)


def test_scope_breadth_clause_does_not_license_ignoring_scope_bounds(
    model: ServedModel,
) -> None:
    """M1 must mean "a big ask is not an excuse", never "bounds are optional".

    The known-good wording was "Time constraints and scope are not valid stopping
    conditions", which reads as though a scope boundary itself were negotiable.
    Restoring it verbatim would have handed the model a licence the user
    explicitly did not want, so the sentence is the one place the parity text was
    clarified rather than copied.
    """
    rendered = _rendered_instructions(model)

    assert "stay inside the requested scope" in rendered
    assert "Time constraints and scope are not valid stopping conditions" not in rendered


def test_the_write_boundary_reaches_the_rendered_prompt(model: ServedModel) -> None:
    """Reading outside the project is fine; writing outside it needs asking.

    Stated to the model, not enforced here: this server never sees a filesystem
    operation, and the sandbox belongs to the client. A green assertion means the
    instruction is present, never that an external write is impossible.
    """
    rendered = _rendered_instructions(model)

    assert WRITE_BOUNDARY_CLAUSE in rendered
    # The permission is asymmetric on purpose, so both halves are pinned.
    assert "Read anything outside it" in rendered
    assert "unless the user explicitly authorizes that write" in rendered
