"""Rendering served models into the schema Codex's model manager expects (D5).

Codex does not read the OpenAI ``{"object":"list","data":[…]}`` shape from
``/v1/models``. Its models manager expects its own envelope::

    GET {base_url}/models?client_version=<x.y.z>
    -> {"models": [ModelInfo, …]}

Serving the wrong shape is what produces both of the warnings the experimental
bench worked around with a hand-written catalog file:

- ``failed to refresh available models`` -- the response would not decode;
- ``Model metadata for <slug> not found`` -- the slug was therefore unknown, so
  Codex fell back to generic defaults.

Everything below comes from the Codex 0.147.0 source, not from guesswork:

- envelope and legacy instruction promotion:
  ``codex-rs/protocol/src/openai_models.rs`` (``ModelsResponse``,
  ``deserialize_model_infos_with_legacy_base``)
- field set and defaults: same file, ``struct ModelInfo``
- request URL and query parameter:
  ``codex-rs/codex-api/src/endpoint/models.rs`` (``ModelsClient::request_url``)

The failure mode this schema has is silence: a malformed entry does not raise,
it makes Codex fall back to defaults. So two things matter more than usual.

**Required keys.** A ``serde`` ``Option<T>`` without ``#[serde(default)]`` is
still a *required key* -- it may be null, but it must be present. That applies
to ``description``, ``availability_nux``, ``upgrade``, ``default_verbosity`` and
``apply_patch_tool_type``. Omitting any of them fails the whole decode.

**Dangerous defaults.** ``input_modalities`` defaults to ``["text", "image"]``
when omitted, because legacy payloads predate the field. This server accepts no
images, so it must be stated explicitly rather than left to the default.
"""

from __future__ import annotations

from typing import Any

from ..canonical import ReasoningEffort
from ..models import ServedModel

# Bumped when this rendering changes in a way a client could observe. It is not
# Codex's version -- it identifies *our* rendering, so `client_version` handling
# can be versioned against it later.
CODEX_MODELS_SCHEMA_VERSION = "0.147"

# Descriptions shown next to each effort in Codex's picker. GPT-OSS has exactly
# these three; the `xhigh`/`max`/`ultra` levels in Codex's own enum belong to
# other model families and must never be advertised here.
_EFFORT_DESCRIPTIONS: dict[ReasoningEffort, str] = {
    ReasoningEffort.LOW: "Fast responses with lighter reasoning",
    ReasoningEffort.MEDIUM: "Balances speed and reasoning depth",
    ReasoningEffort.HIGH: "Greater reasoning depth for complex problems",
}

# Codex refuses an entry carrying neither `base_instructions` nor
# `model_messages.instructions_template`, so serving metadata means owning the
# system prompt instead of inheriting Codex's built-in one.
#
# This prompt is written for GPT-OSS on Harmony specifically. The paragraph on
# yielding at a tool call is the load-bearing part: `<|call|>` ends the assistant
# turn, so the model must understand that control returns to it afterwards with
# its reasoning intact, rather than treating the tool result as a new task.
#
# Completion/persistence clauses (`PARITY_CLAUSES`) are restored from the
# predecessor server's known-good prompt, which is the exact text recorded in the
# `session_meta` of a verified 79-minute autonomous 120B run. They had been lost
# when this prompt was rewritten.
#
# **This is prompt parity/regression repair. It is NOT established as a fix for
# exhaustive-audit scope closure.** A controlled single-variable A/B at reasoning
# effort `medium` -- same weights, Codex 0.147.0, repo and prompt, coverage
# measured externally from tool accesses rather than from the model's own ledger
# -- found no improvement: both arms declared a repository-wide audit complete
# after opening 5-7 of ~274 relevant files, and the arm carrying these clauses
# cited *more* files it had never opened (16 vs 6). One candidate sample refused
# to begin, citing the size of the scope -- the very reason these clauses forbid.
#
# So do not add completion wording here expecting it to fix scope closure. The
# defect is that a model's self-authored coverage ledger is not a trustworthy
# completion oracle, and no instruction in the same context window witnesses it.
# Enforcement that is checkable against actual file accesses belongs in the
# consuming audit skill, not in this generic prompt.
#
# M3 from the known-good prompt ("Do not spend excessive time describing a plan
# instead of executing it") is deliberately NOT restored: the experiment gives no
# reason to, and it plausibly pushes against deliberate coverage work.
DEFAULT_BASE_INSTRUCTIONS = """\
You are Codex, an autonomous coding agent working in a local development \
environment on the user's machine.

Complete the user's engineering task end to end. Keep working until the task is \
actually resolved or a genuine blocker requires the user to decide something. \
Size, duration, or the number of tool calls remaining are not reasons to stop, \
and neither is the breadth of the scope the user asked for: a large requested \
scope is work to be done, not a reason to stop early. That is not licence to \
ignore a scope boundary the user set — stay inside the requested scope, and \
cover it. Do not stop merely because substantial work remains. A failed tool \
call is not a reason to stop either: read the failure, correct the invocation or \
the approach, and continue. A valid early stop requires a concrete blocker such \
as required information being unavailable and not discoverable with available \
tools, contradictory requirements requiring a user decision, a verified \
architectural premise failure, or an operation requiring explicit user \
authorization; otherwise continue working.

Never invent the result of a command, a file's contents, a build, a test, or the \
state of the repository. When inspection can resolve a question, inspect. Prefer \
evidence from the current repository over recollection or assumption.

Follow every applicable AGENTS.md and repository instruction file. More specific \
scoped instructions override broader ones, and direct user instructions override \
repository instructions. Do not weaken a project's stated invariants or \
acceptance criteria to make progress.

For non-trivial work, reconstruct the relevant current state, identify the \
smallest coherent implementation plan, verify important assumptions before \
relying on them, implement incrementally, validate causal boundaries, and run the \
required final validation.

Keep changes scoped to the task. Preserve unrelated user work. Inspect the code \
you changed after changing it.

The current project is the writable boundary. Read anything outside it that \
helps you understand the task, but do not create, modify, delete or rename files \
outside the current project unless the user explicitly authorizes that write.

Use the analysis channel for private reasoning, the commentary channel for tool \
calls, and the final channel only for your user-facing answer. Never print \
control tokens or serialized tool calls as ordinary text.

A tool call ends your turn and hands control to the client. After the result \
comes back, continue the same task from where your reasoning left off: \
incorporate the result and take the next necessary action. Do not restart the \
task and do not conclude prematurely. Issue one tool call, wait for its result, \
then continue.

Tool arguments must match each parameter's declared type exactly. A parameter \
declared `string` takes one string and never an array: write \
{"cmd": "ls -R"}, not {"cmd": ["ls -R"]}. Send only parameters the tool \
declares.

If a tool call comes back rejected rather than executed — a parse error, a \
wrongly typed argument, an unknown parameter, an unroutable name — that is a \
malformed call, not a failed task and not a refusal. Re-read the parameter types \
in the tool definition, re-issue the call with the corrected shape, and carry \
on. Never re-send a shape that was just rejected, and never answer the user \
instead of retrying a call that was rejected for its form.

Distinguish what the code currently does, what the task requires, and what your \
validation has actually established. A symptom disappearing is not proof that \
you fixed the cause. If an assumption about architecture, ownership, ABI, \
lifetime, authority, ordering, or another critical contract is false, stop that \
implementation path and explain the verified conflict rather than silently \
weakening the requirement.

Run the project's normal validation before claiming completion. Do not claim \
completion until the requested implementation and the required validation are \
both complete. Never say a test passed unless you ran it and saw it pass. When \
you cannot validate something, say exactly what remains unverified.

Do not commit or push unless the user explicitly asks.

When you finish, report what changed, the decisions that matter, what you \
actually ran, and what remains uncertain. Be concise and concrete."""


# The completion/persistence semantics restored from the predecessor's known-good
# prompt, each keyed by the id used in the parity analysis. A test asserts every
# one of these survives into the *rendered* instructions, so a future rewrite of
# the prose above cannot drop one silently the way the last rewrite did.
#
# These are semantic anchors, not the full sentences: they must stay short enough
# that legitimate rewording does not break the test for no reason, and specific
# enough that deleting the semantic breaks it.
PARITY_CLAUSES: dict[str, str] = {
    "M1": "neither is the breadth of the scope the user asked for",
    "M2": "reconstruct the relevant current state",
    "M4": "stop that implementation path and explain the verified conflict",
    "M5": "A valid early stop requires a concrete blocker",
    "W1": "Do not stop merely because substantial work remains",
    "W2": "Do not claim completion until the requested implementation",
}

# Which files the model may write, as opposed to which it may read. Reading
# outside the project is often how a task gets understood; writing outside it is
# a different act and needs the user to have asked for it.
#
# This states the invariant to the model. It is not an enforcement mechanism --
# the sandbox is the client's, and this server never sees a filesystem
# operation. Do not read a green test here as evidence that an external write is
# prevented.
WRITE_BOUNDARY_CLAUSE = "The current project is the writable boundary."

# Deliberately absent -- see the note above `DEFAULT_BASE_INSTRUCTIONS`. Pinned so
# that restoring it becomes a decision someone has to make on purpose.
WITHHELD_CLAUSES: dict[str, str] = {
    "M3": "Do not spend excessive time describing a plan instead of executing it",
}


def _reasoning_levels(model: ServedModel) -> list[dict[str, str]]:
    return [
        {"effort": effort.value, "description": _EFFORT_DESCRIPTIONS[effort]}
        for effort in model.reasoning_efforts
        if effort in _EFFORT_DESCRIPTIONS
    ]


def build_model_info(model: ServedModel) -> dict[str, Any]:
    """One ``ModelInfo`` entry, describing what this server can really do."""
    return {
        "slug": model.slug,
        "display_name": model.display_name,
        # Required key. Null is fine; absent is not.
        "description": None,
        "default_reasoning_level": model.default_reasoning_effort.value,
        "supported_reasoning_levels": _reasoning_levels(model),
        # `disabled` makes Codex send no shell tool. While this server cannot
        # route a tool call back to the client, advertising a shell and then
        # rejecting the request that carries it would fail every turn.
        "shell_type": model.codex_shell_type,
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        # Required keys with no serde default. Null, but present.
        "availability_nux": None,
        "upgrade": None,
        "default_verbosity": None,
        # Not proven against this backend, so not advertised. Claiming a patch
        # tool the server has never round-tripped would be exactly the
        # "accepted is not implemented" mistake.
        "apply_patch_tool_type": None,
        "base_instructions": model.base_instructions or DEFAULT_BASE_INSTRUCTIONS,
        "support_verbosity": False,
        # Raw reasoning is emitted as reasoning items; no summariser exists here,
        # so the summary parameter is declined rather than accepted and ignored.
        "supports_reasoning_summary_parameter": False,
        "default_reasoning_summary": "none",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        # Harmony's `<|call|>` ends the assistant turn, so one call per turn.
        "supports_parallel_tool_calls": model.supports_parallel_tool_calls,
        "supports_image_detail_original": False,
        "context_window": model.context_window,
        "max_context_window": model.context_window,
        # 100 makes the advertised window match the real KV limit. Codex's
        # default of 95 would report a smaller window than the server enforces.
        "effective_context_window_percent": model.effective_context_percent,
        "experimental_supported_tools": [],
        # Explicit: the omitted default is ["text", "image"], and this server
        # rejects image input.
        "input_modalities": list(model.input_modalities),
        # No provider-side search executor exists here.
        "supports_search_tool": model.supports_hosted_search,
        "use_responses_lite": False,
        "include_apps_usage_instructions": False,
        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
    }


def build_models_response(models: tuple[ServedModel, ...]) -> dict[str, Any]:
    """The ``GET /v1/models`` body."""
    return {"models": [build_model_info(model) for model in models]}
