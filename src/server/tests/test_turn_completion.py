"""Semantic completion classification at the Harmony/Responses boundary.

Two levels, deliberately:

The first classifies already-assembled results, which is where the rule itself
lives. The second -- everything below `-- the real boundary` -- drives the
daemon with **deterministic Harmony tokens**, through the same parser inference
uses, on both the streaming and the non-streaming path. That distinction is the
point: a classifier that agrees with a hand-built `ParsedGeneration` proves
nothing about what a real `analysis … <|return|>` sequence turns into by the
time it reaches a client.

What none of this establishes is *why* a model would emit that sequence. These
tests pin what QCS does with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from quantum_codex import app as app_module
from quantum_codex.app import _build_result, _completion_error, _record_result
from quantum_codex.canonical import FinishReason, GenerationTiming, ToolCall
from quantum_codex.diagnostics import Diagnostics, Outcome
from quantum_codex.harmony import HarmonyRenderer
from quantum_codex.harmony.render import load_encoding
from quantum_codex.inference.engine import EngineState, GenerationOutcome
from quantum_codex.models import ServedModel


@pytest.fixture(scope="module")
def renderer() -> HarmonyRenderer:
    return HarmonyRenderer()


def stop_id(renderer: HarmonyRenderer, marker: str) -> int:
    return next(
        token
        for token in renderer.stop_tokens
        if renderer.encoding.decode([token]) == marker
    )


def outcome(
    renderer: HarmonyRenderer,
    reason: FinishReason = FinishReason.STOP,
    marker: str | None = "<|return|>",
) -> GenerationOutcome:
    return GenerationOutcome(
        tokens=[1, 2],
        input_tokens=10,
        finish_reason=reason,
        timing=GenerationTiming(prefill_seconds=0.1, decode_seconds=0.2),
        stop_token_id=stop_id(renderer, marker) if marker else None,
    )


def classify(
    renderer: HarmonyRenderer,
    *,
    text: str = "",
    reasoning: tuple[str, ...] = (),
    tool_calls: tuple[ToolCall, ...] = (),
    reason: FinishReason = FinishReason.STOP,
    marker: str | None = "<|return|>",
):
    raw = outcome(renderer, reason, marker)
    result = _build_result(renderer, text, reasoning, tool_calls, raw)
    error = _completion_error(result, raw)
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="req_shape", model="test")
    _record_result(
        diagnostics,
        record,
        result,
        raw,
        renderer=renderer,
        completion_error=error,
    )
    return record, error


def test_a_normal_final_answer_completes_without_recovery(renderer: HarmonyRenderer) -> None:
    record, error = classify(renderer, text="Done.", reasoning=("Checked.",))

    assert error is None
    assert record.outcome is Outcome.COMPLETED
    assert record.had_reasoning is True
    assert record.had_final_output is True
    assert record.had_tool_call is False
    assert record.empty_completion_detected is False
    assert record.recovery_attempted is False
    assert record.recovery_outcome is None
    assert record.terminal_token_class == "harmony_return"


def test_a_tool_handoff_completes_without_recovery(renderer: HarmonyRenderer) -> None:
    call = ToolCall(call_id="call_1", name="exec_command", arguments="{}")
    record, error = classify(
        renderer,
        reasoning=("I need evidence.",),
        tool_calls=(call,),
        marker="<|call|>",
    )

    assert error is None
    assert record.outcome is Outcome.COMPLETED
    assert record.finish_reason == FinishReason.TOOL_CALL.value
    assert record.had_tool_call is True
    assert record.had_final_output is False
    assert record.terminal_token_class == "harmony_call"
    assert record.recovery_attempted is False


def test_reasoning_only_stop_is_failed_instead_of_reported_complete(
    renderer: HarmonyRenderer,
) -> None:
    record, error = classify(renderer, reasoning=("I should call a tool.",))

    assert error is not None
    assert record.outcome is Outcome.FAILED
    assert record.had_reasoning is True
    assert record.had_tool_call is False
    assert record.had_final_output is False
    assert record.empty_completion_detected is True
    assert record.terminal_token_class == "harmony_return"
    assert record.recovery_attempted is False
    assert record.recovery_outcome is None


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (FinishReason.LENGTH, Outcome.INCOMPLETE),
        (FinishReason.CANCELLED, Outcome.CANCELLED),
    ],
)
def test_nonterminal_reasoning_only_outcomes_are_not_recovered(
    renderer: HarmonyRenderer, reason: FinishReason, expected: Outcome
) -> None:
    record, error = classify(
        renderer,
        reasoning=("Partial work.",),
        reason=reason,
        marker=None,
    )

    assert error is None
    assert record.outcome is expected
    assert record.empty_completion_detected is False
    assert record.recovery_attempted is False


def test_engine_errors_are_failed_without_recovery() -> None:
    diagnostics = Diagnostics()
    record = diagnostics.begin(request_id="req_error", model="test")

    diagnostics.finish(record, Outcome.FAILED, error="engine failed")

    assert record.outcome is Outcome.FAILED
    assert record.recovery_attempted is False
    assert record.recovery_outcome is None


# -- the real boundary -------------------------------------------------------
#
# Deterministic Harmony text -> the encoding -> the parser inference uses ->
# Responses assembly -> what a client actually receives.


HARMONY_CASES = {
    # A: a thought and then an answer.
    "final": (
        "<|channel|>analysis<|message|>The user greeted me.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello!<|return|>"
    ),
    # B: a thought and then an action.
    "tool_call": (
        "<|channel|>analysis<|message|>I need to look.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.exec_command "
        '<|constrain|>json<|message|>{"cmd":"ls"}<|call|>'
    ),
    # C: the observed incident. A thought, a terminator, and nothing else.
    "reasoning_only": "<|channel|>analysis<|message|>I should call a tool.<|return|>",
    # C': the same shape with nothing at all in it.
    "silent": "<|channel|>final<|message|><|return|>",
    # E: stopped by the output limit, mid-message.
    "truncated": (
        "<|channel|>analysis<|message|>Working on it.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Partial ans"
    ),
    # G: a mis-sampled header. Observed in a real session as
    # `unexpected tokens remaining in message header: Some("to=functions.exec")`,
    # which the strict parser raises on `<|message|>` -- after the reasoning has
    # already streamed. Repairable: the fragments name one recipient.
    "malformed_header": (
        "<|channel|>analysis<|message|>I need to look.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.exec_command "
        'to=functions.exec_command<|message|>{"cmd":"ls"}<|call|>'
    ),
    # G': a header failure that is not repairable, reached with an item already
    # open. Not a header at all: `<|channel|>` where `<|start|>` is required.
    "unrecoverable": (
        "<|channel|>final<|message|>Partial ans<|end|><|channel|>final<|message|>more"
    ),
    # H: a body carrying a control token the model wrote as text. `<|constrain|>`
    # survives inside a message and reaches the reasoning verbatim, where
    # re-encoding it to count tokens used to raise after the turn had finished.
    "control_text_in_reasoning": (
        "<|channel|>analysis<|message|>I will emit <|constrain|>json now.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    ),
    # I: the observed incident -- the generation began with prose and never
    # opened a message, so reasoning, a terminator and a real tool-call header
    # all landed in one header.
    "never_opened": (
        " after 2 line changes this is still incorrect. Now a better approach:\n\n"
        "<|end|><|start|>assistant to=functions.exec_command <|constrain|>json"
        '<|message|>{"cmd":"ls"}<|call|>'
    ),
    # J: a call Harmony accepts with no channel at all -- it parses, then routes
    # nowhere, so the whole call used to vanish in silence.
    "channel_less_call": ' to=functions.exec_command<|message|>{"cmd":"ls"}<|call|>',
    # J': prose with its first words swallowed into the recipient/content-type
    # fields. Parses, routes nowhere, whole message used to vanish.
    "channel_less_prose": (
        "Sure thing<|message|> — I looked and everything is fine.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>All good.<|return|>"
    ),
    # K: the third live incident. Role, prose, a boundary, a noise word, the
    # recipient, a content type. The turn died on the noise word `!~~~!css`
    # sitting where an author would go, and took the `update_plan` call with it.
    "update_plan_incident": (
        "assistant based on the earlier instructions, let me summarise:"
        "<|end|>!~~~!css to=functions.update_plan json"
        '<|message|>{"summary":"The race condition bug has been fully fixed."}<|call|>'
    ),
    # G'': the same failure, reached while a tool call is still being written.
    "unrecoverable_mid_call": (
        "<|channel|>commentary to=functions.exec_command <|constrain|>json"
        '<|message|>{"cmd":"rm -rf /tmp/x"<|end|><|channel|>final<|message|>more'
    ),
}


def harmony_tokens(case: str) -> list[int]:
    return load_encoding().encode(HARMONY_CASES[case], allowed_special="all")


def stop_token(marker: str) -> int:
    encoding = load_encoding()
    return encoding.encode(marker, allowed_special="all")[0]


@dataclass
class Loaded:
    served_name: str
    quantization: str = "mxfp4-4bit"
    context_length: int = 131072
    adapter: object | None = None


class ScriptedEngine:
    """Replays one fixed completion, on either path.

    It stands in for MLX and for the model, and for nothing else: the tokens it
    yields are real Harmony, and every layer above it -- the parser, the item
    assembly, the classification, the SSE encoder -- is the production one. What
    these tests therefore do *not* cover is sampling and the stop-token loop
    inside the engine itself.
    """

    def __init__(self) -> None:
        self.state = EngineState.UNLOADED
        self.load_elapsed_seconds = None
        self.tokens: list[int] = []
        self.finish_reason = FinishReason.STOP
        self.stop_marker: str | None = "<|return|>"
        self.explode: str | None = None

    async def load(self, path, served_name, context_length, *, adapter_path=None):  # noqa: ANN001
        self.state = EngineState.READY
        return Loaded(served_name=served_name, context_length=context_length)

    async def unload(self) -> None:
        self.state = EngineState.UNLOADED

    def shutdown(self) -> None:
        return None

    def _outcome(self, prompt_tokens) -> GenerationOutcome:  # noqa: ANN001
        return GenerationOutcome(
            tokens=list(self.tokens),
            input_tokens=len(prompt_tokens),
            finish_reason=self.finish_reason,
            timing=GenerationTiming(prefill_seconds=0.01, decode_seconds=0.01),
            stop_token_id=(
                stop_token(self.stop_marker) if self.stop_marker is not None else None
            ),
        )

    async def generate(self, prompt_tokens, **kwargs):  # noqa: ANN001, ARG002
        if self.explode:
            raise RuntimeError(self.explode)
        return self._outcome(prompt_tokens)

    async def generate_stream(self, prompt_tokens, **kwargs):  # noqa: ANN001, ARG002
        if self.explode:
            raise RuntimeError(self.explode)
        for token in self.tokens:
            yield token
        yield self._outcome(prompt_tokens)


MODEL = ServedModel(
    slug="gpt-oss-20b",
    display_name="GPT-OSS 20B",
    context_window=131072,
    library_id="gpt-oss-20b",
    path="/models/gpt-oss-20b",
    quantization="mxfp4-4bit",
)


@pytest.fixture
def engine(monkeypatch) -> ScriptedEngine:
    scripted = ScriptedEngine()
    monkeypatch.setattr(app_module, "MlxEngine", lambda **_: scripted)
    return scripted


@pytest.fixture
def daemon(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTUM_CODEX_HOME", str(tmp_path / "home"))
    app = app_module.create_app(host="127.0.0.1", port=8123)
    with TestClient(app, raise_server_exceptions=False) as client:
        # Set after startup: the library on disk is empty, and what is being
        # tested is the completion boundary rather than model resolution.
        client.app.state.context.registry.replace_all([MODEL])
        yield client


def ask(client, *, stream: bool):
    return client.post(
        "/v1/responses",
        json={"model": "gpt-oss-20b", "input": "hello", "stream": stream},
    )


def events(response) -> list[dict]:
    frames = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                frames.append(json.loads(line.removeprefix("data: ")))
    return frames


def latest(client):
    return client.app.state.context.diagnostics.recent(1)[0]


# -- A: reasoning then a final answer ----------------------------------------


def test_a_real_final_answer_completes_on_both_paths(daemon, engine) -> None:
    engine.tokens = harmony_tokens("final")

    plain = ask(daemon, stream=False)
    assert plain.status_code == 200, plain.text
    assert plain.json()["output"][-1]["content"][0]["text"] == "Hello!"
    record = latest(daemon)
    assert record.outcome is Outcome.COMPLETED
    assert (record.had_reasoning, record.had_final_output, record.had_tool_call) == (
        True,
        True,
        False,
    )
    assert record.empty_completion_detected is False
    assert record.terminal_token_class == "harmony_return"

    streamed = ask(daemon, stream=True)
    kinds = [event["type"] for event in events(streamed)]
    assert "response.completed" in kinds
    assert "response.failed" not in kinds
    assert latest(daemon).outcome is Outcome.COMPLETED


# -- B: reasoning then a tool call -------------------------------------------


def test_a_real_tool_call_completes_on_both_paths(daemon, engine) -> None:
    engine.tokens = harmony_tokens("tool_call")
    engine.stop_marker = "<|call|>"

    plain = ask(daemon, stream=False)
    assert plain.status_code == 200, plain.text
    call = next(item for item in plain.json()["output"] if item["type"] == "function_call")
    assert call["name"] == "exec_command"
    record = latest(daemon)
    assert record.outcome is Outcome.COMPLETED
    assert record.had_tool_call is True
    assert record.had_final_output is False
    assert record.empty_completion_detected is False
    assert record.terminal_token_class == "harmony_call"

    streamed = ask(daemon, stream=True)
    kinds = [event["type"] for event in events(streamed)]
    assert "response.completed" in kinds
    assert "response.failed" not in kinds
    assert latest(daemon).had_tool_call is True


# -- C: the observed incident ------------------------------------------------


@pytest.mark.parametrize("case", ["reasoning_only", "silent"])
def test_a_content_free_stop_is_never_reported_as_a_completed_turn(
    daemon, engine, case: str
) -> None:
    """`analysis … <|return|>` with no answer and no call, through the parser."""
    engine.tokens = harmony_tokens(case)

    plain = ask(daemon, stream=False)

    assert plain.status_code == 500
    assert plain.json()["error"]["code"] == app_module.EMPTY_COMPLETION_CODE
    record = latest(daemon)
    assert record.outcome is Outcome.FAILED
    assert record.empty_completion_detected is True
    assert record.had_tool_call is False
    assert record.had_final_output is False
    assert record.had_reasoning is (case == "reasoning_only")
    assert record.terminal_token_class == "harmony_return"
    # Nothing is retried or continued on the model's behalf.
    assert record.recovery_attempted is False
    assert record.recovery_outcome is None


@pytest.mark.parametrize("case", ["reasoning_only", "silent"])
def test_the_streamed_form_of_the_same_turn_fails_rather_than_completing(
    daemon, engine, case: str
) -> None:
    engine.tokens = harmony_tokens(case)

    streamed = ask(daemon, stream=True)

    frames = events(streamed)
    kinds = [event["type"] for event in frames]
    assert "response.completed" not in kinds
    failure = next(event for event in frames if event["type"] == "response.failed")
    assert failure["response"]["error"]["code"] == app_module.EMPTY_COMPLETION_CODE
    record = latest(daemon)
    assert record.outcome is Outcome.FAILED
    assert record.empty_completion_detected is True
    assert record.recovery_attempted is False


# -- D, E, F: the terminal conditions that already mean something -------------


def test_a_cancelled_turn_is_cancelled_rather_than_failed(daemon, engine) -> None:
    """A disconnect is not a semantic failure, and not something to recover."""
    engine.tokens = harmony_tokens("reasoning_only")
    engine.finish_reason = FinishReason.CANCELLED
    engine.stop_marker = None

    plain = ask(daemon, stream=False)

    assert plain.status_code == 200, plain.text
    record = latest(daemon)
    assert record.outcome is Outcome.CANCELLED
    assert record.empty_completion_detected is False
    assert record.terminal_token_class is None
    assert record.recovery_attempted is False


def test_hitting_the_output_limit_is_reported_as_incomplete(daemon, engine) -> None:
    engine.tokens = harmony_tokens("truncated")
    engine.finish_reason = FinishReason.LENGTH
    engine.stop_marker = None

    plain = ask(daemon, stream=False)

    assert plain.status_code == 200, plain.text
    assert plain.json()["incomplete_details"]["reason"] == "max_output_tokens"
    record = latest(daemon)
    assert record.outcome is Outcome.INCOMPLETE
    assert record.empty_completion_detected is False
    assert record.had_final_output is True


def test_a_generation_error_is_failed_and_not_an_empty_completion(daemon, engine) -> None:
    engine.explode = "worker died"

    plain = ask(daemon, stream=False)
    assert plain.status_code >= 500
    record = latest(daemon)
    assert record.outcome is Outcome.FAILED
    assert record.empty_completion_detected is False

    streamed = ask(daemon, stream=True)
    failure = next(event for event in events(streamed) if event["type"] == "response.failed")
    assert failure["response"]["error"]["code"] != app_module.EMPTY_COMPLETION_CODE
    assert latest(daemon).empty_completion_detected is False


# -- what the diagnostics may say --------------------------------------------


def test_the_record_of_a_rejected_turn_carries_no_model_output(daemon, engine) -> None:
    """Shape and terminal class only. Never a word the model produced."""
    engine.tokens = harmony_tokens("reasoning_only")

    ask(daemon, stream=False)

    payload = json.dumps(latest(daemon).as_dict())
    assert "I should call a tool" not in payload
    assert "hello" not in payload
    for key in (
        "terminal_token_class",
        "had_reasoning",
        "had_tool_call",
        "had_final_output",
        "empty_completion_detected",
    ):
        assert key in payload




# -- non-conformant generation, on both paths --------------------------------
#
# Each of these is a real emission observed against this server. They were once
# repaired; a repaired turn ends as COMPLETED, which hides the defect from
# whoever is judging the model. On a native Codex backend they are reported.


#: Shapes that must be refused. `unrecoverable_mid_call` is here too: it dies
#: while a tool call is open, which is the case that must not announce the call
#: as complete on the way out.
MALFORMED_CASES = [
    "malformed_header",
    "never_opened",
    "channel_less_call",
    "channel_less_prose",
    "update_plan_incident",
    "unrecoverable",
    "unrecoverable_mid_call",
]


@pytest.mark.parametrize("case", MALFORMED_CASES)
def test_non_conformant_generation_is_reported_not_repaired(daemon, engine, case: str) -> None:
    """A stable code on both paths, and never an unhandled 500.

    Before this, the non-streaming path let the parser's own error escape as a
    500 with a traceback, and the streaming path put the raw Rust message on
    the wire. Neither told a client anything it could act on.
    """
    engine.tokens = harmony_tokens(case)

    plain = ask(daemon, stream=False)
    assert plain.status_code != 500, plain.text
    assert plain.json()["error"]["code"] == app_module.MALFORMED_GENERATION_CODE
    # The model's broken output is a server-side diagnostic, not wire content.
    assert plain.json()["error"]["message"] == app_module.MALFORMED_GENERATION
    record = latest(daemon)
    assert record.outcome is Outcome.FAILED
    assert record.malformed_generation, "the shape must be recorded to be countable"

    streamed = ask(daemon, stream=True)
    frames = events(streamed)
    kinds = [event["type"] for event in frames]
    assert "response.completed" not in kinds
    failure = next(event for event in frames if event["type"] == "response.failed")
    assert failure["response"]["error"]["code"] == app_module.MALFORMED_GENERATION_CODE
    assert failure["response"]["error"]["message"] == app_module.MALFORMED_GENERATION
    assert latest(daemon).malformed_generation


def test_the_offending_header_is_recorded_but_never_sent(daemon, engine) -> None:
    """The header is what diagnoses the emission; it belongs in the record.

    Every incident this session was identified from that text alone -- and none
    of it belongs on the wire, where it would only hand a client the model's
    own broken output.
    """
    engine.tokens = harmony_tokens("update_plan_incident")

    streamed = ask(daemon, stream=True)

    record = latest(daemon)
    assert record.malformed_header
    assert "update_plan" in record.malformed_header
    assert "update_plan" not in streamed.text


def test_reasoning_that_parsed_before_the_malformation_still_reaches_the_client(
    daemon, engine
) -> None:
    """Strict is not the same as discarding what was already valid."""
    engine.tokens = harmony_tokens("malformed_header")

    streamed = ask(daemon, stream=True)

    frames = events(streamed)
    done = [event["item"] for event in frames if event["type"] == "response.output_item.done"]
    assert [item["type"] for item in done] == ["reasoning"]
    assert "I need to look." in done[0]["content"][0]["text"]
    # An unfinished tool call is never announced as ready to dispatch.
    assert all(item["type"] != "function_call" for item in done)


def test_control_token_text_in_reasoning_still_completes(daemon, engine) -> None:
    """Legal Harmony that merely contains `<|…|>` as *text* is not malformed.

    The model writes literal control-token text into its own reasoning -- seen
    in the rollouts quoting its own tool-call syntax. Harmony parses it; only
    the server's re-encoding used to fail, after the turn had already produced
    its answer. That fix is kept, and strictness must not swallow it.
    """
    engine.tokens = harmony_tokens("control_text_in_reasoning")

    plain = ask(daemon, stream=False)
    assert plain.status_code == 200, plain.text
    assert plain.json()["output"][-1]["content"][0]["text"] == "Done."
    assert latest(daemon).outcome is Outcome.COMPLETED
    assert latest(daemon).malformed_generation is None

    streamed = ask(daemon, stream=True)
    assert "response.completed" in [event["type"] for event in events(streamed)]
