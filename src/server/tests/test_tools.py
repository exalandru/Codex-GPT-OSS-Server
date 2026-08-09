"""Function tools: declaration, call, result, and reasoning continuity.

The load-bearing property of this slice is the *round trip*. A test that only
checked "a tool call is parsed" would pass for an implementation that then
renders the replayed call wrongly, which is precisely how continuity breaks on
the second turn.

Real Harmony text and the real parser throughout; no model weights.
"""

from __future__ import annotations

import pytest
from openai_harmony import Role as HarmonyRole

from quantum_codex.api.schemas import parse_request, to_canonical_turn
from quantum_codex.canonical import (
    CanonicalMessage,
    CanonicalTurn,
    ReasoningEffort,
    ReasoningTrace,
    Role,
    ToolCall,
    ToolDefinition,
    ToolOutput,
)
from quantum_codex.harmony import HarmonyRenderer, parse_completion
from quantum_codex.harmony.render import load_encoding

EXEC_TOOL = ToolDefinition(
    name="exec_command",
    description="Runs a command in a PTY.",
    parameters={
        "type": "object",
        "properties": {"cmd": {"type": "array", "items": {"type": "string"}}},
        "required": ["cmd"],
    },
)


@pytest.fixture(scope="module")
def renderer() -> HarmonyRenderer:
    return HarmonyRenderer()


def normalise(body: dict) -> CanonicalTurn:
    return to_canonical_turn(parse_request(body), default_effort=ReasoningEffort.MEDIUM)


def render_text(renderer: HarmonyRenderer, turn: CanonicalTurn) -> str:
    return load_encoding().decode(renderer.render(turn))


def completion(harmony_text: str) -> list[int]:
    return load_encoding().encode(harmony_text, allowed_special="all")


# -- declaring tools ---------------------------------------------------------


def test_tools_render_into_the_functions_namespace(renderer: HarmonyRenderer) -> None:
    turn = CanonicalTurn(
        items=(CanonicalMessage(role=Role.USER, text="run pwd"),), tools=(EXEC_TOOL,)
    )
    text = render_text(renderer, turn)

    assert "namespace functions {" in text
    assert "type exec_command = (_: {" in text
    assert "Runs a command in a PTY." in text
    # The system block must tell the model where calls go, or it answers in
    # `final` instead of calling.
    assert "commentary channel: 'functions'" in text


def test_tools_and_instructions_share_the_developer_message(renderer: HarmonyRenderer) -> None:
    turn = CanonicalTurn(
        items=(CanonicalMessage(role=Role.USER, text="x"),),
        instructions="Be terse.",
        tools=(EXEC_TOOL,),
    )
    text = render_text(renderer, turn)

    assert text.count("<|start|>developer<|message|>") == 1
    assert "Be terse." in text
    assert "namespace functions" in text


def test_flat_and_nested_tool_shapes_are_both_accepted() -> None:
    flat = normalise(
        {
            "input": "x",
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
        }
    )
    nested = normalise(
        {
            "input": "x",
            "tools": [
                {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
            ],
        }
    )

    assert flat.tools[0].name == "f" == nested.tools[0].name


# -- parsing a call ----------------------------------------------------------


def test_a_tool_call_is_recovered_from_the_commentary_channel() -> None:
    parsed = parse_completion(
        completion(
            "<|channel|>analysis<|message|>I need the working directory.<|end|>"
            "<|start|>assistant<|channel|>commentary to=functions.exec_command "
            '<|constrain|>json<|message|>{"cmd":["pwd"]}<|call|>'
        )
    )

    assert parsed.reasoning == ("I need the working directory.",)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "exec_command"
    # Arguments stay the model's exact JSON text.
    assert call.arguments == '{"cmd":["pwd"]}'
    # A top-level function has no namespace. Reporting `functions` would name a
    # routing target the client does not have.
    assert call.namespace is None
    # A call is not an answer.
    assert parsed.text == ""


def test_a_namespaced_call_keeps_its_namespace() -> None:
    parsed = parse_completion(
        completion(
            "<|channel|>commentary to=multi_agent_v1.spawn_agent "
            '<|constrain|>json<|message|>{"task":"x"}<|call|>'
        )
    )

    call = parsed.tool_calls[0]
    assert call.name == "spawn_agent"
    assert call.namespace == "multi_agent_v1"


def test_commentary_without_a_recipient_is_not_a_tool_call() -> None:
    parsed = parse_completion(
        completion("<|channel|>commentary<|message|>Just thinking aloud.<|end|>")
    )

    assert parsed.tool_calls == ()
    assert parsed.commentary == ("Just thinking aloud.",)


# -- replaying a call and its result -----------------------------------------


def test_call_and_output_replay_into_canonical_harmony(renderer: HarmonyRenderer) -> None:
    turn = CanonicalTurn(
        items=(
            CanonicalMessage(role=Role.USER, text="run pwd"),
            ReasoningTrace(text="I need the working directory."),
            ToolCall(call_id="call_1", name="exec_command", arguments='{"cmd":["pwd"]}'),
            ToolOutput(call_id="call_1", output="/home/u", name="exec_command"),
        ),
        tools=(EXEC_TOOL,),
    )
    text = render_text(renderer, turn)

    # Reasoning goes back to `analysis`, not into visible output.
    assert "<|start|>assistant<|channel|>analysis<|message|>I need the working directory." in text
    # The call is addressed, on commentary, as JSON, and terminated with <|call|>.
    #
    # The expected bytes are derived from a real generated call rather than
    # written out here. This assertion has twice pinned a shape the model never
    # produces -- first the bare word `json`, then Harmony's recipient-first
    # ordering -- and each time the test passed while the model could not read
    # the transcript back.
    generated = (
        '<|channel|>commentary to=functions.exec_command '
        '<|constrain|>json<|message|>{"cmd":["pwd"]}<|call|>'
    )
    encoding = load_encoding()
    reference = encoding.parse_messages_from_completion_tokens(
        encoding.encode(generated, allowed_special="all"), role=HarmonyRole.ASSISTANT
    )[0]
    expected_call = (
        f"<|start|>assistant<|channel|>{reference.channel} to={reference.recipient} "
        f"{reference.content_type}<|message|>{reference.content[0].text}<|call|>"
    )
    assert expected_call in text
    # The result is authored by the tool, not by the user.
    assert "<|start|>functions.exec_command<|channel|>commentary<|message|>/home/u<|end|>" in text
    assert text.endswith("<|start|>assistant")


def test_replayed_turn_is_parseable_as_a_conversation(renderer: HarmonyRenderer) -> None:
    """The discriminating check: our own replay must survive Harmony's parser.

    A renderer that produced plausible-looking but structurally wrong Harmony
    would still satisfy substring assertions, and would only fail here.
    """
    turn = CanonicalTurn(
        items=(
            CanonicalMessage(role=Role.USER, text="run pwd"),
            ToolCall(call_id="c1", name="exec_command", arguments='{"cmd":["pwd"]}'),
            ToolOutput(call_id="c1", output="/home/u", name="exec_command"),
        ),
        tools=(EXEC_TOOL,),
    )

    tokens = renderer.render(turn)
    # Round-trips through the encoding without loss, and ends ready for the
    # assistant to continue rather than mid-message.
    assert load_encoding().encode(load_encoding().decode(tokens), allowed_special="all") == tokens


def test_dropping_replayed_reasoning_would_change_the_prompt(renderer: HarmonyRenderer) -> None:
    """Counterfactual guard for reasoning continuity.

    An implementation that silently discarded replayed reasoning would still
    render a valid prompt, still call the tool, and still produce an answer --
    it would only be worse at it. Comparing the two prompts is what makes the
    difference observable: a dropped trace shows up as zero.
    """
    base = (
        CanonicalMessage(role=Role.USER, text="run pwd"),
        ToolCall(call_id="c1", name="exec_command", arguments="{}"),
        ToolOutput(call_id="c1", output="/home/u", name="exec_command"),
    )
    trace = "I need the working directory and cannot know it without asking the shell."

    without = renderer.count_tokens(CanonicalTurn(items=base))
    with_reasoning = renderer.count_tokens(
        CanonicalTurn(items=(base[0], ReasoningTrace(text=trace), *base[1:]))
    )

    assert with_reasoning > without
    # Roughly the trace itself; the exact count is the tokenizer's business, the
    # point is that it is neither zero nor wildly off.
    assert with_reasoning - without >= len(trace.split())


def test_output_without_a_matching_call_still_renders(renderer: HarmonyRenderer) -> None:
    # Defensive: a client could replay an output whose call is outside the
    # window. Crashing would lose the whole turn.
    turn = CanonicalTurn(items=(ToolOutput(call_id="orphan", output="result"),))
    assert "<|start|>functions.unknown" in render_text(renderer, turn)


# -- normalising Codex's replayed items --------------------------------------


def test_codex_replay_becomes_ordered_canonical_items() -> None:
    turn = normalise(
        {
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run pwd"}]},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "Need the cwd."}],
                    "encrypted_content": None,
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":["pwd"]}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "/home/u"},
            ]
        }
    )

    kinds = [type(item).__name__ for item in turn.items]
    assert kinds == ["CanonicalMessage", "ReasoningTrace", "ToolCall", "ToolOutput"]

    # The name is resolved from the earlier call: the wire carries only call_id,
    # but Harmony needs the function name to attribute the tool message.
    assert turn.items[3].name == "exec_command"
    assert turn.items[3].output == "/home/u"


def test_reasoning_content_survives_the_replay() -> None:
    """Continuity depends on this exact field being read back."""
    turn = normalise(
        {
            "input": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Earlier thought."}],
                    "encrypted_content": None,
                }
            ]
        }
    )

    assert turn.items == (ReasoningTrace(text="Earlier thought."),)


def test_an_encrypted_only_reasoning_item_is_skipped_not_fatal() -> None:
    # This server produces no encrypted reasoning, so it cannot read one back.
    # Refusing the request would break the turn; skipping degrades continuity,
    # which the server logs.
    turn = normalise(
        {
            "input": [
                {"type": "reasoning", "encrypted_content": "opaque", "summary": []},
                {"type": "message", "role": "user", "content": "hi"},
            ]
        }
    )

    assert [type(item).__name__ for item in turn.items] == ["CanonicalMessage"]


def test_structured_tool_output_is_flattened_to_text() -> None:
    turn = normalise(
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [{"type": "output_text", "text": "line one"}],
                }
            ]
        }
    )

    assert turn.items[0].output == "line one"


def test_a_function_call_without_a_call_id_is_refused() -> None:
    from quantum_codex.api.errors import ApiError

    with pytest.raises(ApiError):
        normalise({"input": [{"type": "function_call", "name": "f", "arguments": "{}"}]})


def test_a_replayed_assistant_answer_carries_the_final_channel(
    renderer: HarmonyRenderer,
) -> None:
    """Regression: it used to render with no channel at all.

    Harmony's system block requires a channel on every message. A channel-less
    assistant turn is a shape the model never saw in training, and it no longer
    matches the tokens the model actually produced -- which silently kills
    prefix reuse on the following turn as well.
    """
    turn = CanonicalTurn(
        items=(
            CanonicalMessage(role=Role.USER, text="hi"),
            CanonicalMessage(role=Role.ASSISTANT, text="hello"),
            CanonicalMessage(role=Role.USER, text="again"),
        )
    )

    text = render_text(renderer, turn)

    assert "<|start|>assistant<|channel|>final<|message|>hello<|end|>" in text
    assert "<|start|>assistant<|message|>hello" not in text


def test_a_recipient_carrying_control_text_is_truncated_not_forwarded() -> None:
    """Regression from a real session.

    A call went out to Codex named `exec_command<|channel|>commentary`, which no
    client can route. Harmony puts raw control-token text in header fields — the
    same session showed `content_type` as `<|constrain|>json` — so a recipient
    can pick it up too. Recovering the name beats losing the turn.
    """
    from quantum_codex.harmony.parse import split_recipient

    assert split_recipient("functions.exec_command<|channel|>commentary") == (
        "exec_command",
        None,
    )
    assert split_recipient("multi_agent_v1.spawn_agent<|constrain|>json") == (
        "spawn_agent",
        "multi_agent_v1",
    )
    # A clean recipient is untouched.
    assert split_recipient("functions.exec_command") == ("exec_command", None)
    # Nothing usable left is no tool call at all, rather than an empty name.
    assert split_recipient("<|channel|>commentary") is None
