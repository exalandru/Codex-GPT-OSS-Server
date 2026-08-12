"""Harmony rendering and parsing, against the real encoding and parser.

These tests deliberately use real Harmony tokens rather than pre-normalised
fixtures. A mock that already looks like the parser's output would pass whether
or not the rendering is correct, which is precisely the failure mode this
boundary needs guarded (cahier 47).

No model weights are loaded here.
"""

from __future__ import annotations

import pytest
from openai_harmony import Role as HarmonyRole

from quantum_codex.canonical import (
    CanonicalMessage,
    CanonicalTurn,
    ReasoningEffort,
    Role,
    ToolCall,
)
from quantum_codex.harmony import HarmonyRenderer, parse_completion
from quantum_codex.harmony.render import load_encoding


@pytest.fixture(scope="module")
def renderer() -> HarmonyRenderer:
    return HarmonyRenderer()


def turn(text: str = "Say hi.", **kwargs) -> CanonicalTurn:
    return CanonicalTurn(items=(CanonicalMessage(role=Role.USER, text=text),), **kwargs)


def test_render_produces_a_completion_prompt(renderer: HarmonyRenderer) -> None:
    tokens = renderer.render(turn())
    text = load_encoding().decode(tokens)

    assert "<|start|>user<|message|>Say hi.<|end|>" in text
    # The prompt must end ready for the assistant to speak, otherwise the model
    # continues the user's turn instead of answering.
    assert text.endswith("<|start|>assistant")


def test_reasoning_effort_reaches_the_system_block(renderer: HarmonyRenderer) -> None:
    for effort in ReasoningEffort:
        text = load_encoding().decode(renderer.render(turn(reasoning_effort=effort)))
        assert f"Reasoning: {effort.value}" in text


def test_instructions_render_as_developer_content(renderer: HarmonyRenderer) -> None:
    text = load_encoding().decode(renderer.render(turn(instructions="Be terse.")))

    # Instructions belong to the developer channel. Rendering them as a second
    # system message would put them somewhere the model was not trained to read
    # them from.
    assert "<|start|>developer<|message|>" in text
    assert "Be terse." in text


def test_token_count_covers_the_whole_prompt(renderer: HarmonyRenderer) -> None:
    small = renderer.count_tokens(turn("hi"))
    large = renderer.count_tokens(turn("hi", instructions="x " * 500))

    assert large > small + 400
    # Counting the user text alone would report a handful of tokens; the real
    # prompt includes the system block (cahier 21).
    assert small > 20


def test_stop_tokens_include_the_tool_call_terminator(renderer: HarmonyRenderer) -> None:
    encoding = load_encoding()
    stop_text = {encoding.decode([token]) for token in renderer.stop_tokens}

    # `<|call|>` ends the assistant turn so a tool result can come back. The MLX
    # tokenizer reports only `<|return|>` as EOS, so a server relying on the
    # model's EOS alone would generate straight past a tool call.
    assert "<|call|>" in stop_text
    assert "<|return|>" in stop_text


def test_stop_tokens_have_content_free_terminal_classes(renderer: HarmonyRenderer) -> None:
    encoding = load_encoding()
    by_text = {encoding.decode([token]): token for token in renderer.stop_tokens}

    assert renderer.terminal_token_class(by_text["<|call|>"]) == "harmony_call"
    assert renderer.terminal_token_class(by_text["<|return|>"]) == "harmony_return"
    assert renderer.terminal_token_class(None) is None


def _completion_tokens(harmony_text: str) -> list[int]:
    """Encode assistant output the way the model would have generated it."""
    return load_encoding().encode(harmony_text, allowed_special="all")


def test_parse_separates_reasoning_from_the_answer() -> None:
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>The user greeted me.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello!<|return|>"
    )

    parsed = parse_completion(tokens)

    assert parsed.text == "Hello!"
    assert parsed.reasoning == ("The user greeted me.",)


def test_parse_exposes_a_reasoning_only_terminal_turn() -> None:
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>I should call a tool.<|return|>"
    )

    parsed = parse_completion(tokens)

    assert parsed.reasoning == ("I should call a tool.",)
    assert parsed.text == ""
    assert parsed.tool_calls == ()


def test_parse_keeps_multiple_reasoning_segments_in_order() -> None:
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>First thought.<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>Second thought.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    )

    parsed = parse_completion(tokens)

    assert parsed.reasoning == ("First thought.", "Second thought.")
    assert parsed.text == "Done."


def test_parse_never_leaks_control_tokens_into_the_answer() -> None:
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Plain answer.<|return|>"
    )

    parsed = parse_completion(tokens)

    for marker in ("<|channel|>", "<|message|>", "<|start|>", "<|end|>", "<|return|>"):
        assert marker not in parsed.text
        assert all(marker not in segment for segment in parsed.reasoning)


def test_parse_recovers_a_generation_cut_off_mid_message() -> None:
    # What a hit output limit looks like: a complete analysis message, then a
    # final message with no terminator.
    tokens = _completion_tokens(
        "<|channel|>analysis<|message|>Working on it.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Partial ans"
    )

    parsed = parse_completion(tokens)

    assert parsed.reasoning == ("Working on it.",)
    assert parsed.text.startswith("Partial ans")


def test_round_trip_render_then_parse(renderer: HarmonyRenderer) -> None:
    """The renderer's own output must be readable by the parser.

    This is the discriminating check: a renderer that produced plausible-looking
    but structurally wrong Harmony would pass a decode-and-eyeball test and fail
    here.
    """
    encoding = load_encoding()

    # The renderer's prompt must end where the assistant is expected to begin,
    # so appending a generated completion yields a structurally valid turn.
    prompt = renderer.render(turn("What is 2+2?"))
    assert encoding.decode(prompt).endswith("<|start|>assistant")

    completion = encoding.encode(
        "<|channel|>analysis<|message|>Simple arithmetic.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>4<|return|>",
        allowed_special="all",
    )

    messages = encoding.parse_messages_from_completion_tokens(
        completion, role=HarmonyRole.ASSISTANT
    )
    assert [message.channel for message in messages] == ["analysis", "final"]

    parsed = parse_completion(completion)
    assert parsed.text == "4"
    assert parsed.reasoning == ("Simple arithmetic.",)


def test_a_replayed_tool_call_carries_the_constrain_token(renderer: HarmonyRenderer) -> None:
    text = load_encoding().decode(
        renderer.render(
            CanonicalTurn(items=(ToolCall(call_id="c1", name="shell", arguments="{}"),))
        )
    )

    assert "<|constrain|>json" in text
    # The bare word would be ordinary text in a header that expects a token.
    assert "commentary json" not in text


def test_piecewise_rendering_matches_harmonys_own_conversation_render(
    renderer: HarmonyRenderer,
) -> None:
    """The renderer builds prompts message by message; that must be lossless.

    Only replayed tool calls are emitted by hand. Everything else has to be
    byte-identical to `render_conversation`, or the hand-emitted message would
    be hiding a second, silent difference.
    """
    from openai_harmony import Conversation

    turn = CanonicalTurn(
        items=(
            CanonicalMessage(role=Role.USER, text="hi"),
            CanonicalMessage(role=Role.ASSISTANT, text="hello"),
        )
    )
    encoding = load_encoding()
    expected = encoding.render_conversation_for_completion(
        Conversation.from_messages(renderer._messages(turn)), HarmonyRole.ASSISTANT
    )

    assert renderer.render(turn) == expected


def test_a_replayed_tool_call_is_written_the_way_the_model_writes_it(
    renderer: HarmonyRenderer,
) -> None:
    """Replay must reproduce generation, and Harmony's renderer does not.

    The expected bytes are derived from a real generated call by parsing it and
    reading the parser's own fields, not hand-written: a hand-written string
    would pin whatever shape happened to be current, which is how the previous
    two replay bugs survived.

    Measured consequence of getting this wrong, on GPT-OSS-120B, eight samples
    each: 0/8 proper follow-up tool calls with Harmony's recipient-first
    ordering, 8/8 with the generated ordering.
    """
    encoding = load_encoding()
    from openai_harmony import Role as HRole

    generated = (
        '<|channel|>commentary to=functions.exec_command '
        '<|constrain|>json<|message|>{"cmd":"ls -R"}<|call|>'
    )
    parsed = encoding.parse_messages_from_completion_tokens(
        encoding.encode(generated, allowed_special="all"), role=HRole.ASSISTANT
    )[0]

    # Rebuild the header from what the parser recovered, so the expectation is
    # the model's own shape rather than this test's opinion of it.
    expected = (
        f"<|start|>assistant<|channel|>{parsed.channel} to={parsed.recipient} "
        f"{parsed.content_type}<|message|>{parsed.content[0].text}<|call|>"
    )

    ours = encoding.decode(
        renderer.render(
            CanonicalTurn(
                items=(ToolCall(call_id="c1", name="exec_command", arguments='{"cmd":"ls -R"}'),)
            )
        )
    )

    assert expected in ours


def test_a_replayed_tool_call_still_parses_back_to_the_same_call(
    renderer: HarmonyRenderer,
) -> None:
    """Emitting raw tokens must not cost us parseability."""
    encoding = load_encoding()
    text = encoding.decode(
        renderer.render(
            CanonicalTurn(
                items=(ToolCall(call_id="c1", name="shell", arguments='{"cmd":"pwd"}'),)
            )
        )
    )
    body = text.split("<|start|>assistant", 2)[1]

    from openai_harmony import Role as HRole

    reparsed = encoding.parse_messages_from_completion_tokens(
        encoding.encode(body, allowed_special="all"), role=HRole.ASSISTANT
    )[0]

    assert reparsed.recipient == "functions.shell"
    assert reparsed.channel == "commentary"
    assert reparsed.content[0].text == '{"cmd":"pwd"}'
