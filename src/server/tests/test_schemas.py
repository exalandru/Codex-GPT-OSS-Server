"""Request normalisation into the IR, and the refusals along the way."""

from __future__ import annotations

import pytest

from quantum_codex.api.errors import ApiError
from quantum_codex.api.schemas import ResponsesRequest, parse_request, to_canonical_turn
from quantum_codex.canonical import ReasoningEffort, Role

MEDIUM = ReasoningEffort.MEDIUM


def normalise(body: dict, *, default_effort: ReasoningEffort = MEDIUM):
    return to_canonical_turn(parse_request(body), default_effort=default_effort)


def test_string_input_becomes_a_user_message() -> None:
    turn = normalise({"input": "hello"})

    assert len(turn.items) == 1
    assert turn.items[0].role is Role.USER
    assert turn.items[0].text == "hello"


def test_message_items_keep_their_order_and_roles() -> None:
    turn = normalise(
        {
            "input": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ]
        }
    )

    assert [(m.role, m.text) for m in turn.items] == [
        (Role.USER, "first"),
        (Role.ASSISTANT, "second"),
        (Role.USER, "third"),
    ]


def test_content_parts_are_concatenated() -> None:
    turn = normalise(
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "part one "},
                        {"type": "input_text", "text": "part two"},
                    ],
                }
            ]
        }
    )

    assert turn.items[0].text == "part one part two"


def test_output_text_parts_are_accepted_on_replayed_turns() -> None:
    # A replayed assistant turn uses `output_text` where a user turn uses
    # `input_text`. Rejecting it would break every multi-turn conversation.
    turn = normalise(
        {"input": [{"role": "assistant", "content": [{"type": "output_text", "text": "prior"}]}]}
    )

    assert turn.items[0].text == "prior"


def test_reasoning_effort_overrides_the_model_default() -> None:
    assert normalise({"input": "x"}).reasoning_effort is MEDIUM
    assert normalise({"input": "x", "reasoning": {"effort": "high"}}).reasoning_effort is (
        ReasoningEffort.HIGH
    )


def test_model_default_effort_is_used_when_the_request_is_silent() -> None:
    turn = normalise({"input": "x"}, default_effort=ReasoningEffort.LOW)
    assert turn.reasoning_effort is ReasoningEffort.LOW


def test_instructions_and_sampling_reach_the_turn() -> None:
    turn = normalise(
        {"input": "x", "instructions": "Be brief.", "temperature": 0.4, "top_p": 0.9,
         "max_output_tokens": 128}
    )

    assert turn.instructions == "Be brief."
    assert turn.temperature == 0.4
    assert turn.top_p == 0.9
    assert turn.max_output_tokens == 128


# -- refusals ---------------------------------------------------------------


def test_missing_input_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        normalise({})
    assert caught.value.param == "input"


def test_image_content_is_refused_rather_than_dropped() -> None:
    # Silently skipping a non-text part would answer a question the user did not
    # ask, with no indication that anything was lost.
    with pytest.raises(ApiError) as caught:
        normalise(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_image", "image_url": "http://x/y.png"}],
                    }
                ]
            }
        )
    assert "input_image" in caught.value.message


def test_unknown_input_item_type_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        normalise({"input": [{"type": "some_future_item", "data": {}}]})
    assert caught.value.param == "input[0].type"


def test_unknown_role_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        normalise({"input": [{"role": "tool", "content": "x"}]})
    assert caught.value.param == "input[0].role"


def test_unknown_top_level_fields_are_visible_to_the_caller() -> None:
    # The compatibility layer decides what to do with these; the schema's job is
    # to make sure they are not silently swallowed.
    request = ResponsesRequest.model_validate({"input": "x", "some_new_codex_field": True})
    assert "some_new_codex_field" in request.unknown_fields


# -- schema violations must not become server errors ------------------------


def test_a_reasoning_level_from_another_model_family_is_a_client_error() -> None:
    """Regression: this used to escape as a 500.

    `xhigh` exists for other model families but not for GPT-OSS. Pydantic
    rejects it before any capability check runs, and an unhandled
    ValidationError becomes an unactionable Internal Server Error.
    """
    with pytest.raises(ApiError) as caught:
        parse_request({"input": "x", "reasoning": {"effort": "xhigh"}})

    assert caught.value.status_code == 400
    assert caught.value.param == "reasoning.effort"


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"input": "x", "temperature": 9.0}, "temperature"),
        ({"input": "x", "top_p": 0.0}, "top_p"),
        ({"input": "x", "max_output_tokens": 0}, "max_output_tokens"),
        ({"input": 42}, "input"),
    ],
)
def test_field_violations_come_back_as_400_with_a_param(body: dict, param: str) -> None:
    with pytest.raises(ApiError) as caught:
        parse_request(body)

    assert caught.value.status_code == 400
    assert caught.value.param is not None
    assert caught.value.param.startswith(param)
