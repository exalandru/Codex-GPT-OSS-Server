"""Context accounting and overflow behaviour.

The load-bearing property here is that overflow is *refused*, not absorbed. A
rotating KV cache would keep answering while dropping the oldest tokens, which
looks like success until the model contradicts something it can no longer see.
"""

from __future__ import annotations

import pytest

from quantum_codex.api.errors import ApiError
from quantum_codex.app import DEFAULT_MAX_OUTPUT_TOKENS, _resolve_max_output


def test_unrequested_limit_defaults_but_stays_inside_the_window() -> None:
    assert _resolve_max_output(requested=None, prompt_length=10, context_window=100_000) == (
        DEFAULT_MAX_OUTPUT_TOKENS
    )
    # Near the end of the window the default must shrink rather than promise
    # tokens there is no room for.
    assert _resolve_max_output(requested=None, prompt_length=99_900, context_window=100_000) == 100


def test_requested_limit_is_honoured_when_it_fits() -> None:
    assert _resolve_max_output(requested=500, prompt_length=1_000, context_window=10_000) == 500


def test_a_prompt_larger_than_the_window_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        _resolve_max_output(requested=None, prompt_length=200_000, context_window=131_072)

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.param == "input"


def test_a_prompt_exactly_filling_the_window_leaves_no_room_to_answer() -> None:
    # Boundary: the prompt fits, but zero tokens remain. Generating nothing and
    # reporting success would be worse than saying so.
    with pytest.raises(ApiError) as caught:
        _resolve_max_output(requested=None, prompt_length=1_000, context_window=1_000)

    assert caught.value.code == "context_length_exceeded"


def test_an_output_limit_that_would_overflow_is_refused() -> None:
    with pytest.raises(ApiError) as caught:
        _resolve_max_output(requested=5_000, prompt_length=8_000, context_window=10_000)

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.param == "max_output_tokens"


def test_an_output_limit_exactly_filling_the_window_is_allowed() -> None:
    assert _resolve_max_output(requested=2_000, prompt_length=8_000, context_window=10_000) == 2_000
