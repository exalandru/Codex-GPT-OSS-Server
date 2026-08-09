"""Prompt cache identity, honesty and budgets.

These tests use a fake KV entry rather than real model state: what is under test
is the *authority* layer -- which entries may be reused, and what gets reported
-- not `mlx_lm`'s trie, which is upstream's.

The hazard this guards is specific. A cache bug does not usually raise; it
serves attention state from the wrong prompt and the answer is merely wrong. So
the assertions are about isolation and about the honesty of `cached_tokens`,
never about "did it get faster".
"""

from __future__ import annotations

import pytest

from quantum_codex.inference.prompt_cache import ModelIdentity, PromptCache


class FakeKV:
    """Stands in for one layer of KV state.

    Only `nbytes` matters here: the session cache never trims, because GPT-OSS's
    sliding-window layers cannot be trimmed past the window (see the module
    under test).
    """

    def __init__(self, nbytes: int = 1024) -> None:
        self.nbytes = nbytes


def identity(name: str = "gpt-oss-20b", generation: int = 1) -> ModelIdentity:
    return ModelIdentity(served_name=name, path=f"/models/{name}", generation=generation)


def cache(**kwargs) -> PromptCache:
    kwargs.setdefault("max_entries", 8)
    kwargs.setdefault("max_bytes", 1 << 30)
    return PromptCache(**kwargs)


# -- identity ----------------------------------------------------------------


def test_a_different_model_never_reuses_entries() -> None:
    """The failure this prevents is silent and severe.

    Serving one model's attention state to another produces a plausible answer
    computed from the wrong weights. No timing test would notice.
    """
    store = cache()
    tokens = [1, 2, 3, 4, 5]
    store.store(identity("gpt-oss-20b"), tokens, [FakeKV()])

    lookup = store.fetch(identity("gpt-oss-120b"), tokens)

    assert lookup.prompt_cache is None
    assert lookup.cached_tokens == 0


def test_reloading_the_same_model_invalidates_its_entries() -> None:
    # The served name is unchanged across a reload, so it cannot be the whole
    # identity: the weights behind it may differ.
    store = cache()
    tokens = [1, 2, 3, 4, 5]
    store.store(identity(generation=1), tokens, [FakeKV()])

    assert store.fetch(identity(generation=2), tokens).prompt_cache is None


def test_identity_equality_is_the_compatibility_rule() -> None:
    assert identity() == identity()
    assert identity() != identity(generation=2)
    assert identity() != identity(name="other")


# -- the client hint is never trusted ----------------------------------------


def test_prompt_cache_key_does_not_create_a_hit() -> None:
    """A key is a claim; the tokens are the evidence.

    Two unrelated prompts sharing a `prompt_cache_key` must not share state --
    trusting the key would serve a cache never verified against the prompt.
    """
    store = cache()
    store.store(identity(), [1, 2, 3, 4, 5], [FakeKV()])

    lookup = store.fetch(identity(), [9, 9, 9, 9, 9], hint="same-key")

    assert lookup.prompt_cache is None
    assert lookup.cached_tokens == 0


def test_a_matching_prefix_hits_without_any_key() -> None:
    # The converse: the tokens are sufficient on their own.
    store = cache()
    store.store(identity(), [1, 2, 3, 4, 5], [FakeKV()])

    lookup = store.fetch(identity(), [1, 2, 3, 4, 5, 6, 7])

    assert lookup.prompt_cache is not None
    assert lookup.cached_tokens == 5
    assert lookup.tokens_to_evaluate == [6, 7]


# -- honest accounting -------------------------------------------------------


def test_cached_tokens_come_from_what_was_actually_reused() -> None:
    store = cache()
    store.store(identity(), [1, 2, 3], [FakeKV()])

    lookup = store.fetch(identity(), [1, 2, 3, 4, 5, 6])

    # Three reused, three to evaluate. The two must add up to the prompt, or the
    # reported usage is describing a different request than the one that ran.
    assert lookup.cached_tokens == 3
    assert len(lookup.tokens_to_evaluate) == 3
    assert lookup.cached_tokens + len(lookup.tokens_to_evaluate) == 6


def test_a_cold_lookup_reports_no_hit() -> None:
    lookup = cache().fetch(identity(), [1, 2, 3])

    assert lookup.prompt_cache is None
    assert lookup.cached_tokens == 0
    assert lookup.hit is False
    assert lookup.tokens_to_evaluate == [1, 2, 3]


def test_a_fully_covered_prompt_runs_cold_rather_than_empty() -> None:
    """Reuse must always leave something to evaluate.

    A session covering the prompt exactly would hand the generation loop an
    empty prompt. Giving a token back would need trimming, which GPT-OSS's
    sliding-window layers do not allow past the window, so the honest outcome is
    a cold run.
    """
    store = cache()
    tokens = [1, 2, 3, 4]
    store.store(identity(), tokens, [FakeKV()])

    lookup = store.fetch(identity(), tokens)

    assert lookup.prompt_cache is None
    assert lookup.tokens_to_evaluate == tokens


def test_counters_track_hits_and_misses() -> None:
    store = cache()
    store.fetch(identity(), [1, 2, 3])  # miss
    store.store(identity(), [1, 2, 3], [FakeKV()])
    store.fetch(identity(), [1, 2, 3, 4])  # hit

    stats = store.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_ratio == 0.5
    assert stats.cached_tokens_total == 3


def test_hit_ratio_is_requests_not_tokens() -> None:
    # One enormous hit must not make nine misses look like a warm cache.
    store = cache()
    for i in range(9):
        store.fetch(identity(), [i, i + 1])
    store.store(identity(), list(range(1000)), [FakeKV()])
    store.fetch(identity(), [*range(1000), 1001])

    assert store.stats().hit_ratio == pytest.approx(0.1)


# -- budgets and administration ----------------------------------------------


def test_entries_are_evicted_past_the_entry_budget() -> None:
    store = cache(max_entries=2)
    for i in range(5):
        store.store(identity(), [i * 100, i * 100 + 1], [FakeKV()])

    assert store.stats().entries <= 2


def test_entries_are_evicted_past_the_byte_budget() -> None:
    store = cache(max_entries=100, max_bytes=4096)
    for i in range(10):
        store.store(identity(), [i * 100, i * 100 + 1], [FakeKV(nbytes=1024)])

    assert store.stats().bytes <= 4096


def test_clearing_drops_entries_but_keeps_the_record() -> None:
    store = cache()
    store.store(identity(), [1, 2, 3], [FakeKV()])
    store.fetch(identity(), [1, 2, 3, 4])

    store.clear()
    stats = store.stats()

    assert stats.entries == 0
    assert stats.bytes == 0
    # Counters are a record of what the server did, not of what it currently
    # holds; resetting them would erase evidence.
    assert stats.hits == 1
    assert store.fetch(identity(), [1, 2, 3, 4]).prompt_cache is None


def test_a_zero_budget_disables_the_cache_entirely() -> None:
    store = cache(max_entries=0)
    store.store(identity(), [1, 2, 3], [FakeKV()])

    assert store.enabled is False
    assert store.fetch(identity(), [1, 2, 3]).cached_tokens == 0
    assert store.stats().entries == 0


def test_per_model_accounting_is_reported() -> None:
    # `mlx_lm.stats_by_type` groups by cache type, not by model.
    store = cache()
    store.store(identity("gpt-oss-20b"), [1, 2, 3], [FakeKV()])
    store.store(identity("gpt-oss-120b"), [4, 5, 6], [FakeKV()])

    assert set(store.stats().by_model) == {"gpt-oss-20b@1", "gpt-oss-120b@1"}


def test_a_layer_without_a_byte_count_does_not_break_accounting() -> None:
    # Byte accounting is operational reporting. An unusual cache layer must
    # degrade the number, never fail the request that produced it.
    store = cache()
    store.store(identity(), [1, 2, 3], [object()])  # no .nbytes

    assert store.stats().entries == 1
    assert store.stats().bytes == 0


def test_a_diverging_prompt_runs_cold_rather_than_resuming() -> None:
    """The cost of the no-trim constraint, made explicit.

    A prompt sharing a long prefix but diverging cannot resume: continuing from
    a diverged state would run the model against attention state that does not
    match the prompt. Slower is acceptable; wrong is not.
    """
    store = cache()
    store.store(identity(), [1, 2, 3, 4, 5, 6], [FakeKV()])

    lookup = store.fetch(identity(), [1, 2, 3, 99, 100])

    assert lookup.prompt_cache is None
    assert lookup.cached_tokens == 0


def test_a_session_advances_in_place_across_turns() -> None:
    """The Codex shape: one conversation, growing.

    Turn N's session must be *reused and extended*, not duplicated -- otherwise
    a long session would evict itself through the entry budget.
    """
    store = cache(max_entries=2)
    turn1 = [1, 2, 3]
    store.store(identity(), turn1, [FakeKV()])

    turn2_prompt = [1, 2, 3, 4, 5]
    lookup = store.fetch(identity(), turn2_prompt)
    assert lookup.cached_tokens == 3
    store.store(identity(), [*turn2_prompt, 6], lookup.prompt_cache)

    turn3_prompt = [1, 2, 3, 4, 5, 6, 7]
    lookup = store.fetch(identity(), turn3_prompt)

    assert lookup.cached_tokens == 6
    assert lookup.tokens_to_evaluate == [7]
    assert store.stats().entries == 1


# -- lifecycle under failure -------------------------------------------------


def test_a_lookup_hands_out_a_copy_not_the_session() -> None:
    """The stored session must survive whatever the caller does to its cache.

    Generation mutates whatever it is handed. If the session lent out its own
    object, a request cancelled mid-prefill would leave a session whose recorded
    tokens no longer describe its contents -- state that looks reusable and is
    silently wrong.
    """
    store = cache()
    original = [FakeKV()]
    store.store(identity(), [1, 2, 3], original)

    lookup = store.fetch(identity(), [1, 2, 3, 4])

    assert lookup.prompt_cache is not original
    assert lookup.prompt_cache[0] is not original[0]


def test_an_abandoned_generation_leaves_the_session_reusable() -> None:
    """Cancellation and disconnect must not cost the session.

    Simulates a request that resumed a session and then died before storing
    anything: the next request must still find the original session intact.
    """
    store = cache()
    store.store(identity(), [1, 2, 3], [FakeKV()])

    borrowed = store.fetch(identity(), [1, 2, 3, 4, 5])
    assert borrowed.cached_tokens == 3
    # The caller mutates its copy and then vanishes without calling store().
    borrowed.prompt_cache.append(FakeKV())

    again = store.fetch(identity(), [1, 2, 3, 4, 5])

    assert again.cached_tokens == 3
    assert len(again.prompt_cache) == 1
    assert store.stats().entries == 1


def test_two_conversations_do_not_contaminate_each_other() -> None:
    """Independent sessions must stay independent.

    Two Codex sessions against the same model share a system prompt and diverge
    after it. Each must resume its own branch, not the other's.
    """
    store = cache(max_entries=4)
    shared = [1, 2, 3]
    store.store(identity(), [*shared, 10, 11], [FakeKV()])
    store.store(identity(), [*shared, 20, 21], [FakeKV()])

    a = store.fetch(identity(), [*shared, 10, 11, 12])
    b = store.fetch(identity(), [*shared, 20, 21, 22])

    assert a.cached_tokens == 5
    assert a.tokens_to_evaluate == [12]
    assert b.cached_tokens == 5
    assert b.tokens_to_evaluate == [22]


def test_the_longest_matching_session_wins() -> None:
    # With several sessions on one conversation, the most advanced one saves the
    # most work; picking any other would silently do redundant prefill.
    store = cache(max_entries=4)
    store.store(identity(), [1, 2], [FakeKV()])
    store.store(identity(), [1, 2, 3, 4], [FakeKV()])

    lookup = store.fetch(identity(), [1, 2, 3, 4, 5])

    assert lookup.cached_tokens == 4


def test_evicting_one_session_leaves_the_others_usable() -> None:
    store = cache(max_entries=2)
    store.store(identity(), [1, 1, 1], [FakeKV()])
    store.store(identity(), [2, 2, 2], [FakeKV()])
    store.store(identity(), [3, 3, 3], [FakeKV()])  # evicts the oldest

    assert store.fetch(identity(), [1, 1, 1, 9]).cached_tokens == 0
    assert store.fetch(identity(), [2, 2, 2, 9]).cached_tokens == 3
    assert store.fetch(identity(), [3, 3, 3, 9]).cached_tokens == 3
    assert store.stats().evictions == 1


def test_reuse_requires_the_exact_tokens_not_merely_the_same_length() -> None:
    # A same-length prefix that differs anywhere must not resume: the KV encodes
    # the actual tokens, not their count.
    store = cache()
    store.store(identity(), [1, 2, 3], [FakeKV()])

    assert store.fetch(identity(), [1, 2, 4, 5]).cached_tokens == 0
    assert store.fetch(identity(), [0, 2, 3, 5]).cached_tokens == 0
