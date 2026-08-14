"""The supported GPT-OSS models are part of the product, not a user's homework.

All of them appear whether or not they are installed, exactly once, grouped by
the tier that decides where they are offered, and an installed directory is
recognised as the entry it is a copy of rather than shown beside it as a
stranger.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from quantum_codex.library.catalog import (
    SUPPORTED,
    CatalogEntry,
    ModelTier,
    catalog_slug_for,
    merge,
)
from quantum_codex.library.registry import ModelState


@dataclass
class FakeEntry:
    name: str
    path: str


@dataclass
class FakeReport:
    entry: FakeEntry
    state: ModelState = ModelState.READY

    def as_dict(self) -> dict:
        return {"name": self.entry.name, "state": self.state.value}


def report(name: str, state: ModelState = ModelState.READY) -> FakeReport:
    return FakeReport(entry=FakeEntry(name=name, path=f"/m/{name}"), state=state)


def slug_of(merged: list[dict], slug: str) -> dict:
    """The one merged entry with this slug, or a failure that says so."""
    matches = [item for item in merged if item["slug"] == slug]
    assert len(matches) == 1, f"expected exactly one {slug!r}, got {len(matches)}"
    return matches[0]


def test_every_supported_model_appears_with_nothing_installed() -> None:
    merged = merge([])

    assert [m["slug"] for m in merged] == ["gpt-oss-coder", "gpt-oss-20b", "gpt-oss-120b"]
    assert not any(m["installed"] for m in merged)


def test_the_optimized_model_is_offered_before_the_stock_ones() -> None:
    """Order is a product statement, and the catalogue is where it is made.

    A page that renders this list top to bottom gets the intended prominence
    without deciding anything, so this is the assertion that keeps the decision
    on the server.
    """
    merged = merge([])

    assert [m["tier"] for m in merged] == [
        ModelTier.OPTIMIZED,
        ModelTier.STOCK,
        ModelTier.STOCK,
    ]


def test_the_catalogue_no_longer_ships_a_supported_flag() -> None:
    """`tier` replaced it, and two fields answering one question can disagree."""
    assert all("supported" not in item for item in merge([report("some-other-model")]))


def test_no_catalogue_entry_claims_the_other_tier() -> None:
    """`other` is what `merge` synthesises for a model it does not know.

    A catalogue entry declaring it would be drawn as inventory the user brought,
    losing the Download and Locate… actions that are the whole point of listing
    an uninstalled model.
    """
    assert all(entry.tier is not ModelTier.OTHER for entry in SUPPORTED)


def test_a_supported_model_carries_the_repository_to_download() -> None:
    """The one place this id exists; the interface must never carry its own."""
    merged = merge([])

    assert slug_of(merged, "gpt-oss-20b")["repo"] == "mlx-community/gpt-oss-20b-MXFP4-Q8"
    assert slug_of(merged, "gpt-oss-coder")["repo"] == "exalandru/GPT-OSS-Coder-MLX"
    assert all(m["download_bytes"] > 0 for m in merged)


def test_an_installed_directory_is_recognised_as_its_catalog_entry() -> None:
    """The discriminating case: no duplicate card beside the supported one."""
    merged = merge([report("gpt-oss-20b-mxfp4-bf16")])

    assert len(merged) == 3
    twenty = slug_of(merged, "gpt-oss-20b")
    assert twenty["installed"] is True
    assert twenty["model"]["name"] == "gpt-oss-20b-mxfp4-bf16"


def test_the_other_supported_model_stays_visible_and_uninstalled() -> None:
    merged = merge([report("gpt-oss-20b-mxfp4-bf16")])

    other = next(m for m in merged if m["slug"] == "gpt-oss-120b")
    assert other["installed"] is False
    assert other["model"] is None


def test_a_model_that_is_not_supported_keeps_its_own_entry_after_the_presets() -> None:
    merged = merge([report("some-other-model")])

    assert [m["slug"] for m in merged[:3]] == ["gpt-oss-coder", "gpt-oss-20b", "gpt-oss-120b"]
    assert merged[3]["slug"] == "some-other-model"
    assert merged[3]["tier"] == ModelTier.OTHER


def test_an_unusable_installed_model_still_reconciles_rather_than_duplicating() -> None:
    """A model on an unplugged volume is that entry, in a bad state."""
    merged = merge([report("gpt-oss-120b-mxfp4-bf16", ModelState.MISSING_VOLUME)])

    assert len(merged) == 3
    hundred = slug_of(merged, "gpt-oss-120b")
    assert hundred["installed"] is True
    assert hundred["model"]["state"] == ModelState.MISSING_VOLUME.value


def test_a_second_directory_for_one_preset_remains_configurable() -> None:
    merged = merge([report("gpt-oss-20b-mxfp4-bf16"), report("gpt-oss-20b-MXFP4-Q8")])

    assert [m["slug"] for m in merged] == [
        "gpt-oss-coder",
        "gpt-oss-20b",
        "gpt-oss-120b",
        "gpt-oss-20b",
    ]
    assert merged[3]["tier"] == ModelTier.OTHER
    assert merged[3]["installed"] is True


def test_every_catalog_entry_has_a_distinct_slug() -> None:
    slugs = [entry.slug for entry in SUPPORTED]

    assert len(slugs) == len(set(slugs))


# -- which directory is which model ------------------------------------------


def test_a_directory_named_after_the_repository_is_that_model() -> None:
    """The rule the suffix heuristic cannot reach.

    `GPT-OSS-Coder-MLX` ends in `-MLX`, which is in no quantisation-suffix
    table, so `slug_for` returns the whole name. The second assertion is what
    makes this discriminating: it fails if the new rule silently stopped firing
    and the old one happened to agree.
    """
    from quantum_codex.models import slug_for

    assert catalog_slug_for("GPT-OSS-Coder-MLX") == "gpt-oss-coder"
    assert slug_for("GPT-OSS-Coder-MLX") != "gpt-oss-coder"


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        # The stock repositories, whose downloaded names both rules agree on.
        ("gpt-oss-20b-MXFP4-Q8", "gpt-oss-20b"),
        ("gpt-oss-120b-MXFP4-Q8", "gpt-oss-120b"),
        # Laid out by hand, which only the suffix rule recognises.
        ("gpt-oss-20b-mxfp4-bf16", "gpt-oss-20b"),
        # The tuned build, which only the exact rule reaches.
        ("GPT-OSS-Coder-MLX", "gpt-oss-coder"),
        # Differing in case is a *different* directory, deliberately. Folding
        # the comparison would capture a directory that resolves to its own name
        # today, renaming a model a `config.toml` may already ask for.
        ("gpt-oss-20b-mxfp4-q8", "gpt-oss-20b-mxfp4-q8"),
        ("gpt-oss-coder-mlx", "gpt-oss-coder-mlx"),
        # A stranger stays a stranger.
        ("some-other-model", "some-other-model"),
    ],
)
def test_directory_names_resolve_to_the_model_they_hold(directory: str, expected: str) -> None:
    assert catalog_slug_for(directory) == expected


def test_the_catalogue_rule_changes_the_answer_for_no_name_that_already_resolved() -> None:
    """The safety property of widening the join at all.

    `catalog_slug` decides the served name, the display name and the shipped
    defaults, so capturing a directory that already resolved would silently
    rename someone's installed model. The only name whose answer may differ from
    `slug_for` is a repository directory the suffix table cannot reach.
    """
    from quantum_codex.models import slug_for

    names = [
        "gpt-oss-20b-MXFP4-Q8",
        "gpt-oss-120b-MXFP4-Q8",
        "gpt-oss-20b-mxfp4-bf16",
        "gpt-oss-120b-mxfp4-bf16",
        "gpt-oss-20b-mxfp4-q8",
        "gpt-oss-20b-8bit",
        "my-own-gpt-oss",
        "some-other-model",
        "GPT-OSS-Coder-MLX",
    ]
    changed = {name for name in names if catalog_slug_for(name) != slug_for(name)}

    assert changed == {"GPT-OSS-Coder-MLX"}


def test_a_downloaded_catalogue_model_produces_no_second_entry() -> None:
    """The failure this whole join exists to prevent: one model, two places.

    Before the catalogue could recognise its own repository name, these weights
    would have shown as an uninstalled card *and* as a row underneath it.
    """
    merged = merge([report("GPT-OSS-Coder-MLX")])

    assert len(merged) == 3
    assert slug_of(merged, "gpt-oss-coder")["installed"] is True
    assert not [m for m in merged if m["tier"] == ModelTier.OTHER]


def test_the_merged_list_is_grouped_by_tier_whatever_order_entries_are_declared_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the grouping, not the current tuple's order.

    Asserting against the shipped catalogue would pass just as well if `merge`
    did no grouping at all and `SUPPORTED` merely happened to be sorted.
    """
    scrambled = (
        CatalogEntry(
            slug="a-stock",
            display_name="A stock",
            repo="x/a-stock",
            parameters="20B",
            download_bytes=1,
            tier=ModelTier.STOCK,
            note="",
        ),
        CatalogEntry(
            slug="z-optimized",
            display_name="Z optimized",
            repo="x/z-optimized",
            parameters="20B",
            download_bytes=1,
            tier=ModelTier.OPTIMIZED,
            note="",
        ),
        CatalogEntry(
            slug="m-stock",
            display_name="M stock",
            repo="x/m-stock",
            parameters="20B",
            download_bytes=1,
            tier=ModelTier.STOCK,
            note="",
        ),
    )
    monkeypatch.setattr("quantum_codex.library.catalog.SUPPORTED", scrambled)

    merged = merge([])

    assert [m["tier"] for m in merged] == [
        ModelTier.OPTIMIZED,
        ModelTier.STOCK,
        ModelTier.STOCK,
    ]
    # Alphabetically `z-optimized` sorts last, so a sort on the slug -- or no
    # sort at all -- fails here. Stable within a tier: `a-stock` was declared
    # before `m-stock` and stays before it.
    assert [m["slug"] for m in merged] == ["z-optimized", "a-stock", "m-stock"]


# -- adapters ----------------------------------------------------------------


def test_a_card_says_when_modified_weights_answer_for_a_model() -> None:
    """Otherwise the card is identical with and without a LoRA applied, and a
    user comparing two answers has nothing to attribute the difference to."""
    merged = merge(
        [report("gpt-oss-20b-mxfp4-bf16")],
        overrides={"gpt-oss-20b": {"adapter_path": "/adapters/style-fr"}},
    )

    twenty = next(m for m in merged if m["slug"] == "gpt-oss-20b")
    hundred = next(m for m in merged if m["slug"] == "gpt-oss-120b")
    assert twenty["adapter_path"] == "/adapters/style-fr"
    assert hundred["adapter_path"] is None


def test_a_model_with_no_adapter_says_so_rather_than_omitting_the_key() -> None:
    # The interface reads this field on every card; an absent key and a null
    # would have to be read differently for no reason.
    for item in merge([report("gpt-oss-20b-mxfp4-bf16")]):
        assert item["adapter_path"] is None
