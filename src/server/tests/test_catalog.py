"""The supported GPT-OSS models are part of the product, not a user's homework.

Both appear whether or not they are installed, exactly once, and an installed
directory is recognised as the supported entry it corresponds to rather than
shown beside it as a stranger.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantum_codex.library.catalog import SUPPORTED, merge
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


def test_both_supported_models_appear_with_nothing_installed() -> None:
    merged = merge([])

    assert [m["slug"] for m in merged] == ["gpt-oss-20b", "gpt-oss-120b"]
    assert all(m["supported"] for m in merged)
    assert not any(m["installed"] for m in merged)


def test_a_supported_model_carries_the_repository_to_download() -> None:
    """The one place this id exists; the interface must never carry its own."""
    merged = merge([])

    assert merged[0]["repo"] == "mlx-community/gpt-oss-20b-MXFP4-Q8"
    assert merged[0]["download_bytes"] > 0


def test_an_installed_directory_is_recognised_as_its_catalog_entry() -> None:
    """The discriminating case: no duplicate card beside the supported one."""
    merged = merge([report("gpt-oss-20b-mxfp4-bf16")])

    assert len(merged) == 2
    twenty = next(m for m in merged if m["slug"] == "gpt-oss-20b")
    assert twenty["installed"] is True
    assert twenty["model"]["name"] == "gpt-oss-20b-mxfp4-bf16"


def test_the_other_supported_model_stays_visible_and_uninstalled() -> None:
    merged = merge([report("gpt-oss-20b-mxfp4-bf16")])

    other = next(m for m in merged if m["slug"] == "gpt-oss-120b")
    assert other["installed"] is False
    assert other["model"] is None


def test_a_model_that_is_not_supported_keeps_its_own_entry_after_the_presets() -> None:
    merged = merge([report("some-other-model")])

    assert [m["slug"] for m in merged[:2]] == ["gpt-oss-20b", "gpt-oss-120b"]
    assert merged[2]["slug"] == "some-other-model"
    assert merged[2]["supported"] is False


def test_an_unusable_installed_model_still_reconciles_rather_than_duplicating() -> None:
    """A model on an unplugged volume is that entry, in a bad state."""
    merged = merge([report("gpt-oss-120b-mxfp4-bf16", ModelState.MISSING_VOLUME)])

    assert len(merged) == 2
    hundred = next(m for m in merged if m["slug"] == "gpt-oss-120b")
    assert hundred["installed"] is True
    assert hundred["model"]["state"] == ModelState.MISSING_VOLUME.value


def test_a_second_directory_for_one_preset_remains_configurable() -> None:
    merged = merge([report("gpt-oss-20b-mxfp4-bf16"), report("gpt-oss-20b-MXFP4-Q8")])

    assert [m["slug"] for m in merged] == [
        "gpt-oss-20b",
        "gpt-oss-120b",
        "gpt-oss-20b",
    ]
    assert merged[2]["supported"] is False
    assert merged[2]["installed"] is True


def test_every_catalog_entry_has_a_distinct_slug() -> None:
    slugs = [entry.slug for entry in SUPPORTED]

    assert len(slugs) == len(set(slugs))


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
