"""The GPT-OSS models this server is built for.

Quantum Codex is specialised on GPT-OSS, so the supported models are part of
the product rather than something a user has to know the Hugging Face id for.
They are listed here whether or not they are installed: "not installed yet" is
a state with an action attached, not an absence.

One definition, on the server. The desktop renders it and never carries its own
copy of a repository id -- a second list would drift the first time a
quantisation changed, and the user would download the wrong weights.

This is a *catalogue*, not a registry: it says what the product supports. What
is actually on disk is the model library's business, and the two are joined by
:func:`catalog_slug_for` in :func:`merge`.

The catalogue also says how prominently each model is offered, as
:class:`ModelTier`. That is a product statement -- this build recommends its own
tuned weights over the stock ones -- so it is decided here and rendered
verbatim, rather than inferred from a name by whoever is drawing the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import StrEnum
from typing import Any


class ModelTier(StrEnum):
    """How prominently a model is offered, and therefore where it is shown.

    Declared in display order: :func:`merge` groups the catalogue by this, so
    the order of these members is the order of the sections on the page. Adding
    a tier here places it without anything else needing to agree -- as long as
    it is declared *before* `OTHER`, which is not itself sorted: `merge` appends
    the models it does not recognise after everything in the catalogue, so a
    tier declared below `OTHER` would still be rendered above it.
    """

    #: Tuned by this project for the work it is built for.
    OPTIMIZED = "optimized"
    #: The upstream GPT-OSS releases.
    STOCK = "stock"
    #: Not in the catalogue at all: whatever the user imported or a root scan
    #: found. Never declared by a `CatalogEntry` -- `merge` synthesises it.
    OTHER = "other"


@dataclass(frozen=True)
class CatalogEntry:
    """A model this build is specialised for."""

    #: Stable identity. What :func:`catalog_slug_for` resolves a directory of
    #: these weights to, which is what lets an installed copy be recognised as
    #: this entry.
    slug: str
    display_name: str
    #: Where the weights come from. The one place this id exists.
    repo: str
    parameters: str
    #: Roughly what the download costs. Informational: the preflight in
    #: `downloads.py` sizes the transfer from the repository's own published
    #: file sizes, because that is the number that is actually true.
    download_bytes: int
    #: Which section this belongs to. Required, and deliberately without a
    #: default: a new entry states where it is offered rather than drifting
    #: into whichever tier happened to be listed first.
    tier: ModelTier
    note: str
    #: This model's intended defaults, before any user override.
    #:
    #: Only what is genuinely *model*-specific belongs here. A field left out
    #: inherits the server-wide default deliberately -- see `DEFAULTS_INHERITED`
    #: for which those are and why, so an omission is a decision rather than an
    #: oversight.
    defaults: dict[str, Any] = dc_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "defaults": dict(self.defaults),
            "slug": self.slug,
            "display_name": self.display_name,
            "repo": self.repo,
            "parameters": self.parameters,
            "download_bytes": self.download_bytes,
            "tier": self.tier,
            "note": self.note,
        }

    @property
    def repo_directory(self) -> str:
        """What a download of these weights is named on disk by default.

        `DownloadManager._default_destination` writes to
        `download_root() / repo.split("/", 1)[1]`. This takes the last segment
        rather than the second, which is the same string only because
        `REPO_PATTERN` admits exactly one slash -- worth stating, since the two
        expressions would diverge the day that pattern allowed a nested id.

        "By default" is the real limit here: `models download --destination`
        names the directory whatever the caller likes, and that copy is not
        recognised as this entry.
        """
        return self.repo.rsplit("/", 1)[-1]


#: The supported set. mxfp4 because that is what GPT-OSS ships as and what the
#: engine has been measured against; another quantisation is a different model
#: as far as this product is concerned.
#: Per-model settings the presets ship with no opinion about.
#:
#: Stated rather than omitted, so "no catalogue default" is visibly a decision:
#:
#: ``max_output_tokens``  one budget suits every entry; it is bounded by
#:                        whatever remains of the context window anyway.
#: ``temperature``        GPT-OSS is used here for coding, where the useful
#: ``top_p``              value does not differ by model size.
#: ``adapter_path``       no adapter ships with any entry, and the product
#:                        cannot know where a user's would live. Note that
#:                        absence here means *no adapter at all* rather than an
#:                        inherited value: unlike the three above, there is no
#:                        server-wide default behind it.
#:
#: A user may still override any of them per model; this is only about what is
#: shipped.
DEFAULTS_INHERITED: tuple[str, ...] = (
    "max_output_tokens",
    "temperature",
    "top_p",
    "adapter_path",
)

SUPPORTED: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        slug="gpt-oss-coder",
        display_name="GPT-OSS Coder",
        repo="exalandru/GPT-OSS-Coder-MLX",
        # A fine-tune of gpt-oss-120b: 36 layers, 128 experts, mxfp4 experts
        # with bf16 attention. It is the 120B in every respect the server cares
        # about, so it inherits the 120B's shipped settings rather than a
        # separate opinion.
        parameters="120B",
        download_bytes=61 * 1024**3,
        tier=ModelTier.OPTIMIZED,
        note="Tuned for agentic coding. The recommended model for Codex.",
        defaults={
            "display_name": "GPT-OSS Coder",
            "served_model_name": "gpt-oss-coder",
            "reasoning_effort": "medium",
            "context_length": 131072,
        },
    ),
    CatalogEntry(
        slug="gpt-oss-20b",
        display_name="GPT-OSS 20B",
        repo="mlx-community/gpt-oss-20b-MXFP4-Q8",
        parameters="20B",
        download_bytes=13 * 1024**3,
        tier=ModelTier.STOCK,
        note="Fast to load. Best for ordinary coding turns.",
        defaults={
            "display_name": "GPT-OSS 20B",
            # The public id, not the directory. `gpt-oss-20b-mxfp4-bf16` is where
            # the weights happen to sit; what a client asks for is the model.
            "served_model_name": "gpt-oss-20b",
            "reasoning_effort": "medium",
            "context_length": 131072,
        },
    ),
    CatalogEntry(
        slug="gpt-oss-120b",
        display_name="GPT-OSS 120B",
        repo="mlx-community/gpt-oss-120b-MXFP4-Q8",
        parameters="120B",
        download_bytes=61 * 1024**3,
        tier=ModelTier.STOCK,
        note="Needed for namespaced tools: MCP, multi-agent and Codex apps.",
        defaults={
            "display_name": "GPT-OSS 120B",
            "served_model_name": "gpt-oss-120b",
            "reasoning_effort": "medium",
            "context_length": 131072,
        },
    ),
)


def defaults_for(slug: str) -> dict[str, Any]:
    """The catalogue's intended settings for a model, before any override.

    Empty for a model that is not one of the presets: a directory a user
    imported has no shipped opinion attached, and its sensible fallbacks come
    from what is discovered on disk.
    """
    for entry in SUPPORTED:
        if entry.slug == slug:
            return dict(entry.defaults)
    return {}


def display_name_for(slug: str) -> str | None:
    """The catalogue's human-facing name, when this is a preset."""
    for entry in SUPPORTED:
        if entry.slug == slug:
            return entry.display_name
    return None


def catalog_slug_for(name: str) -> str:
    """Which catalogue entry a model directory denotes.

    A directory named exactly as the repository the weights come from *is* that
    model, whatever its quantisation suffix happens to look like. That rule is
    derived from `repo`, which this module already owns as the one place the id
    exists, so recognising a download needs no second table to agree with.

    Everything else falls back to `slug_for`, which strips a known quantisation
    suffix. That covers the directories a user lays out by hand -- but it is a
    heuristic over a fixed suffix list, and a repository whose name ends in
    anything else (`-MLX`, say) is invisible to it. Hence the exact rule first.

    Deliberately case-*sensitive*, and that is the whole safety argument for
    this function. Hugging Face writes the repository's name exactly, and every
    stock repository's directory already resolves to its own slug through
    `slug_for` alone, so an exact match changes the answer for no directory that
    resolved before. Matching casefolded would widen it: `gpt-oss-20b-mxfp4-q8`
    is *not* recognised by the suffix table (which is case-sensitive), so a user
    holding that directory is being served it under that name today, and folding
    the comparison would silently move it onto the catalogue's name and defaults
    -- renaming a model that a `config.toml` may already ask for by name.

    Recognising more directories must never be allowed to re-identify one that
    already worked.
    """
    # Imported here rather than at module scope for the same reason `merge`
    # does it: `models` asks this module for the presets' defaults.
    from ..models import slug_for

    stripped = name.strip()
    for entry in SUPPORTED:
        if stripped == entry.repo_directory:
            return entry.slug
    return slug_for(name)


def merge(
    reports: list[Any], *, overrides: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """The catalogue joined with what is installed.

    Every supported model appears exactly once, carrying its library report if
    one matches. Matching is :func:`catalog_slug_for`, so a directory
    named ``gpt-oss-20b-mxfp4-bf16`` or ``GPT-OSS-Coder-MLX`` is recognised as
    the entry it is a copy of, rather than shown beside it as a second,
    unrelated card.

    Models the user installed that are not in the catalogue keep their own
    entries, after the supported ones. They are equally usable; they are just
    not what this build was specialised and measured for.

    The result is grouped by :class:`ModelTier`, in the order the tiers are
    declared. Grouping here rather than in the caller is what lets a page render
    sections by walking the list once, and what keeps a fourth catalogue entry
    from landing in the wrong one because it was appended in the wrong place.
    """
    # Imported here rather than at module scope: `models` asks this module for
    # the presets' defaults, so a top-level import in both directions would be a
    # cycle. Only `merge` needs identity, and only at call time.
    from ..models import resolved_model_names

    remaining = list(reports)

    # Stable, so entries within one tier keep the order they are declared in.
    tier_order = list(ModelTier)
    merged: list[dict[str, Any]] = []
    for entry in sorted(SUPPORTED, key=lambda item: tier_order.index(item.tier)):
        report = next(
            (item for item in remaining if catalog_slug_for(item.entry.name) == entry.slug),
            None,
        )
        if report is not None:
            remaining.remove(report)
        names = (
            resolved_model_names(report, overrides=overrides)
            if report is not None
            else None
        )
        merged.append(
            {
                **entry.as_dict(),
                "id": names.library_id if names is not None else entry.slug,
                "served_name": names.served_name if names is not None else entry.slug,
                "display_name": names.display_name if names is not None else entry.display_name,
                "installed": report is not None,
                "model": report.as_dict() if report is not None else None,
            }
        )

    for report in remaining:
        slug = catalog_slug_for(report.entry.name)
        names = resolved_model_names(report, overrides=overrides)
        merged.append(
            {
                "slug": slug,
                "id": names.library_id,
                "display_name": names.display_name,
                "served_name": names.served_name,
                "repo": None,
                "parameters": None,
                "download_bytes": 0,
                "tier": ModelTier.OTHER,
                "note": "",
                "installed": True,
                "model": report.as_dict(),
            }
        )

    # A name two installed models both answer to is served by neither. Saying so
    # here is what keeps that from being visible only as a log line on a server
    # the user has no reason to be reading: the card shows the same fact the
    # daemon acted on.
    def contends(item: dict[str, Any]) -> bool:
        # Only what would actually be served can contend for a name: an entry on
        # an unplugged volume is not advertised, so it takes no name from
        # anything.
        return bool(item["model"] and item["model"].get("usable"))

    claimed: dict[str, int] = {}
    for item in merged:
        if contends(item):
            claimed[item["served_name"]] = claimed.get(item["served_name"], 0) + 1
    for item in merged:
        item["served_conflict"] = contends(item) and claimed[item["served_name"]] > 1

    # Whether modified weights answer for this model. A card that showed only
    # the model's name would be identical whether or not a LoRA was configured,
    # and a user comparing two answers would have nothing to attribute the
    # difference to. This is the *configured* adapter: what is actually applied
    # is measured at load and reported by the daemon's status.
    for item in merged:
        stored = (overrides or {}).get(item["id"], {})
        item["adapter_path"] = stored.get("adapter_path") or None

    return merged
