"""The GPT-OSS models this server is built for.

Quantum Codex is specialised on GPT-OSS, so the two supported models are part
of the product rather than something a user has to know the Hugging Face id
for. They are listed here whether or not they are installed: "not installed
yet" is a state with an action attached, not an absence.

One definition, on the server. The desktop renders it and never carries its own
copy of a repository id -- a second list would drift the first time a
quantisation changed, and the user would download the wrong weights.

This is a *catalogue*, not a registry: it says what the product supports. What
is actually on disk is the model library's business, and the two are joined by
slug in :func:`merge`.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any


@dataclass(frozen=True)
class CatalogEntry:
    """A model this build is specialised for."""

    #: Stable identity. The same slug the library derives from a directory
    #: name, which is what lets an installed copy be recognised as this entry.
    slug: str
    display_name: str
    #: Where the weights come from. The one place this id exists.
    repo: str
    parameters: str
    #: Roughly what the download costs, for the space warning before it starts.
    download_bytes: int
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
            "note": self.note,
        }


#: The supported set. mxfp4 because that is what GPT-OSS ships as and what the
#: engine has been measured against; another quantisation is a different model
#: as far as this product is concerned.
#: Per-model settings the presets ship with no opinion about.
#:
#: Stated rather than omitted, so "no catalogue default" is visibly a decision:
#:
#: ``max_output_tokens``  the same budget suits both; it is bounded by whatever
#:                        remains of the context window anyway.
#: ``temperature``        GPT-OSS is used here for coding, where the useful
#: ``top_p``              value does not differ by model size.
#: ``adapter_path``       no adapter ships with either preset, and the product
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
        slug="gpt-oss-20b",
        display_name="GPT-OSS 20B",
        repo="mlx-community/gpt-oss-20b-MXFP4-Q8",
        parameters="20B",
        download_bytes=13 * 1024**3,
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


def merge(
    reports: list[Any], *, overrides: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """The catalogue joined with what is installed.

    Every supported model appears exactly once, carrying its library report if
    one matches. Matching is by slug, so a directory named
    ``gpt-oss-20b-mxfp4-bf16`` is recognised as the 20B rather than shown beside
    it as a second, unrelated card.

    Models the user installed that are not in the catalogue keep their own
    entries, after the supported ones. They are equally usable; they are just
    not what this build was specialised and measured for.
    """
    # Imported here rather than at module scope: `models` asks this module for
    # the presets' defaults, so a top-level import in both directions would be a
    # cycle. Only `merge` needs identity, and only at call time.
    from ..models import resolved_model_names, slug_for

    remaining = list(reports)

    merged: list[dict[str, Any]] = []
    for entry in SUPPORTED:
        report = next(
            (item for item in remaining if slug_for(item.entry.name) == entry.slug),
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
                "supported": True,
                "installed": report is not None,
                "model": report.as_dict() if report is not None else None,
            }
        )

    for report in remaining:
        slug = slug_for(report.entry.name)
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
                "note": "",
                "supported": False,
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
