"""Harmony rendering and parsing.

This package is the only place in the codebase that imports ``openai_harmony``.
Everywhere else speaks the canonical IR.

GPT-OSS must never be driven by an improvised chat template, and its output must
never be parsed with hand-rolled regular expressions. Harmony has real concepts
for channels, recipients and namespaces; using them is what makes reasoning
continuity and tool calling correct rather than approximately correct.
"""

from .parse import ANALYSIS, COMMENTARY, FINAL, ParsedGeneration, StreamingParser, parse_completion
from .render import HarmonyRenderer

__all__ = [
    "ANALYSIS",
    "COMMENTARY",
    "FINAL",
    "HarmonyRenderer",
    "ParsedGeneration",
    "StreamingParser",
    "parse_completion",
]
