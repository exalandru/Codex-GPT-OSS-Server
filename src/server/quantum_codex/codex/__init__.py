"""Codex-facing compatibility.

Everything that exists because the client is Codex lives here: what its
`/v1/models` schema looks like, and which request elements this server supports,
accepts inertly, or refuses.

Keeping it in one package is what stops Codex's dialect from leaking into the
canonical IR, the Harmony renderer, or the MLX engine.
"""

from .capabilities import CAPABILITIES, CompatProblem, ResponsesCapabilities, ToolSupport
from .model_metadata import CODEX_MODELS_SCHEMA_VERSION, build_models_response

__all__ = [
    "CAPABILITIES",
    "CODEX_MODELS_SCHEMA_VERSION",
    "CompatProblem",
    "ResponsesCapabilities",
    "ToolSupport",
    "build_models_response",
]
