"""Quantum Codex GPT-OSS Server.

A local server and control plane specialised on exactly one path:

    Codex -> Responses -> GPT-OSS Harmony -> MLX -> Apple Silicon

The specialisation is a design constraint, not a temporary limitation. Nothing
here is meant to become a generic LLM runtime, a multi-harness adapter, or a
model-family registry.
"""

__version__ = "1.0.1"

# The installed executable names. Canonical first: this binary controls the
# Quantum Codex GPT-OSS Server, so it says so. `qcs` is the short alias and
# matches the bundle id (`com.exalandru.qcs`) and the Codex provider id.
#
# Anything that prints a command for the user to run resolves it from here
# rather than spelling it out, so a rename stays a one-line change instead of a
# grep across Python, Rust, docs and tests.
CLI_NAME = "quantum-codex-server"
CLI_ALIAS = "qcs"
