"""Environment readiness report.

`doctor` answers one question: would `serve` work here, and if not, why. It must
run and produce a useful report even when the environment is broken, so every
probe is individually guarded rather than relying on module-level imports.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys

from . import CLI_NAME

# The installed distribution, which keeps the older name. Only the executable
# was renamed for v1.0.0; renaming the distribution too would churn the lockfile
# and the managed-runtime fingerprint to no user-visible benefit. The report
# prints the command name because that is what someone types.
DISTRIBUTION = "quantum-codex"

# Pinned in pyproject. Reported here so a drifted environment is visible at a
# glance instead of surfacing later as a protocol or inference change.
PINNED = ("mlx", "mlx-lm", "openai-harmony", "fastapi", "pydantic", "uvicorn")


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "MISSING"


def _metal_status() -> str:
    try:
        import mlx.core as mx
    except Exception as exc:  # noqa: BLE001 - a broken mlx must be reported, not raised
        return f"unavailable ({exc.__class__.__name__}: {exc})"

    try:
        info = mx.device_info()
    except Exception as exc:  # noqa: BLE001
        return f"present, device query failed ({exc.__class__.__name__}: {exc})"

    memory_gb = info.get("max_recommended_working_set_size", 0) / 1024**3
    return f"available, max recommended working set {memory_gb:.1f} GB"


# Wide enough that the longest label still leaves a separating space.
# `quantum-codex-server` is exactly 20 characters, so a 20-column field printed
# `quantum-codex-server1.0.0`.
_LABEL = max(len(CLI_NAME), len("platform"), *(len(p) for p in PINNED)) + 2


def run_doctor() -> int:
    print(f"{'platform':<{_LABEL}}{platform.platform()} ({platform.machine()})")
    print(f"{'python':<{_LABEL}}{sys.version.split()[0]} at {sys.executable}")
    print(f"{CLI_NAME:<{_LABEL}}{_version(DISTRIBUTION)}")
    print()

    problems = 0
    for package in PINNED:
        version = _version(package)
        if version == "MISSING":
            problems += 1
        print(f"{package:<{_LABEL}}{version}")

    print()
    print(f"{'metal':<{_LABEL}}{_metal_status()}")

    if platform.machine() != "arm64":
        problems += 1
        print()
        print("PROBLEM: this server targets Apple Silicon only; MLX will not run here.")

    if problems:
        print()
        print(f"{problems} problem(s) found.")
        return 1

    return 0
