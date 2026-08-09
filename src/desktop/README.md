# Quantum Codex GPT-OSS Server (desktop)

A macOS control plane over the headless server. It is a *client*, not an owner:
the Python server and its CLI remain the single source of truth for
capabilities, profiles, model validation and configuration.

Nothing here can be done only here. Every action maps to something the terminal
can already do (cahier 40, 52).

## What the Rust side does

Only what a browser cannot:

- [`paths.rs`](src-tauri/src/paths.rs) — locate the application-support
  directory, mirroring `quantum_codex/config.py`. Locations only, never file
  shapes.
- [`daemon.rs`](src-tauri/src/daemon.rs) — read `runtime.json`, reach the
  management plane with its token, spawn the server detached, signal it by pid.
- [`lib.rs`](src-tauri/src/lib.rs) — the command surface.

Server responses cross into the interface as opaque JSON. **No schema is
repeated in Rust or TypeScript**, so a capability added on the server appears in
the interface without a change here. The frontend reads fields defensively and
renders `—` for anything absent, rather than substituting a plausible default.

## The managed runtime

The bundle stays light: **58 MB**, almost all of it the `uv` binary. It carries
`uv` and the locked project — manifest, lock, interpreter pin, package — and
nothing else. Python, the dependency tree and MLX are absent; uv rebuilds them
on the user's machine into Application Support. Measured on a clean bootstrap:
**387 MB** of environment and interpreter that never travelled in the bundle.

Nothing depends on system Python, Homebrew, the shell `PATH`, or a development
checkout.

`--python` and `--project` are passed explicitly rather than left to uv's
discovery, which keys off the current directory — unpredictable for a sidecar
launched from a `.app`. Without them uv takes the newest interpreter satisfying
`requires-python`, and the lock resolves a different package set from the tested
one.

### Runtime states

A long install never happens as a side effect of pressing Start. The state is
explicit and the app asks first:

| state | meaning |
| --- | --- |
| `UNINITIALIZED` | nothing installed; offer to initialise |
| `INITIALIZING` | a sync is running |
| `READY` | usable |
| `UPDATE_REQUIRED` | installed, but built from a different project |
| `BROKEN` | present and unusable, or nothing to install from |

Obsolescence is detected by a **content fingerprint** — a hash of the
interpreter pin, manifest, lock and every source file — not by version number.
Quantum Diffusion Server learned why: code edited without a version bump left
the environment looking current, so the app kept serving whatever the first
install captured. Their fix compares modification times; a hash is stricter,
because an mtime changes when a file is merely touched and does not change when
one is restored from an archive.

Rebuilding replaces `server/`, `env/` and the uv caches. It never touches
`profiles.json`, `settings.json` or any model directory, so repair costs a
download and nothing else.

Runtime state and model state are independent: a healthy runtime with no GPT-OSS
weights configured is a normal situation, and the dashboard says so separately.

## The daemon owns itself

The desktop app never holds a child process handle. It starts the server in its
own session and afterwards talks to whatever `runtime.json` describes, so
closing the window cannot take the server with it (D1) — a Codex session runs
for hours and must not depend on a window staying open.

That also means a runtime file can outlive its server. Every read treats it as a
claim: the pid check is a cheap negative test, and only a successful management
request establishes that something is answering.

Stopping sends `SIGTERM`, waits, and escalates to `SIGKILL` only for a process
that will not go.

## Running it

```sh
make dev-desktop     # from the repository root
```

`QUANTUM_CODEX_COMMAND` says how to invoke the CLI. The Makefile sets it to
`uv run --project src/server quantum-codex-server` for a development checkout; an
installed app would use a bundled executable.

```sh
make build-desktop   # .app and .dmg
```

Both stage `build/desktop/staging/` first — the uv sidecar and the Python
project that `tauri.conf.json` bundles. Tauri's build script checks those paths
exist, so a bare `cargo check`, `cargo clippy` or `cargo test` needs them too
and has no hook that creates them. `make stage-desktop` is that step on its own,
and it is what CI runs before the cargo commands.

## What it shows

Server state and endpoint, the loaded model and its quantization, the
capabilities the server reports, live inference counters, prompt cache
occupancy and hit ratio, and the tail of the server log.

Start, stop, restart, clear the cache, and produce the Codex launch command —
the last one by running `quantum-codex-server codex launch`, not by rebuilding the
provider wiring here. Two definitions of that wiring would drift the first time
Codex changed anything.

## The four views

**Dashboard** — server state and endpoint, loaded model and quantization,
reported capabilities, live inference counters, prompt-cache occupancy and hit
ratio, log tail. Start, stop, restart, clear the cache, and produce the Codex
launch command by running `quantum-codex-server codex launch` rather than rebuilding
the provider wiring here — two definitions of it would drift the first time
Codex changed anything.

**Models** — the library with per-model state, quantization, context, disk size
and volume status. Import, scan, download from Hugging Face with progress and
real cancellation, reveal in Finder, forget.

**Configuration** — a form *generated* from the schema the server publishes.
Adding a setting on the server makes it appear here with no frontend change.

**Diagnostics** — lifetime outcomes, recent-window medians, prefix-reuse figures
and a table of recent requests. Numbers, tool names and outcomes only.

Model list, scan, import and forget are **disk state** and work with no server
running — you import a model in order to configure and start one.
