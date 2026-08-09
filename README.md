# Quantum Codex GPT-OSS Server

**Run [OpenAI Codex](https://developers.openai.com/codex/cli) against GPT-OSS on
your own Mac.** No cloud, no API key, no telemetry — a local server that speaks
the Responses API exactly as Codex uses it, and a macOS app to operate it.

> **Version 1.0.0.**
> The core loop is validated end to end against real Codex and real weights.
> The desktop interface has had far less real-world use than the server. Expect
> rough edges, expect to read a log, and do not put this in front of anything
> you cannot afford to restart. Interfaces and on-disk formats may change.

---

## Why this exists

You can already point Codex at a generic local backend. The problem is that
Codex is not a generic client, and GPT-OSS is not a generic model.

Codex talks the **Responses API**, not Chat Completions: reasoning is a
first-class item that must survive across tool turns, tool calls end the
assistant turn, and metadata is fetched from `GET /v1/models` in Codex's own
schema. GPT-OSS talks **Harmony**: private reasoning on an `analysis` channel,
tool calls on `commentary`, answers on `final`, tools declared in real
namespaces, and `<|call|>` as an assistant stop token the model's own EOS list
does not contain.

A generic OpenAI-compatible shim flattens all of that into text. What you lose
is not cosmetic:

- **reasoning continuity** — drop the `analysis` channel between a tool call and
  its result and the model restarts the task instead of continuing it;
- **tool routing** — Codex dispatches on `name` and `namespace` as separate
  fields, so a flattened `namespace.name` is refused outright;
- **prompt reuse** — Codex replays the whole conversation every turn, so each
  prompt extends the last. Reusing that prefix is the difference between 1.56 s
  and 0.12 s of prefill on a 3-turn session;
- **honest limits** — a wrong `/v1/models` shape does not error, it silently
  degrades into fallback defaults.

This project goes deep on **one** path instead of wide across many:

```
Harness        Codex CLI 0.147.0
Model family   GPT-OSS (20B / 120B, mxfp4)
Model protocol OpenAI Harmony
Runtime        MLX
Platform       macOS, Apple Silicon
Wire protocol  Responses API, as Codex actually uses it
```

The specialisation is a design constraint, not a temporary limitation. This is
not an Ollama replacement, not a universal Hugging Face runtime, and not a
multi-harness adapter. If another Responses-compatible client happens to work,
that is welcome — it is not a product contract.

---

## Requirements

| | |
| --- | --- |
| Platform | macOS on Apple Silicon (developed on M3 Ultra) |
| Memory | 24 GB for the 20B, 96 GB recommended for the 120B |
| Codex | CLI **0.147.0** — the only version this is tested against |
| Model | GPT-OSS 20B or 120B in MLX format, mxfp4 |
| Toolchain | only for a source checkout: Python 3.12, [uv](https://docs.astral.sh/uv/), Node, Rust |

**Which model to use is not a matter of taste.** See
[Choosing a model](#choosing-a-model) below — it changes what works.

---

## Install

### The app

Open `Codex GPT-OSS Server.dmg` and drag it to Applications.

The bundle is **58 MB** and contains no Python, no dependencies, no MLX. On
first run the app offers to build its own runtime with the `uv` binary it
carries, producing about **387 MB** under Application Support. This takes a few
minutes and needs a network connection.

Nothing depends on system Python, Homebrew, your shell `PATH`, or a development
checkout. The runtime is tracked by a content fingerprint, so an app update that
changes the server rebuilds it; repairing it never touches your models,
profiles or settings.

### From source

```sh
make install       # uv sync + npm install
make doctor        # environment, runtime and model readiness
make dev-server    # serve GPT-OSS-20B on 127.0.0.1:8123
make dev-desktop   # the macOS control plane
```

---

## Get a model

Either bring your own or let the app fetch one.

**Import an existing directory** — Models → *Import existing*, or:

```sh
quantum-codex-server models import ~/models/gpt-oss-20b-mxfp4-bf16
```

**Download from Hugging Face** — Models → *Download*, or:

```sh
quantum-codex-server models download mlx-community/gpt-oss-20b-MXFP4-Q8
```

Downloads preflight free space, resume where they stopped, and cancel for real
rather than decoratively. The 120B is about 61 GiB.

**Models on external volumes are first-class.** A model whose disk is unplugged
reports `MISSING_VOLUME`, which is deliberately distinct from `MISSING` — "plug
the drive in" is not the same problem as "download it again", and a scan never
removes a model from your library.

**Locate** attaches a directory to a specific catalog entry and refuses if it is
a different model, so a 20B never ends up filed as the 120B. **Scan** walks the
configured roots and registers what it finds. Both leave your files where they
are; only *Download* writes anything.

**Where downloads go is configurable.** Point it at an external volume and the
120B never touches your boot disk:

```sh
quantum-codex-server models storage ~/models   # show it with no argument
```

### Per-model configuration

Each model carries its own settings, so the 20B's fast loop and the 120B's
deeper reasoning do not have to share one compromise:

```sh
quantum-codex-server models config gpt-oss-120b                       # show
quantum-codex-server models config gpt-oss-120b reasoning_effort=high
quantum-codex-server models config gpt-oss-120b max_output_tokens=    # clear
```

`reasoning_effort` matters more than it looks. Codex offers no way to change the
effort of a custom provider's model after launch — its picker is for its own
models — so whatever is set here is what the generated launch command carries.
GPT-OSS has exactly `low`, `medium` and `high`; the `xhigh`/`max` levels in
Codex's own list belong to other model families and are never advertised.

### Idle unload

Weights are released after a configurable idle period (default 10 minutes) and
reloaded transparently on the next request. The daemon keeps running and
`/v1/models` keeps answering, so Codex never sees the server disappear. Set
`--model-idle-timeout-minutes 0` to keep the model resident forever, or release
it now with `quantum-codex-server models unload`.

---

## Launch Codex against it

Start the server (from the app, or `quantum-codex-server serve --profile dev`), then
ask it for the exact command:

```sh
quantum-codex-server codex launch
```

which prints something like:

```sh
codex \
  -c model="gpt-oss-20b" \
  -c model_reasoning_effort="medium" \
  -c model_provider="qcs" \
  -c model_providers.qcs.name="QCS" \
  -c model_providers.qcs.base_url="http://127.0.0.1:8123/v1" \
  -c model_providers.qcs.wire_api="responses" \
  -c 'model_providers.qcs.auth={command="echo", args=["local"]}'
```

Everything is passed on the command line: **your `~/.codex/config.toml` is never
touched**, so your cloud Codex setup keeps working exactly as before.

For the Codex CLI's persistent configuration and for the VS Code extension —
neither of which takes `-c` overrides — ask for the file form instead:

```sh
quantum-codex-server codex launch --config
```

It prints a `~/.codex/config.toml` fragment to append yourself, built from the
same resolved settings as the command above, so the two cannot describe
different providers.

The `auth` line is not security — this server has no authentication and binds
loopback. Codex 0.147 only refreshes model metadata for a provider that declares
command-backed auth; without it Codex never calls `/v1/models` and silently
falls back to generic defaults.

No `model_catalog_json` is required. The server serves Codex's own metadata
schema, including base instructions written for GPT-OSS on Harmony.

---

## Choosing a model

This is the one finding most likely to surprise you, so it is stated plainly
rather than buried.

**For namespaced tools — MCP, multi-agent, Codex apps — use GPT-OSS-120B.**

Harmony's system prompt carries a single routing sentence, and the namespace in
it is a hardcoded literal: *"Calls to these tools must go to the commentary
channel: `'functions'`."* There is no way to make it name any other namespace,
so a model handed a second namespace is never told how to address it. The two
models react differently. Measured with a namespace declared and a prompt that
asks for it, three samples per cell:

| prompt structure | 20B | 120B |
| --- | --- | --- |
| namespace declared in the developer block | 0/3 | 2/3 |
| + explicit routing instruction | **0/3** | **3/3** |
| namespace declared in the system block | 0/3 | 2/3 |
| + explicit routing instruction | **0/3** | **3/3** |

The 120B addresses namespaces natively. The 20B never does — it emits
`functions.spawn_agent`, which Codex refuses with `unsupported call`.

The server keeps the 20B working through **conservative recipient
normalisation**: when the route a model emitted does not exist, and exactly one
declared tool justifies a correction structurally, the recipient is normalised
and the correction is logged. It never chooses between candidates. A name
declared in two namespaces, a prefix whose remainder is not a member of that
namespace, and a namespace nothing declared are all forwarded untouched, and
Codex refuses them.

So: **the 20B is excellent for ordinary coding turns** — shell tools, reasoning,
long sessions — and is what you want for a fast loop. Reach for the 120B when
namespaces are involved, or when two namespaces might declare the same tool
name, which is the case normalisation deliberately will not resolve.

---

## What is validated

Every item below was exercised against real Codex 0.147 and real GPT-OSS
weights, not mocked.

- Non-streaming and streaming `/v1/responses`, with a deterministic SSE order
- Reasoning as first-class `reasoning` items, replayed so continuity survives
  tool turns
- Codex-native `GET /v1/models`, with no `model_catalog_json`
- Sequential function tools, including two calls in one turn chain
- **Namespaced tools**, with `name` and `namespace` as separate fields
- **Multi-agent**: `spawn_agent` → real subagent → `wait_agent` → result
- Prompt-prefix reuse: 1.56 s → 0.16 s → 0.12 s across a real 3-turn session
- Cancellation and client disconnect leaving neither worker nor cache corrupted
- Streaming heartbeat during long prefills
- Packaged first-run bootstrap from bundle resources alone
- Model library, imports, Hugging Face download, external-volume states
- Idle model unload: the 20B released at exactly its configured idle period
  (witnessed at 1 minute; the default is 10), RSS 13.7 GB → 0.6 GB, the daemon
  still running and `/v1/models` still served, and the next request reloading it
  transparently in 1.5 s

---

## Known limitations

**MCP — PARTIAL.** MCP needs no MCP-specific code here: Codex presents an MCP
server as a namespace and this server carries it. A real MCP server *was*
reached through Codex, so declaration and dispatch are established. The result
does not come back, because a non-interactive `codex exec` answers every MCP
tool call with `user cancelled MCP tool call`, and Codex's one automated
approval path requires a structured output this server does not implement. This
is the client's approval behaviour, and no workaround will be added to bypass
it. Interactive Codex is untested.

**`tool_search` — BLOCKED.** Codex 0.147 does not offer `tool_search` to a
custom provider. `--enable tool_search` produces a byte-identical tool list, and
advertising `experimental_supported_tools` changes nothing. No deferred-tool
protocol will be written without a real client that exercises it.

**Instruction following on exhaustive tasks — not guaranteed.** The server sends
GPT-OSS base instructions that tell it to keep working until the task is
actually resolved. That is steering, not a guarantee. Measured on the 120B at
both `medium` and `high` effort: asked for an exhaustive repository-wide audit,
it opened a small fraction of the relevant files and then declared the work
complete, and strengthening the wording did not change that. If you need
exhaustive coverage, have the task carry its own checkable completion criterion
rather than trusting the model's account of what it did.

**Also missing or constrained**

- No encrypted reasoning; an encrypted-only reasoning item on replay is skipped,
  degrading continuity for that turn
- No structured output, no provider-executed hosted tools, no `web_search`
- One tool call per turn — Harmony's `<|call|>` ends the turn. This is correct
  semantics, not a limitation to work around
- One model loaded at a time
- Prefix reuse is exact-prefix only: a diverging prompt runs cold, because
  GPT-OSS's sliding-window layers make cache trimming unavailable
- Diagnostics are in memory and do not survive a restart
- No diagnostic bundle export, no benchmark command
- No long-session soak has been run, so bounded resource growth over hours is
  unestablished

---

## Using it from a terminal

Nothing is reachable only through the app.

| Command | Purpose |
| --- | --- |
| `serve [--profile NAME]` | Run the inference server |
| `status` | Report on the running server |
| `models list \| scan \| import \| download \| forget \| roots \| inspect` | The model library |
| `models unload` | Release the resident model; the server keeps running |
| `requests` | Recent request diagnostics |
| `cache stats \| clear` | Prompt cache counters and reset |
| `profiles list \| show \| set \| add \| remove \| default` | Named server configurations |
| `models storage [PATH]` | Where downloads are written |
| `models config SLUG [field=value …]` | Per-model settings, including reasoning effort |
| `codex launch [PROMPT]` | Print the codex command that talks to this server |
| `codex launch --config` | Print the `config.toml` fragment instead |
| `doctor` | Environment, runtime and model readiness |

The executable installs under two names. `quantum-codex-server` is the canonical
one used throughout this document; **`qcs` is a shorter alias for exactly the
same program**, so `qcs status` and `quantum-codex-server status` are
interchangeable.

The server binds `127.0.0.1`. A LAN bind must be a deliberate act.

---

## Development

```sh
make ci
```

runs the whole gate: `ruff`, `pytest`, TypeScript, `vitest`, the Vite build,
`cargo fmt --check`, `clippy` and `cargo test`. It runs no inference and needs
no model, so it is safe on any machine.

---

## Privacy

Everything runs on your machine. There is no telemetry and no outbound call
except a model download you asked for.

Request diagnostics record numbers, tool names and outcomes — **never prompts,
never reasoning text, never tool arguments or outputs.** A test fails if such a
field appears.

---

## License

MIT
