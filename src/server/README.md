# Quantum Codex GPT-OSS Server (headless)

The headless part of Quantum-Codex-OSS-MLX-Server: an HTTP server that lets
Codex drive GPT-OSS through Harmony on MLX, plus the CLI that operates it.

This package never needs the desktop app. Everything essential is reachable from
a terminal (cahier 40).

## Running

```sh
uv run quantum-codex-server serve \
  --model ~/models/gpt-oss-20b-mxfp4-bf16 \
  --served-model-name gpt-oss-20b
```

`--served-model-name` gives clients a stable id instead of a filesystem path.

Binding is `127.0.0.1` by default. A LAN bind must be a deliberate act.

## Commands

Everything essential works from a terminal, with no GUI involved (cahier 40).

| Command | Purpose |
| --- | --- |
| `serve [--profile NAME]` | Run the inference server |
| `status` | Report on the running server |
| `models list\|scan\|import\|download\|forget\|roots` | The model library |
| `models inspect PATH` | Is this a GPT-OSS model this server can run? |
| `requests` | Recent request diagnostics |
| `cache stats` / `cache clear` | Prompt cache counters and reset |
| `profiles list\|show\|set\|add\|remove\|default` | Named server configurations |
| `codex launch [PROMPT]` | Print the codex command that talks to this server |
| `doctor` | Environment, runtime and model readiness |

`status`, `cache stats` and `cache clear` talk to a running server through its
management plane. `models inspect` exits non-zero on `UNSUPPORTED`, so a script
can gate on it without parsing output.

## Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Liveness and queue state |
| `/v1/models` | `GET` | Model metadata, in the schema Codex expects |
| `/v1/responses` | `POST` | Responses API, streaming and non-streaming |

## Using it from Codex

No `model_catalog_json` is needed — the server serves Codex's own model metadata
schema. Pass the provider on the command line rather than editing
`~/.codex/config.toml`, so your cloud Codex setup stays untouched:

```sh
codex exec \
  -c model="gpt-oss-20b" \
  -c model_reasoning_effort="medium" \
  -c model_provider="qcs" \
  -c model_providers.qcs.name="QCS" \
  -c model_providers.qcs.base_url="http://127.0.0.1:8123/v1" \
  -c model_providers.qcs.wire_api="responses" \
  -c 'model_providers.qcs.auth={command="echo", args=["local"]}' \
  "your prompt"
```

The `auth` line is not about security — the server has no auth. Codex 0.147 only
refreshes model metadata for a provider that declares command-backed auth, so
without it Codex never calls `/v1/models` and falls back to generic defaults.

### What is supported today

Text and reasoning, streaming and non-streaming, sequential function tools with
reasoning continuity across tool turns, prompt-prefix reuse, and **namespaced
tools** — declared as real Harmony namespaces and returned with `name` and
`namespace` as separate fields, which is what Codex's router dispatches on.
Multi-agent works on that path; MCP is carried by it and is PARTIAL for a
client-side reason (see the root README).

Namespaced work wants **GPT-OSS-120B**. The 20B never addresses a namespace and
depends on conservative recipient normalisation (D7); see the root README for
the measurements.

Provider-executed `web_search` is dropped — there is no search executor here.
Each distinct dropped tool is logged once.

## Configuration

State lives under `~/Library/Application Support/com.exalandru.qcs/`,
split by owner and lifetime rather than merged into one file (cahier 39):

| File | Holds | Lifetime |
| --- | --- | --- |
| `profiles.json` | named server configurations | edited deliberately |
| `settings.json` | user preferences | rarely changes |
| `runtime.json` | endpoint, pid and management token of a running server | dies with the process |

`QUANTUM_CODEX_HOME` overrides the directory. Each file carries a schema
`version`; one written by a newer build is refused rather than read with older
rules and silently stripped of what it added. Writes go through a temporary file
and `os.replace`, so an interrupted write leaves the previous file intact.

## Management plane

`/internal/status` and `/internal/cache` are this project's own operational
surface, separate from the `/v1` contract owed to Codex and free to change with
the CLI and GUI that consume it. Every route needs the bearer token from
`runtime.json`, which is owner-readable: the server binds loopback, and a
loopback port is reachable by every process on the machine.

## Layout

```
quantum_codex/
    api/         HTTP surface, SSE, error envelope, Responses state machine
    codex/       compatibility policy: supported / accepted-inert / unknown -> 400
    harmony/     render and parse; the only place openai-harmony is touched
    inference/   MlxEngine, the single long-lived worker thread, prompt cache
    library/     model registry, volumes, downloads
    routing.py   recipient normalisation against declared tool topology (D7)
```

The boundaries matter more than the file names.

## Development

```sh
make install   # from the repository root
make doctor
make test
make lint
```

Critical dependency versions (`mlx`, `mlx-lm`, `openai-harmony`) are pinned
exactly. Bumping one is a deliberate task that needs a real Codex run behind it.

## Prompt cache

A Codex turn replays the whole conversation, so each turn's prompt extends the
previous one. The server keeps live KV sessions and resumes the matching one
instead of re-evaluating the shared prefix.

Measured on a real three-turn `codex exec` session against GPT-OSS-20B:

| turn | input tokens | reused | prefill |
| --- | --- | --- | --- |
| 1 | 3348 | 0 | 1.56 s |
| 2 | 3550 | 3349 | 0.16 s |
| 3 | 3632 | 3551 | 0.12 s |

`GET /health` reports entries, bytes, hits, misses, hit ratio and per-model
totals. `cached_tokens` in a response's usage is what the cache actually
returned — `prompt_cache_key` is recorded and never used to claim a hit.

Budgets: `--cache-max-entries` (default 4, `0` disables) and `--cache-max-bytes`
(default 8 GiB).
