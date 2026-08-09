/**
 * Stage the Python project and the uv binary into the Tauri bundle.
 *
 * This is the only thing that creates `build/desktop/staging/`, and every path
 * `tauri.conf.json` lists under `externalBin` and `bundle.resources` points
 * into it. `tauri_build::build()` validates those paths, so this has to have
 * run before *any* cargo command that executes the build script — not just
 * before a bundle. `tauri dev` and `tauri build` reach it through
 * `beforeDevCommand`/`beforeBuildCommand`; a bare `cargo check`, `cargo clippy`
 * or `cargo test` does not, which is what `make stage-desktop` is for. Both
 * routes end here; there is no second implementation to keep in step.
 *
 * Nothing is duplicated in version control: `build/desktop/staging/` is
 * generated before every dev run and every build. What ships is the minimum for
 * `uv sync --frozen` to rebuild the environment on the user's machine — the
 * manifest, the lock, the interpreter pin, and the package itself.
 *
 * The dependencies are *not* shipped. mlx alone carries a large Metal shader
 * library, and the whole tree is far past what belongs in an application
 * bundle. uv is a single binary that can fetch CPython and rebuild the exact
 * locked set, so that is what travels.
 */
import { execFileSync } from "node:child_process";
import { chmodSync, cpSync, existsSync, mkdirSync, realpathSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const repo = resolve(desktop, "../..");
const server = join(repo, "src", "server");
const staging = join(repo, "build", "desktop", "staging");
const resources = join(staging, "resources", "server");

/**
 * What to embed, relative to `src/server`.
 *
 * `README.md` is not documentation here: `pyproject.toml` declares it as
 * `readme`, and hatchling refuses to build the wheel without it.
 *
 * `.python-version` pins the interpreter. Without it uv takes the newest
 * version satisfying `requires-python`, and the lock then resolves a different,
 * untested package set.
 */
const PAYLOAD = ["pyproject.toml", "uv.lock", ".python-version", "README.md", "quantum_codex"];

/**
 * The staging tree is rebuilt from scratch rather than updated in place, so its
 * contents are a function of the checkout and nothing else. Updating in place
 * would let a stale sidecar, or a file that has since left `PAYLOAD`, keep
 * satisfying Tauri's resource checks on a machine that had built before — which
 * is precisely the difference between a development machine and a clean CI
 * checkout, and precisely the difference that should not exist.
 *
 * Only `staging/` is removed. The cargo target directory that `app:dev` and
 * `app:build` point at is its sibling under `build/desktop/`, and rebuilding
 * the Rust tree on every sync would be an unrelated cost.
 */
function reset() {
  rmSync(staging, { recursive: true, force: true });
}

function syncPython() {
  mkdirSync(resources, { recursive: true });

  for (const entry of PAYLOAD) {
    const source = join(server, entry);
    if (!existsSync(source)) {
      throw new Error(
        `Missing payload entry: ${source}. ` +
          (entry === "uv.lock" ? "Run `uv lock --project src/server`." : ""),
      );
    }
    cpSync(source, join(resources, entry), {
      recursive: true,
      // Bytecode and tool caches would bloat the bundle and be regenerated
      // anyway.
      filter: (path) => !/(__pycache__|\.pyc$|\.pytest_cache|\.ruff_cache|\.venv)/.test(path),
    });
  }
  console.log(`✓ Python project staged in ${resources}`);
}

function syncUv() {
  const triple = execFileSync("rustc", ["--print", "host-tuple"], { encoding: "utf8" }).trim();
  const binaries = join(staging, "binaries");
  mkdirSync(binaries, { recursive: true });

  let uv;
  try {
    uv = execFileSync("which", ["uv"], { encoding: "utf8" }).trim();
  } catch {
    throw new Error("`uv` is not on PATH; install it before building.");
  }

  // Tauri requires this exact suffix for an `externalBin` entry.
  const target = join(binaries, `uv-${triple}`);
  // Resolved first, because `which` frequently answers with a symlink —
  // Homebrew links `bin/uv` into the Cellar, and an installer that manages
  // versions is no different. Copying the link rather than the executable
  // stages something that satisfies Tauri's existence check and still ships an
  // application pointing at a path on the machine that built it, and it would
  // make the chmod below modify the developer's own uv install instead of the
  // staged copy. (`cpSync`'s own `dereference` is not the answer: on Node 24 it
  // rejects a symlink to a file as a directory.)
  cpSync(realpathSync(uv), target);
  chmodSync(target, 0o755);
  const version = execFileSync(uv, ["--version"], { encoding: "utf8" }).trim();
  console.log(`✓ sidecar ${version} staged as uv-${triple}`);
}

reset();
syncPython();
syncUv();
