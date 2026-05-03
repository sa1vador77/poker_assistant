"""Rewrite ephemeral build-venv paths in compile_commands.json to stable ones.

Why this exists
---------------
scikit-build-core builds the native extension inside an ephemeral
virtualenv that uv tears down immediately after the build. The
compile_commands.json that CMake emits records absolute paths to
that vanished venv, which makes it useless for IDE tooling
(VSCode C/C++, clangd) — they cannot resolve include paths to
files that no longer exist.

The fix is to redirect every reference from the ephemeral venv to
the project's own .venv, where pybind11 is also installed (as a dev
dependency for exactly this reason). The two installations come from
the same package version and contain identical headers, so
substitution is safe.

Usage
-----
    uv run python tools/fix_compile_commands.py

The script is idempotent: running it twice is a no-op.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPILE_COMMANDS_GLOB = "build/*/compile_commands.json"
EPHEMERAL_VENV_PATTERN = re.compile(
    r"/Users/[^/]+/\.cache/uv/builds-v\d+/\.tmp[^/]+/lib/python\d+\.\d+/site-packages"
)


def find_compile_commands() -> Path:
    matches = list(REPO_ROOT.glob(COMPILE_COMMANDS_GLOB))
    if not matches:
        raise SystemExit(
            f"No compile_commands.json found under {COMPILE_COMMANDS_GLOB}. Run `uv sync` first."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple compile_commands.json found, expected one: {matches}. "
            f"Clean stale build directories with `rm -rf build/`."
        )
    return matches[0]


def find_stable_site_packages() -> Path:
    candidates = list((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one .venv site-packages directory, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def main() -> int:
    compile_commands_path = find_compile_commands()
    stable_site_packages = find_stable_site_packages()

    raw = compile_commands_path.read_text()
    rewritten = EPHEMERAL_VENV_PATTERN.sub(str(stable_site_packages), raw)

    if rewritten == raw:
        print("No ephemeral paths found; nothing to rewrite.")
        return 0

    # Round-trip through json to make sure we did not break the file.
    json.loads(rewritten)
    compile_commands_path.write_text(rewritten)
    print(f"Rewrote ephemeral paths in {compile_commands_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
