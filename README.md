# poker_assistant

A live poker assistant that watches a poker client's log on screen and,
when it is the player's turn, recommends an action backed by equity and
expected-value math.

The pipeline reads a screenshot, segments the log into lines,
recognises events (actions, cards, amounts, players), maintains the
state of the current hand, and produces a recommendation. It is built
as a layered Python application with a C++17 backend for the
performance-critical equity calculations.

## Project status

Active development. The lower layers of the domain are complete and
covered by tests; the orchestration and UI layers are not yet built.

| Layer             | Module                                                                 | Status      |
| ----------------- | ---------------------------------------------------------------------- | ----------- |
| Infrastructure    | Repository scaffold, logging, tooling                                  | Done        |
| Domain primitives | `domain/cards`, `domain/ranges`, `domain/hand`, `domain/board` | Done        |
| Compute           | Native build system,`_native_equity.cpp`                             | Done        |
| Compute           | `card_encoding`, `equity_protocol`                                 | Done        |
| Compute           | Python and native equity backends                                      | In progress |
| Domain composite  | `domain/equity`, `domain/hero` (made hand, draws, snapshot)        | Pending     |
| Domain top        | `domain/decision`, `domain/state`                                  | Pending     |
| Vision            | Screenshot to event extraction                                         | Pending     |
| Application       | Orchestrator, live loop, on-screen overlay                             | Pending     |

The README is updated as layers land. See the commit history for the
full chronology.

## Requirements

- Python 3.13.7 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) 0.5 or newer
- A C++17 compiler. On macOS this comes with the Xcode Command Line
  Tools (`xcode-select --install`).
- [CMake](https://cmake.org/) 3.20 or newer and
  [ninja](https://ninja-build.org/) for building the native equity
  extension. On macOS:
  ```bash
  brew install cmake ninja
  ```

The native extension is platform-specific (built fresh on each
machine). It has been verified on Apple Silicon (macOS arm64) and is
expected to build on Linux x86\_64; Windows is not currently targeted.

## Setup

```bash
uv python install 3.13.7
uv sync
uv run pre-commit install
```

`uv sync` creates `.venv/`, resolves dependencies from `pyproject.toml`
into `uv.lock`, installs the project in editable mode, and triggers
the build of the C++ extension via scikit-build-core.

If the build fails, the most common cause is a missing system tool;
re-read the Requirements section. The build directory is `build/` and
is regenerated from scratch when needed.

To verify the install:

```bash
uv run python -c "from poker_assistant.compute import _native_equity; \
    print(_native_equity.supports_exact_weighted_postflop(1, 5))"
```

It should print `True`.

## Development

### Daily commands

```bash
uv run pytest                         # run the test suite
uv run mypy src tests                 # static type checking
uv run ruff check .                   # lint
uv run ruff format .                  # auto-format
uv run pre-commit run --all-files     # run every pre-commit hook
```

The pre-commit hooks (ruff, ruff-format, mypy, uv-lock,
trailing-whitespace, end-of-file-fixer) run automatically on every
`git commit`. A failing hook blocks the commit; modified files are
written back to the working tree for review.

### Editing pure-Python code

The editable install picks up Python changes immediately. No
re-install is required after editing `.py` files.

### Editing the native extension

After editing `src/poker_assistant/compute/_native_equity.cpp`, the
compiled `.so` is **not** rebuilt automatically — uv's build
environment is ephemeral, which is incompatible with
scikit-build-core's on-import rebuild mode. Force a rebuild
explicitly:

```bash
uv sync --reinstall-package poker-assistant
```

This is the standard workflow for native extensions under uv and
matches how projects like NumPy or scikit-learn operate. To start
from scratch (rare, only when something looks corrupted):

```bash
rm -rf build/
uv sync --reinstall-package poker-assistant
```

### IDE setup for the native extension (optional)

VSCode and other tools that index C++ need to know where pybind11
headers live. CMake emits `compile_commands.json` into the build
directory; point your IDE at it.

For VSCode with the C/C++ extension, create `.vscode/c_cpp_properties.json`:

```json
{
    "version": 4,
    "configurations": [
        {
            "name": "CMake-driven",
            "compileCommands": "${workspaceFolder}/build/<wheel-tag>/compile_commands.json",
            "intelliSenseMode": "macos-clang-arm64",
            "cStandard": "c17",
            "cppStandard": "c++17"
        }
    ]
}
```

Replace `<wheel-tag>` with the directory CMake actually produced
(`ls build/`). The `.vscode/` directory is ignored by git so each
developer can configure their tooling independently.

### Adding a dependency

```bash
uv add <package>             # runtime dependency
uv add --dev <package>       # dev-only dependency (linters, test tools)
```

Both update `pyproject.toml` and `uv.lock` in the same step. The
`uv-lock` pre-commit hook keeps the lock file in sync if either is
edited by hand.

## Architecture

The project follows a strict layered architecture with one-way
dependencies. No layer imports from a layer above it.

```
cli  →  application  →  vision  →  domain  ←  compute
                          ↓          ↑
                         core  ←─────┘
```

| Layer           | Responsibility                                                                                                                      |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `core`        | Cross-cutting infrastructure: logging, configuration, shared utilities.                                                             |
| `domain`      | Pure poker logic: cards, ranges, hand evaluation, board texture, hero state, decision. Knows nothing about IO, screens, or the CLI. |
| `compute`     | Computational backends behind protocols, including the native equity extension.                                                     |
| `vision`      | Pixels to typed events: frame preparation, line extraction, template matching.                                                      |
| `application` | Orchestration: vision events drive domain state, which drives the recommendation.                                                   |
| `cli`         | Entry points and developer tools (debug runners, inspectors).                                                                       |

Two rules keep the layering honest:

- The `domain` layer never imports from `vision`, `application`, or
  `cli`. This is enforced by reading; there is no automated check yet.
- `compute` is consumed by `domain` only through `Protocol` interfaces
  defined in `compute/`, not by directly importing concrete backends.
  This lets the native and Python equity backends be swapped at
  runtime, with the Python backend serving as the reference
  implementation.

## Project layout

```
poker_assistant/
├── pyproject.toml             # single source of project configuration
├── uv.lock                    # resolved dependency graph (committed)
├── CMakeLists.txt             # native extension build definition
├── .python-version            # pinned Python version (committed)
├── src/
│   └── poker_assistant/
│       ├── core/              # logger, shared infrastructure
│       ├── domain/            # poker logic
│       │   ├── cards.py       # Card, Rank, Suit, SuitOrder, parsing
│       │   ├── ranges/        # combos, hand classes, parser, presets
│       │   ├── hand/          # 5-7 card hand evaluator
│       │   └── board/         # board texture analysis
│       ├── compute/           # equity backends + native extension
│       │   ├── _native_equity.cpp
│       │   ├── card_encoding.py
│       │   └── equity_protocol.py
│       ├── vision/            # planned: screenshot to events
│       ├── application/       # planned: orchestrator
│       └── cli/               # planned: entry points
├── tests/
│   ├── unit/                  # fast, layer-isolated tests
│   ├── integration/           # planned: vision + domain on real screenshots
│   └── fixtures/              # screenshots, templates, gold outputs
└── tools/                     # ad-hoc developer scripts (not packaged)
```

`build/` is a CMake build tree, regenerated by uv and ignored by git.

## Testing strategy

Each layer has its own test suite under `tests/unit/<layer>/`. The
suites are designed to be fast and independent: domain tests do not
touch the filesystem; vision tests will use small screenshot
fixtures; integration tests will exercise vision + domain end to end
on full screenshots.

The C++ extension is tested indirectly through the equity layer's
test suite, which compares its output against the Python reference
implementation. There are no separate C++ unit tests; the Python
suite is the contract.

## License

MIT — see [LICENSE](LICENSE).
