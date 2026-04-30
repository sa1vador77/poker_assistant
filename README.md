# poker_assistant

Live poker assistant with a screen overlay. The pipeline reads poker log
screenshots, recognises events (actions, cards, amounts, players), maintains
game state, and produces decision recommendations on the user's turn.

## Requirements

* Python 3.13.7
* [uv](https://docs.astral.sh/uv/) 0.5+

## Setup

```bash
uv python install 3.13.7
uv sync
uv run pre-commit install
```

`uv sync` creates `.venv/`, installs runtime and dev dependencies from
`pyproject.toml`, generates `uv.lock`, and installs the project itself in
editable mode.

## Common commands

```bash
uv run pytest                 # run tests
uv run mypy src tests tools   # type-check
uv run ruff check .           # lint
uv run ruff format .          # format
uv run pre-commit run --all-files
```

To run any script in the project's environment:

```bash
uv run python -m poker_assistant.cli.<entry_point>
```

## Architecture

Layered, with one-way dependencies:

```
cli  →  application  →  vision  →  domain  ←  compute
                          ↓          ↑
                         core  ←─────┘
```

| Layer           | Responsibility                                                             |
| --------------- | -------------------------------------------------------------------------- |
| `core`        | Cross-cutting infrastructure (logging, configuration, shared types).       |
| `domain`      | Pure poker logic: cards, ranges, equity, hand evaluation, state, decision. |
| `compute`     | Computational backends (e.g. native equity solver) behind protocols.       |
| `vision`      | Pixels → log events: frame prep, line extraction, template detection.     |
| `application` | Orchestrator: vision events → domain state → recommendations.            |
| `cli`         | Entry points and developer tools (debug runners, inspectors).              |

Rules:

* `domain` does not know about vision, IO, or the CLI.
* `vision` produces a narrow, typed event contract consumed by `application`.
* `compute` is invoked by `domain` only through `Protocol` interfaces.
* `core` is a leaf package and depends on nothing in the project.

## Project layout

```
src/poker_assistant/
    core/            # logger, config, shared types
    domain/          # poker business logic
        cards/
        ranges/
        equity/
        hand/
        state/
        decision/
        range_inference/
    compute/         # equity backends (Python, native)
    vision/          # screenshot → events
    application/     # orchestration
    cli/             # entry points
    _resources/      # bundled data (templates)
        templates/
            keyword/  ranks/  suits/  time/  amount/  parens/
tests/
    unit/
    integration/
    fixtures/
tools/               # ad-hoc developer scripts (not packaged)
```
