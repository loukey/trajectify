# Trajectify

A trajectory collection platform that runs AI agents on tasks in Docker containers, then exports the interaction data in formats suitable for **RL training** (rollout with token IDs + logprobs) and **SFT training** (instruction + trajectory pairs).

## Features

- **Dockerized task environments** — each task runs in an isolated container with configurable CPU, memory, and storage limits
- **Pluggable agents** — register custom agents via `@register_agent` decorator; ships with `terminus` (LLM-driven terminal agent using tmux)
- **Pluggable exporters** — ATIF, SFT, and Rollout export formats; easily extensible
- **Pluggable task loaders** — load tasks from `task.toml` + `instruction.md` (TerminalBench format)
- **Parallel orchestration** — run multiple tasks concurrently with configurable concurrency
- **Automatic verification** — run test scripts inside containers to score agent outputs
- **Context summarization** — compress long conversations to stay within LLM context limits

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- Docker (with Docker Compose v2)

## Quick Start

```bash
# Install dependencies
uv sync

# Run a single task
uv run trajectify --task ./tasks/example-task --agent terminus

# Run with a YAML config file
uv run trajectify config.example.yaml
```

## Configuration

Copy `config.example.yaml` and customize:

```yaml
tasks:
  - ./tasks/example-task

agent:
  name: terminus
  model: anthropic/claude-sonnet-4-20250514
  temperature: 0.7
  max_turns: 100

environment:
  type: docker

exporter:
  formats:
    - atif
    - sft

n_concurrent: 1
output_dir: ./output
```

LLM API keys are configured in `configs/llm_config.yaml` (not tracked by git — see `config.example.yaml` for the structure).

## Architecture

```
cli.py → Orchestrator (parallel scheduling) → Runner (single run lifecycle)
  → Agent + Environment + Verifier → Exporters
```

### Core Flow

1. **CLI** parses args and loads YAML config
2. **Orchestrator** schedules runs with concurrency control
3. **Runner** manages a single run lifecycle: environment start → agent setup → agent execution (with timeout) → verification → export → cleanup
4. **Agent** (e.g. Terminus) interacts with the Docker environment via tmux, driven by an LLM
5. **Verifier** runs test scripts inside the container to produce reward scores
6. **Exporters** write trajectory data in configured formats

### Project Structure

```
trajectify/
├── agents/
│   ├── base.py              # BaseAgent ABC
│   ├── registry.py          # @register_agent decorator
│   ├── nop.py               # No-op agent for testing
│   └── terminus/            # LLM-driven terminal agent
│       ├── agent.py         # Main agent loop
│       ├── parser.py        # JSON response extraction
│       ├── summarizer.py    # Context compression
│       └── tmux.py          # Tmux session management
├── config/
│   ├── loader.py            # YAML config loading
│   └── models.py            # Pydantic config models
├── environments/
│   ├── base.py              # BaseEnvironment ABC
│   ├── docker.py            # Docker Compose implementation
│   └── compose/             # Compose YAML templates
├── exporters/
│   ├── base.py              # BaseExporter ABC
│   ├── atif.py              # ATIF format exporter
│   ├── sft.py               # SFT format exporter
│   └── rollout.py           # Rollout format exporter
├── llm/
│   ├── base.py              # LLM provider ABC
│   ├── chat.py              # Chat session with history
│   └── litellm_provider.py  # LiteLLM-based provider
├── models/
│   ├── result.py            # AgentResult
│   ├── rollout.py           # Rollout data models
│   ├── task.py              # TaskConfig
│   ├── trajectory.py        # TrajectoryRecorder
│   └── usage.py             # UsageInfo
├── task_loaders/
│   ├── base.py              # BaseTaskLoader ABC
│   └── terminal_bench.py    # TerminalBench loader
├── cli.py                   # Entry point
├── orchestrator.py          # Parallel scheduling
├── runner.py                # Single run lifecycle
├── verifier.py              # Test script verification
└── log.py                   # Logging setup
```

## Testing

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run a specific test
uv run pytest tests/unit/test_parser.py::test_parse_direct_json
```

## License

See [LICENSE](LICENSE) for details.
