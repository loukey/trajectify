# Trajectify

A trajectory collection platform that runs AI agents on tasks in Docker containers, then exports the interaction data in formats suitable for **RL training** (rollout with token IDs + logprobs) and **SFT training** (instruction + trajectory pairs).

## Features

- **Dockerized task environments** — each task runs in an isolated container with configurable CPU, memory, and storage limits
- **Pluggable agents** — register custom agents via `@register_agent` decorator; ships with `terminus` (LLM-driven terminal agent using tmux)
- **Pluggable exporters** — ATIF, SFT, and Rollout export formats; easily extensible
- **Pluggable task loaders** — supports both Terminal-Bench 1.0 (`task.yaml`) and 2.0 (`task.toml`) formats, with auto-detection
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
uv run trajectify --task ./tasks/terminal-bench-2/regex-log --agent terminus

# Run with a YAML config file
uv run trajectify config.example.yaml
```

## Running Tasks

### Single Task (CLI)

```bash
uv run trajectify --task <task-dir> --agent terminus [options]
```

Options:
- `--model <model>` — LLM model name (default: from `configs/llm_config.yaml`)
- `--temperature <float>` — sampling temperature (default: 0.7)
- `--output-dir <dir>` — output directory (default: `./output`)
- `--export-formats <fmts>` — comma-separated: `atif`, `sft`, `rollout` (default: `atif`)
- `--collect-rollout` — collect token-level rollout details (logprobs, token IDs)
- `--disable-verifier` — skip verification step

### Multiple Tasks (YAML Config)

Create a YAML config to run multiple tasks in parallel:

```yaml
tasks:
  - ./tasks/terminal-bench-2/regex-log
  - ./tasks/terminal-bench-2/openssl-selfsigned-cert
  - ./tasks/terminal-bench-1/hello-world

agent:
  name: terminus
  model: openai/gpt-4o
  temperature: 0.7
  max_turns: 100

environment:
  type: docker
  force_build: false
  delete_after: true

verifier:
  disable: false

exporter:
  formats:
    - atif
    - sft

n_concurrent: 3
output_dir: ./output
```

```bash
uv run trajectify config.yaml
```

The Orchestrator runs up to `n_concurrent` tasks simultaneously. Each task gets its own Docker container, agent instance, and output directory.

### LLM Configuration

API keys and model settings are configured in `configs/llm_config.yaml` (not tracked by git):

```yaml
model: openai/MiniMax-M2.5
api_base: https://api.minimaxi.com/v1
api_key: your-api-key-here
```

The platform uses [LiteLLM](https://docs.litellm.ai/) under the hood, so any model supported by LiteLLM can be used (OpenAI, Anthropic, local models, etc.).

## Supported Task Formats

Trajectify auto-detects the task format based on the config file present in the task directory.

### Terminal-Bench 2.0 (`task.toml`)

The current standard format, used by [Terminal-Bench 2.0](https://www.tbench.ai/). 89 carefully verified tasks.

```
task-name/
├── task.toml          # Metadata, timeouts, resource limits
├── instruction.md     # Task description (English)
├── environment/       # Docker build context
│   └── Dockerfile
├── tests/             # Verification scripts
│   └── test.sh        # Writes reward to /logs/verifier/reward.txt
└── solution/          # Reference solution
```

`task.toml` example:
```toml
[metadata]
difficulty = "medium"
category = "data-processing"
tags = ["regex", "log-analysis"]

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 180.0

[environment]
docker_image = "alexgshaw/regex-log:20251031"
cpus = 1
memory = "2G"
```

### Terminal-Bench 1.0 (`task.yaml`)

The original format from [Terminal-Bench](https://github.com/laude-institute/terminal-bench). 241 tasks.

```
task-name/
├── task.yaml          # Metadata + inline instruction
├── Dockerfile         # At task root (not in subdirectory)
├── docker-compose.yaml
├── run-tests.sh       # Test script (uses $TEST_DIR env var)
├── tests/             # Test data
│   └── test_outputs.py
└── solution.sh        # Reference solution
```

Key differences from 2.0:
- Instruction is **inline** in `task.yaml` (not a separate file)
- Dockerfile is at the **task root** (not in `environment/`)
- Test script is `run-tests.sh` at root (not `tests/test.sh`)
- Timeout fields: `max_agent_timeout_sec` / `max_test_timeout_sec` (instead of nested `agent.timeout_sec`)
- Test scripts don't write reward files — Trajectify infers reward from the exit code

## Terminus Agent

Terminus is the built-in LLM-driven terminal agent. It operates inside a tmux session within the Docker container, sending commands and observing terminal output in a loop.

### How It Works

```
┌─────────────────────────────────────────────────┐
│                  Agent Loop                      │
│                                                  │
│  1. Capture terminal screen (tmux capture-pane)  │
│  2. Send screen + instruction to LLM            │
│  3. LLM responds with JSON:                     │
│     { analysis, plan, commands, is_task_complete }│
│  4. Execute commands via tmux send-keys          │
│  5. Wait for output, then repeat from step 1    │
│                                                  │
│  Exit when: is_task_complete confirmed twice     │
│             OR timeout reached                   │
│             OR max_turns exceeded                │
└─────────────────────────────────────────────────┘
```

Each turn:
1. The agent captures the current terminal screen via `tmux capture-pane`
2. The screen content is sent to the LLM as a user message
3. The LLM responds with a structured JSON containing:
   - `analysis` — what it observes in the terminal
   - `plan` — its next intended action
   - `commands` — keystrokes to send (with wait durations)
   - `is_task_complete` — whether the task is done
4. Commands are executed via `tmux send-keys` (literal mode + Enter)
5. The agent waits for the specified duration, then captures the new screen

When the LLM sets `is_task_complete: true`, the agent asks for confirmation — the LLM must confirm once more before the loop exits. This prevents premature termination.

### Error Recovery

- **Parse errors**: If the LLM returns invalid JSON (e.g. XML instead of JSON), the agent sends an error message asking for valid JSON and continues
- **Context overflow**: Triggers automatic summarization (see below)
- **Output truncation**: Uses partial response and continues

### Context Summarization

Long-running tasks can exceed the LLM's context window. The Summarizer handles this with a **hybrid summarize-and-buffer approach**:

1. **Monitoring**: Before each turn, the summarizer estimates remaining context capacity (model's context limit minus estimated usage). When the remaining space drops below 8,000 tokens, summarization triggers.

2. **Compression**: The conversation is split into older messages and recent messages (last 6 turns). The older messages are summarized into a 2-3 paragraph summary by calling the LLM.

3. **Reconstruction**: The conversation history is rebuilt as:
   ```
   [System prompt]
   [User: CONVERSATION SUMMARY — compressed history]
   [Assistant: Acknowledged]
   [Recent 6 turns preserved in full fidelity]
   ```

4. **Forced summarization**: Also triggers when a `ContextLengthExceededError` is caught, as a recovery mechanism.

This allows the agent to work on complex tasks requiring many steps without losing critical context about what has been accomplished.

## Architecture

```
cli.py → Orchestrator (parallel scheduling) → Runner (single run lifecycle)
  → Agent + Environment + Verifier → Exporters
```

### Core Flow

1. **CLI** parses args / loads YAML config, applies LLM config
2. **Orchestrator** schedules runs with asyncio semaphore-based concurrency control
3. **Runner** manages a single run lifecycle:
   - Environment start (Docker build + compose up)
   - Agent setup (install tmux, create session)
   - Agent execution (main loop, with timeout)
   - Verification (upload tests, run test script, read reward)
   - Export (ATIF, SFT, rollout)
   - Cleanup (compose down, remove images)
4. **TrajectoryRecorder** is injected into the agent; it collects all interaction steps, then is passed to exporters

### Project Structure

```
trajectify/
├── agents/
│   ├── base.py                # BaseAgent ABC
│   ├── registry.py            # @register_agent decorator
│   ├── nop.py                 # No-op agent for testing
│   └── terminus/              # LLM-driven terminal agent
│       ├── agent.py           # Main agent loop (~240 lines)
│       ├── parser.py          # JSON response extraction
│       ├── summarizer.py      # Context compression
│       ├── tmux.py            # Tmux session management
│       └── prompts/           # System/timeout prompt templates
├── config/
│   ├── loader.py              # YAML config loading
│   └── models.py              # Pydantic config models
├── environments/
│   ├── base.py                # BaseEnvironment ABC
│   ├── docker.py              # Docker Compose implementation
│   └── compose/               # Compose YAML templates
├── exporters/
│   ├── base.py                # BaseExporter ABC
│   ├── atif.py                # ATIF format exporter
│   ├── sft.py                 # SFT format exporter
│   └── rollout.py             # Rollout format exporter
├── llm/
│   ├── base.py                # LLM provider ABC
│   ├── chat.py                # Chat session with history
│   └── litellm_provider.py    # LiteLLM-based provider
├── models/
│   ├── result.py              # AgentResult, RunResult
│   ├── rollout.py             # Rollout data models
│   ├── task.py                # TaskConfig, TaskPaths
│   ├── trajectory.py          # TrajectoryRecorder
│   └── usage.py               # UsageInfo
├── task_loaders/
│   ├── base.py                # BaseTaskLoader ABC
│   ├── terminal_bench.py      # Terminal-Bench 2.0 loader
│   └── terminal_bench_v1.py   # Terminal-Bench 1.0 loader
├── cli.py                     # Entry point
├── orchestrator.py            # Parallel scheduling
├── runner.py                  # Single run lifecycle
├── verifier.py                # Test script verification
└── log.py                     # Logging setup
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
