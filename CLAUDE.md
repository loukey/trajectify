# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # Install dependencies
uv run pytest tests/unit/ -v         # Run all unit tests
uv run pytest tests/unit/test_parser.py::test_parse_direct_json  # Single test
uv run trajectify config.example.yaml                            # Run with YAML config
uv run trajectify --task ./tasks/example-task --agent terminus   # Run single task
```

## Architecture

Trajectify is a trajectory collection platform that runs AI agents on tasks in Docker containers, then exports the interaction data in formats suitable for RL training (rollout with token IDs + logprobs) and SFT training (instruction + trajectory pairs).

### Core Flow

`cli.py` → `Orchestrator` (parallel scheduling) → `Runner` (single run lifecycle) → Agent + Environment + Verifier → Exporters

### Key Design Patterns

**TrajectoryRecorder injection**: The `Runner` creates a `TrajectoryRecorder` and injects it into agents. Agents call `recorder.add_agent_step()` etc. After the run, Runner passes the recorder to each configured exporter. This decouples recording from agent logic.

**Agent interface**: `BaseAgent.run()` takes `(instruction, env, recorder)` and returns `AgentResult`. Agents are registered via `@register_agent` decorator in `agents/registry.py`.

**Pluggable exporters**: `BaseExporter` ABC with implementations `AtifExporter`, `SftExporter`, `RolloutExporter`. Configured via YAML `exporter.formats` list. All receive the same recorder and produce different output files.

**Pluggable task loaders**: `BaseTaskLoader` ABC with `TerminalBenchLoader` for task.toml + instruction.md format.

**Unified UsageInfo**: Single Pydantic model in `models/usage.py` used everywhere (LLM responses, trajectory steps, aggregated metrics). Field name is `cache_tokens` (not `cached_tokens`).

### Runner Lifecycle (runner.py)

Environment start → Agent setup → Agent execution (with timeout) → Verification → Export → Cleanup. Each phase has separate `TimingInfo`. Partial results are preserved on timeout/error.

### Terminus Agent (agents/terminus/)

LLM-driven terminal agent split across: `agent.py` (~180 lines, main loop), `summarizer.py` (context compression), `parser.py` (JSON extraction), `tmux.py` (terminal interaction). The summarizer uses `Chat.llm` property (not `_model` directly).

### Configuration Hierarchy

`JobConfig` (YAML) → expanded to `RunConfig` list (one per task) → each RunConfig includes `AgentConfig`, `EnvironmentRunConfig`, `VerifierRunConfig`, `ExporterConfig`. Task-level config comes from `task.toml` via `TaskConfig`.

### Flat Modules

`orchestrator.py`, `runner.py`, `verifier.py`, `log.py` are top-level files in the package (not subdirectories with single files).
