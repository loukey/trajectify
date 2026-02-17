"""Terminal-bench v1 task loader — loads task.yaml (instruction inline)."""

from __future__ import annotations

from pathlib import Path

import yaml

from trajectify.models.task import Task, TaskConfig, TaskPaths
from trajectify.task_loaders.base import BaseTaskLoader
from trajectify.task_loaders.terminal_bench import _parse_memory_string


class TerminalBenchV1Loader(BaseTaskLoader):
    """Loads tasks in terminal-bench v1 format (task.yaml with inline instruction)."""

    @staticmethod
    def format_name() -> str:
        return "terminal_bench_v1"

    def load(self, task_dir: Path) -> Task:
        task_dir = Path(task_dir).resolve()

        config_path = task_dir / "task.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"task.yaml not found in {task_dir}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        instruction = raw.pop("instruction", "")

        # Map v1 field names to v2 structure
        config_dict: dict = {"metadata": {}}

        # Metadata fields
        for key in (
            "author_name", "author_email", "difficulty", "category", "tags",
            "parser_name", "run_tests_in_same_shell", "disable_asciinema",
            "expert_time_estimate_min", "junior_time_estimate_min",
        ):
            if key in raw:
                config_dict["metadata"][key] = raw.pop(key)

        # Timeouts
        agent_timeout = raw.pop("max_agent_timeout_sec", None)
        test_timeout = raw.pop("max_test_timeout_sec", None)
        if agent_timeout is not None:
            config_dict["agent"] = {"timeout_sec": float(agent_timeout)}
        if test_timeout is not None:
            config_dict["verifier"] = {"timeout_sec": float(test_timeout)}

        # Environment
        env_raw = raw.pop("environment", {})
        if isinstance(env_raw, dict) and "memory" in env_raw and isinstance(env_raw["memory"], str):
            env_raw["memory_mb"] = _parse_memory_string(env_raw["memory"])
        if env_raw:
            config_dict["environment"] = env_raw

        config = TaskConfig.model_validate(config_dict)

        paths = TaskPaths(
            task_dir=task_dir,
            config_path=config_path,
            instruction_path=None,
            environment_dir=task_dir,
            tests_dir=task_dir / "tests",
            test_script_path=task_dir / "run-tests.sh",
            solution_dir=None,
        )

        return Task(
            name=task_dir.name,
            instruction=instruction,
            config=config,
            paths=paths,
        )
