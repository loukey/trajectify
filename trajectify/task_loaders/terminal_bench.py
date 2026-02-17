"""Terminal-bench task loader — loads task.toml + instruction.md."""

from __future__ import annotations

import re
from pathlib import Path

import toml

from trajectify.models.task import Task, TaskConfig, TaskPaths
from trajectify.task_loaders.base import BaseTaskLoader


def _parse_memory_string(value: str) -> int:
    """Convert a memory string like '2G', '512M' to megabytes."""
    value = value.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(G|GB|M|MB|K|KB)?$", value)
    if not match:
        raise ValueError(f"Cannot parse memory string: {value!r}")

    num = float(match.group(1))
    unit = match.group(2) or "M"

    if unit.startswith("G"):
        return int(num * 1024)
    elif unit.startswith("K"):
        return max(1, int(num / 1024))
    else:
        return int(num)


class TerminalBenchLoader(BaseTaskLoader):
    """Loads tasks in terminal-bench format (task.toml + instruction.md)."""

    @staticmethod
    def format_name() -> str:
        return "terminal_bench"

    def load(self, task_dir: Path) -> Task:
        task_dir = Path(task_dir).resolve()

        config_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"

        if not config_path.exists():
            raise FileNotFoundError(f"task.toml not found in {task_dir}")
        if not instruction_path.exists():
            raise FileNotFoundError(f"instruction.md not found in {task_dir}")

        paths = TaskPaths(
            task_dir=task_dir,
            config_path=config_path,
            instruction_path=instruction_path,
            environment_dir=task_dir / "environment",
            tests_dir=task_dir / "tests",
            test_script_path=task_dir / "tests" / "test.sh",
            solution_dir=task_dir / "solution",
        )

        raw = toml.loads(config_path.read_text(encoding="utf-8"))

        # Handle TB2 extensions: memory string, docker_image, storage
        env_raw = raw.get("environment", {})
        if "memory" in env_raw and isinstance(env_raw["memory"], str):
            env_raw["memory_mb"] = _parse_memory_string(env_raw["memory"])

        config = TaskConfig.model_validate(raw)
        instruction = instruction_path.read_text(encoding="utf-8")

        return Task(
            name=task_dir.name,
            instruction=instruction,
            config=config,
            paths=paths,
        )
