"""TaskFactory ABC — base class for batch task generation."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from trajectify.log import logger


@dataclass
class GeneratedTask:
    """All files needed for a single TB 2.0 task."""

    name: str  # kebab-case task name
    task_toml: str  # task.toml content
    instruction: str  # instruction.md content
    dockerfile: str  # Dockerfile content
    test_script: str  # test.sh content
    test_files: dict[str, str] = field(default_factory=dict)  # extra {filename: content}


class TaskFactory(ABC):
    """Base class for factories that batch-generate TB 2.0 tasks."""

    @staticmethod
    @abstractmethod
    def factory_name() -> str:
        """Unique registry key for this factory."""

    @abstractmethod
    def param_space(self) -> dict[str, list]:
        """Return parameter space: {dimension_name: [candidate_values]}."""

    @abstractmethod
    def generate(self, params: dict, seed: int) -> GeneratedTask:
        """Generate a single task from concrete params + random seed."""

    def generate_all(
        self,
        output_dir: Path,
        max_count: int | None = None,
    ) -> list[Path]:
        """Iterate over the Cartesian product of param_space, call generate(),
        and write each task to output_dir/<task-name>/."""
        space = self.param_space()
        keys = list(space.keys())
        values = [space[k] for k in keys]

        generated: list[Path] = []
        for i, combo in enumerate(itertools.product(*values)):
            if max_count is not None and i >= max_count:
                break
            params = dict(zip(keys, combo))
            seed = params.pop("seed", 0)
            task = self.generate(params, seed)
            task_dir = self._write_task(task, output_dir)
            generated.append(task_dir)

        logger.info(
            "Factory '%s' generated %d tasks in %s",
            self.factory_name(),
            len(generated),
            output_dir,
        )
        return generated

    @staticmethod
    def _write_task(task: GeneratedTask, output_dir: Path) -> Path:
        """Write a GeneratedTask to disk in TB 2.0 directory layout."""
        task_dir = output_dir / task.name
        env_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"

        task_dir.mkdir(parents=True, exist_ok=True)
        env_dir.mkdir(exist_ok=True)
        tests_dir.mkdir(exist_ok=True)

        (task_dir / "task.toml").write_text(task.task_toml, encoding="utf-8")
        (task_dir / "instruction.md").write_text(task.instruction, encoding="utf-8")
        (env_dir / "Dockerfile").write_text(task.dockerfile, encoding="utf-8")
        (tests_dir / "test.sh").write_text(task.test_script, encoding="utf-8")

        for filename, content in task.test_files.items():
            (tests_dir / filename).write_text(content, encoding="utf-8")

        return task_dir
