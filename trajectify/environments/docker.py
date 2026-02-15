"""Docker environment implementation using docker compose."""

from __future__ import annotations

import asyncio
import asyncio.subprocess
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from trajectify.environments.base import BaseEnvironment, ExecResult

_TEXT_EXTENSIONS = frozenset({
    ".sh", ".bash", ".py", ".toml", ".cfg", ".txt", ".md",
    ".yaml", ".yml", ".json", ".ini", ".conf", ".env",
    ".dockerfile",
})
_TEXT_EXACT_NAMES = frozenset({
    "Dockerfile", "Makefile", ".dockerignore", ".gitignore",
})


class DockerEnvironment(BaseEnvironment):
    """Runs tasks inside a Docker container managed by docker compose."""

    _COMPOSE_DIR = Path(__file__).parent / "compose"
    _BASE_YAML = _COMPOSE_DIR / "base.yaml"
    _BUILD_YAML = _COMPOSE_DIR / "build.yaml"

    _build_locks: dict[str, asyncio.Lock] = {}

    def _compose_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "IMAGE_NAME": f"tj__{self.environment_name}",
                "CONTEXT_DIR": str(self.environment_dir.resolve()),
                "HOST_AGENT_DIR": str(self.host_agent_dir.resolve()),
                "HOST_VERIFIER_DIR": str(self.host_verifier_dir.resolve()),
                "CPUS": str(self.cpus),
                "MEMORY": f"{self.memory_mb}M",
            }
        )
        return env

    def _project_name(self) -> str:
        return self.session_id.lower().replace(".", "-")

    async def _compose(
        self,
        args: list[str],
        *,
        check: bool = True,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        cmd = [
            "docker", "compose",
            "-p", self._project_name(),
            "--project-directory", str(self.environment_dir.resolve()),
            "-f", str(self._BASE_YAML.resolve()),
            "-f", str(self._BUILD_YAML.resolve()),
            *args,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=self._compose_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            if timeout_sec:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_sec,
                )
            else:
                stdout_bytes, _ = await proc.communicate()
        except asyncio.TimeoutError:
            proc.terminate()
            try:
                stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                stdout_bytes, _ = await proc.communicate()
            raise RuntimeError(
                f"docker compose command timed out after {timeout_sec}s"
            )

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        result = ExecResult(stdout=stdout, return_code=proc.returncode or 0)

        if check and result.return_code != 0:
            raise RuntimeError(
                f"docker compose failed (rc={result.return_code}): "
                f"{' '.join(cmd)}\n{result.stdout}"
            )
        return result

    @staticmethod
    def _fix_line_endings(target_dir: Path) -> None:
        """Convert CRLF to LF for text files in *target_dir* (in-place)."""
        for path in target_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _TEXT_EXTENSIONS and path.name not in _TEXT_EXACT_NAMES:
                continue
            data = path.read_bytes()
            if b"\r\n" in data:
                path.write_bytes(data.replace(b"\r\n", b"\n"))

    async def start(self, force_build: bool = False) -> None:
        image_name = f"tj__{self.environment_name}"
        lock = self._build_locks.setdefault(image_name, asyncio.Lock())

        # On Windows, copy the build context to a temp dir and fix CRLF
        # so that shell scripts / Dockerfiles work inside the Linux container.
        tmp_ctx: tempfile.TemporaryDirectory | None = None
        if sys.platform == "win32":
            tmp_ctx = tempfile.TemporaryDirectory(prefix="tj_ctx_")
            ctx_path = Path(tmp_ctx.name) / self.environment_dir.name
            shutil.copytree(self.environment_dir, ctx_path)
            self._fix_line_endings(ctx_path)
        else:
            ctx_path = None

        try:
            async with lock:
                if ctx_path is not None:
                    # Override CONTEXT_DIR just for the build step
                    orig_env_dir = self.environment_dir
                    self.environment_dir = ctx_path
                    try:
                        await self._compose(["build"])
                    finally:
                        self.environment_dir = orig_env_dir
                else:
                    await self._compose(["build"])
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        try:
            await self._compose(["down", "--remove-orphans"], check=False)
        except RuntimeError:
            pass

        await self._compose(["up", "-d"])
        self.logger.info("Environment %s started", self.environment_name)

    async def stop(self, delete: bool = True) -> None:
        try:
            if delete:
                await self._compose(
                    ["down", "--rmi", "all", "--volumes", "--remove-orphans"],
                    check=False,
                )
            else:
                await self._compose(["down"], check=False)
        except RuntimeError as exc:
            self.logger.warning("docker compose down failed: %s", exc)

    async def upload_file(self, source: Path | str, target: str) -> None:
        await self._compose(["cp", str(source), f"main:{target}"])

    async def upload_dir(self, source: Path | str, target: str) -> None:
        await self._compose(["cp", f"{source}/.", f"main:{target}"])

    async def download_file(self, source: str, target: Path | str) -> None:
        await self._compose(["cp", f"main:{source}", str(target)])

    async def download_dir(self, source: str, target: Path | str) -> None:
        await self._compose(["cp", f"main:{source}/.", str(target)])

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        exec_cmd: list[str] = ["exec", "-T"]

        if cwd:
            exec_cmd.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                exec_cmd.extend(["-e", f"{key}={shlex.quote(value)}"])

        exec_cmd.extend(["main", "bash", "-lc", command])

        return await self._compose(exec_cmd, check=False, timeout_sec=timeout_sec)
