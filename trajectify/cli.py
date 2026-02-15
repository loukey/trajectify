"""CLI entry point for trajectify."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import shortuuid
import yaml

from trajectify.config.loader import load_llm_config
from trajectify.config.models import (
    AgentConfig,
    EnvironmentRunConfig,
    ExporterConfig,
    JobConfig,
    RunConfig,
    VerifierRunConfig,
)
from trajectify.log import logger
from trajectify.orchestrator import Orchestrator


def _build_run_configs(job: JobConfig) -> list[RunConfig]:
    """Expand a JobConfig into one RunConfig per task."""
    configs = []
    for task_path in job.tasks:
        task_path = Path(task_path).resolve()
        run_name = f"{task_path.name}__{shortuuid.uuid()[:8]}"
        configs.append(
            RunConfig(
                run_name=run_name,
                task_path=task_path,
                agent=job.agent,
                environment=job.environment,
                verifier=job.verifier,
                exporter=job.exporter,
                output_dir=job.output_dir,
            )
        )
    return configs


def _load_yaml_config(path: Path) -> JobConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return JobConfig.model_validate(raw)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="trajectify — trajectory collection runner")
    p.add_argument("config", nargs="?", help="Path to a YAML config file")
    p.add_argument("--task", type=str, help="Path to a single task directory")
    p.add_argument("--agent", type=str, default="terminus")
    p.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-20250514")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--n-concurrent", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--collect-rollout", action="store_true")
    p.add_argument("--disable-verifier", action="store_true")
    p.add_argument(
        "--export-formats",
        type=str,
        default="atif",
        help="Comma-separated export formats: atif,sft,rollout",
    )
    return p.parse_args()


async def _main() -> None:
    llm_cfg = load_llm_config()
    llm_cfg.apply_to_env()
    logger.info(
        "LLM config loaded: model=%s, api_base=%s",
        llm_cfg.model, llm_cfg.api_base or "(default)",
    )

    args = _parse_args()
    export_formats = [f.strip() for f in args.export_formats.split(",")]

    if args.config:
        job = _load_yaml_config(Path(args.config))
        if not job.agent.kwargs.get("api_base") and llm_cfg.api_base:
            job.agent.kwargs["api_base"] = llm_cfg.api_base
        if job.agent.model == AgentConfig.model_fields["model"].default:
            job.agent.model = llm_cfg.model
    elif args.task:
        model = args.model
        if model == "anthropic/claude-sonnet-4-20250514":
            model = llm_cfg.model

        job = JobConfig(
            tasks=[Path(args.task)],
            agent=AgentConfig(
                name=args.agent,
                model=model,
                temperature=args.temperature,
                collect_rollout_details=args.collect_rollout,
                kwargs={"api_base": llm_cfg.api_base} if llm_cfg.api_base else {},
            ),
            environment=EnvironmentRunConfig(),
            verifier=VerifierRunConfig(disable=args.disable_verifier),
            exporter=ExporterConfig(formats=export_formats),
            n_concurrent=args.n_concurrent,
            output_dir=Path(args.output_dir),
        )
    else:
        logger.error("Provide a YAML config file or --task path.")
        sys.exit(1)

    configs = _build_run_configs(job)
    orchestrator = Orchestrator(configs=configs, n_concurrent=job.n_concurrent)
    results = await orchestrator.run()

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK" if r.exception_info is None else f"FAIL ({r.exception_info.exception_type})"
        reward = "N/A"
        if r.verifier_result and r.verifier_result.rewards:
            reward = ", ".join(f"{k}={v}" for k, v in r.verifier_result.rewards.items())
        duration = "?"
        if r.finished_at and r.started_at:
            duration = f"{(r.finished_at - r.started_at).total_seconds():.1f}s"
        print(f"  {r.run_name:<40} {status:<20} reward={reward:<10} time={duration}")
    print("=" * 60)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
