"""JSON response parser for the Terminus agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Command:
    """A single keystroke command parsed from the LLM response."""

    keystrokes: str
    duration: float = 0.5


@dataclass
class ParsedResponse:
    """Structured result of parsing the LLM's JSON output."""

    analysis: str = ""
    plan: str = ""
    commands: list[Command] = field(default_factory=list)
    is_task_complete: bool = False
    raw_json: dict | None = None


class ParseError(Exception):
    """Raised when the LLM response cannot be parsed as valid JSON."""


def parse_response(text: str) -> ParsedResponse:
    """Parse a JSON response from the Terminus agent."""
    text = text.strip()

    data = _try_parse(text)

    if data is None:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            data = _try_parse(match.group(1).strip())

    if data is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            data = _try_parse(text[start : end + 1])

    if data is None or not isinstance(data, dict):
        raise ParseError(f"Could not extract JSON from response:\n{text[:500]}")

    if "commands" not in data:
        raise ParseError(
            f"Response JSON missing 'commands' field:\n{json.dumps(data, indent=2)[:500]}"
        )

    commands = []
    for cmd in data.get("commands", []):
        if isinstance(cmd, dict) and "keystrokes" in cmd:
            commands.append(
                Command(
                    keystrokes=cmd["keystrokes"],
                    duration=float(cmd.get("duration", 0.5)),
                )
            )

    return ParsedResponse(
        analysis=str(data.get("analysis", "")),
        plan=str(data.get("plan", "")),
        commands=commands,
        is_task_complete=bool(data.get("is_task_complete", False)),
        raw_json=data,
    )


def _try_parse(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
