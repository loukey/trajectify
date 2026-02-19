"""LogAnalysisFactory — generates log-parsing tasks with deterministic data."""

from __future__ import annotations

import json
import random
import textwrap
from collections import Counter

from trajectify.task_factory import register_factory
from trajectify.task_factory.base import GeneratedTask, TaskFactory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]
_STATUS_CODES = [200, 200, 200, 201, 301, 302, 304, 400, 401, 403, 404, 500, 502, 503]
_PATHS = [
    "/", "/index.html", "/api/users", "/api/users/1", "/api/orders",
    "/api/products", "/login", "/logout", "/static/style.css",
    "/static/app.js", "/images/logo.png", "/health", "/api/search",
    "/api/auth/token", "/favicon.ico", "/docs", "/api/v2/items",
]
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "curl/7.81.0",
    "python-requests/2.28.1",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
]
_REFERRERS = [
    "-", "https://www.google.com/", "https://github.com/",
    "https://example.com/", "-", "-",
]
_IP_POOL = [f"192.168.1.{i}" for i in range(1, 51)] + [
    f"10.0.{subnet}.{host}"
    for subnet in range(1, 6)
    for host in range(1, 11)
]
_JSON_LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
_JSON_LOG_SERVICES = ["api-gateway", "auth-service", "user-service", "order-service"]

_ANALYSIS_FIELD_GROUPS: list[dict[str, str]] = [
    {
        "key": "group_a",
        "label": "traffic overview",
        "fields": ["total_requests", "unique_ips", "status_codes", "requests_per_method"],
    },
    {
        "key": "group_b",
        "label": "top resources",
        "fields": ["total_requests", "top_paths", "top_ips", "total_bytes"],
    },
    {
        "key": "group_c",
        "label": "full analysis",
        "fields": [
            "total_requests", "unique_ips", "status_codes",
            "top_paths", "top_ips", "total_bytes", "requests_per_method",
        ],
    },
]

_DIFFICULTY_CONFIG = {
    "easy": {
        "timeout_agent": 300,
        "timeout_verifier": 120,
        "expert_min": 5,
        "junior_min": 20,
    },
    "medium": {
        "timeout_agent": 600,
        "timeout_verifier": 300,
        "expert_min": 10,
        "junior_min": 40,
    },
    "hard": {
        "timeout_agent": 900,
        "timeout_verifier": 300,
        "expert_min": 20,
        "junior_min": 60,
    },
}


# ---------------------------------------------------------------------------
# Log generators
# ---------------------------------------------------------------------------

def _generate_nginx_combined(rng: random.Random, num_lines: int) -> list[str]:
    """Generate nginx combined-format log lines."""
    lines = []
    for _ in range(num_lines):
        ip = rng.choice(_IP_POOL)
        method = rng.choices(_HTTP_METHODS, weights=[50, 20, 10, 5, 5, 10])[0]
        path = rng.choice(_PATHS)
        status = rng.choice(_STATUS_CODES)
        size = rng.randint(0, 50000) if status < 400 else rng.choice([0, 0, 162, 548])
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        ua = rng.choice(_USER_AGENTS)
        ref = rng.choice(_REFERRERS)
        line = (
            f'{ip} - - [15/Jan/2025:{hour:02d}:{minute:02d}:{second:02d} +0000] '
            f'"{method} {path} HTTP/1.1" {status} {size} "{ref}" "{ua}"'
        )
        lines.append(line)
    return lines


def _generate_apache_common(rng: random.Random, num_lines: int) -> list[str]:
    """Generate Apache common-format log lines."""
    lines = []
    for _ in range(num_lines):
        ip = rng.choice(_IP_POOL)
        method = rng.choices(_HTTP_METHODS, weights=[50, 20, 10, 5, 5, 10])[0]
        path = rng.choice(_PATHS)
        status = rng.choice(_STATUS_CODES)
        size = rng.randint(0, 50000) if status < 400 else rng.choice([0, 0, 162, 548])
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        line = (
            f'{ip} - - [15/Jan/2025:{hour:02d}:{minute:02d}:{second:02d} +0000] '
            f'"{method} {path} HTTP/1.1" {status} {size}'
        )
        lines.append(line)
    return lines


def _generate_json_structured(rng: random.Random, num_lines: int) -> list[str]:
    """Generate JSON-structured log lines."""
    lines = []
    for _ in range(num_lines):
        ip = rng.choice(_IP_POOL)
        method = rng.choices(_HTTP_METHODS, weights=[50, 20, 10, 5, 5, 10])[0]
        path = rng.choice(_PATHS)
        status = rng.choice(_STATUS_CODES)
        size = rng.randint(0, 50000) if status < 400 else rng.choice([0, 0, 162, 548])
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        entry = {
            "timestamp": f"2025-01-15T{hour:02d}:{minute:02d}:{second:02d}Z",
            "remote_addr": ip,
            "method": method,
            "path": path,
            "status": status,
            "body_bytes_sent": size,
            "level": rng.choice(_JSON_LOG_LEVELS),
            "service": rng.choice(_JSON_LOG_SERVICES),
        }
        lines.append(json.dumps(entry))
    return lines


_LOG_GENERATORS = {
    "nginx_combined": _generate_nginx_combined,
    "apache_common": _generate_apache_common,
    "json_structured": _generate_json_structured,
}


# ---------------------------------------------------------------------------
# Expected-answer computation
# ---------------------------------------------------------------------------

def _parse_log_entries(log_format: str, lines: list[str]) -> list[dict]:
    """Parse generated log lines into uniform dicts for answer computation."""
    entries = []
    for line in lines:
        if log_format == "json_structured":
            obj = json.loads(line)
            entries.append({
                "ip": obj["remote_addr"],
                "method": obj["method"],
                "path": obj["path"],
                "status": obj["status"],
                "size": obj["body_bytes_sent"],
            })
        else:
            # nginx combined or apache common — same request-line position
            parts = line.split('"')
            ip = line.split(" ")[0]
            request_part = parts[1]  # e.g. "GET /path HTTP/1.1"
            req_fields = request_part.split()
            method = req_fields[0]
            path = req_fields[1]
            after_request = parts[2].strip().split()
            status = int(after_request[0])
            size = int(after_request[1])
            entries.append({
                "ip": ip,
                "method": method,
                "path": path,
                "status": status,
                "size": size,
            })
    return entries


def _compute_expected(entries: list[dict], fields: list[str]) -> dict:
    """Compute the expected JSON report from parsed entries."""
    result: dict = {}

    if "total_requests" in fields:
        result["total_requests"] = len(entries)

    if "unique_ips" in fields:
        result["unique_ips"] = len({e["ip"] for e in entries})

    if "status_codes" in fields:
        codes: dict[str, int] = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0}
        for e in entries:
            bucket = f"{e['status'] // 100}xx"
            if bucket in codes:
                codes[bucket] += 1
        result["status_codes"] = codes

    if "top_paths" in fields:
        path_counts = Counter(e["path"] for e in entries)
        top5 = path_counts.most_common(5)
        result["top_paths"] = [{"path": p, "count": c} for p, c in top5]

    if "top_ips" in fields:
        ip_counts = Counter(e["ip"] for e in entries)
        top5 = ip_counts.most_common(5)
        result["top_ips"] = [{"ip": ip, "count": c} for ip, c in top5]

    if "total_bytes" in fields:
        result["total_bytes"] = sum(e["size"] for e in entries)

    if "requests_per_method" in fields:
        method_counts = Counter(e["method"] for e in entries)
        result["requests_per_method"] = dict(sorted(method_counts.items()))

    return result


# ---------------------------------------------------------------------------
# Template renderers
# ---------------------------------------------------------------------------

_FORMAT_DESCRIPTIONS = {
    "nginx_combined": "nginx Combined Log Format",
    "apache_common": "Apache Common Log Format",
    "json_structured": "JSON structured log format (one JSON object per line)",
}

_FIELD_DESCRIPTIONS = {
    "total_requests": "`total_requests` (int): Total number of log lines",
    "unique_ips": "`unique_ips` (int): Count of distinct client IP addresses",
    "status_codes": (
        "`status_codes` (object): Counts grouped by HTTP status class — "
        '`{"2xx": N, "3xx": N, "4xx": N, "5xx": N}`'
    ),
    "top_paths": (
        "`top_paths` (array): Top 5 most requested URL paths sorted by count descending, "
        'each `{"path": "...", "count": N}`'
    ),
    "top_ips": (
        "`top_ips` (array): Top 5 IP addresses by request count sorted descending, "
        'each `{"ip": "...", "count": N}`'
    ),
    "total_bytes": "`total_bytes` (int): Sum of all response body sizes (treat `-` as 0)",
    "requests_per_method": (
        "`requests_per_method` (object): Count of requests grouped by HTTP method "
        '(e.g. `{"GET": N, "POST": N, ...}`)'
    ),
}


def _render_instruction(
    log_format: str,
    fields: list[str],
    difficulty: str,
    num_lines: int,
) -> str:
    """Render the instruction.md from templates."""
    fmt_desc = _FORMAT_DESCRIPTIONS[log_format]
    field_list = "\n".join(f"- {_FIELD_DESCRIPTIONS[f]}" for f in fields)

    # Build the expected JSON structure skeleton
    skeleton: dict = {}
    for f in fields:
        if f in ("total_requests", "unique_ips", "total_bytes"):
            skeleton[f] = "<int>"
        elif f == "status_codes":
            skeleton[f] = {"2xx": "<int>", "3xx": "<int>", "4xx": "<int>", "5xx": "<int>"}
        elif f == "top_paths":
            skeleton[f] = [{"path": "<string>", "count": "<int>"}]
        elif f == "top_ips":
            skeleton[f] = [{"ip": "<string>", "count": "<int>"}]
        elif f == "requests_per_method":
            skeleton[f] = {"GET": "<int>", "POST": "<int>", "...": "..."}
    skeleton_json = json.dumps(skeleton, indent=2)

    extra_notes = ""
    if difficulty == "hard":
        extra_notes = (
            "\n**Additional requirements:**\n"
            "- Your solution must handle malformed lines gracefully (skip them).\n"
            "- Do not use any third-party Python packages — standard library only.\n"
        )
    elif difficulty == "easy":
        extra_notes = (
            "\n**Hint:** You may use any Python packages available in the container.\n"
        )

    return (
        f"You are given a log file at `/data/access.log` in {fmt_desc}.\n"
        f"The file contains {num_lines} HTTP request log entries.\n"
        f"\n"
        f"Write a Python script at `/app/analyze.py` that parses the log and produces\n"
        f"a JSON report at `/app/report.json` with the following fields:\n"
        f"\n"
        f"{field_list}\n"
        f"\n"
        f"Expected JSON structure:\n"
        f"```json\n"
        f"{skeleton_json}\n"
        f"```\n"
        f"{extra_notes}\n"
        f"After writing the script, run it to generate the report:\n"
        f"```bash\n"
        f"python3 /app/analyze.py\n"
        f"```\n"
        f"\n"
        f"The report must be valid JSON written to `/app/report.json`.\n"
    )


def _render_task_toml(difficulty: str, log_format: str, fields_key: str) -> str:
    cfg = _DIFFICULTY_CONFIG[difficulty]
    tags = json.dumps(["log-analysis", "data-processing", log_format, fields_key])
    return textwrap.dedent(f"""\
        version = "1.0"

        [metadata]
        author_name = "generated"
        author_email = "task-factory@trajectify"
        difficulty = "{difficulty}"
        category = "data-processing"
        tags = {tags}
        expert_time_estimate_min = {cfg["expert_min"]:.1f}
        junior_time_estimate_min = {cfg["junior_min"]:.1f}

        [verifier]
        timeout_sec = {cfg["timeout_verifier"]:.1f}

        [agent]
        timeout_sec = {cfg["timeout_agent"]:.1f}

        [environment]
        build_timeout_sec = 300.0
        cpus = 1
        memory = "1G"
        storage = "5G"
    """)


def _render_dockerfile(log_lines: list[str]) -> str:
    """Render Dockerfile that embeds log data."""
    # Escape log lines for heredoc
    log_data = "\n".join(log_lines)
    return textwrap.dedent("""\
        FROM python:3.13-slim

        WORKDIR /app
        RUN mkdir -p /data

        COPY access.log /data/access.log
    """), log_data


def _render_test_script() -> str:
    return textwrap.dedent("""\
        #!/bin/bash

        # Install curl
        apt-get update > /dev/null 2>&1
        apt-get install -y curl > /dev/null 2>&1

        # Install uv
        curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh > /dev/null 2>&1

        source $HOME/.local/bin/env

        if [ "$PWD" = "/" ]; then
            echo "Error: No working directory set."
            exit 1
        fi

        uvx \\
          -p 3.13 \\
          -w pytest==8.4.1 \\
          -w pytest-json-ctrf==0.3.5 \\
          pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

        if [ $? -eq 0 ]; then
          echo 1 > /logs/verifier/reward.txt
        else
          echo 0 > /logs/verifier/reward.txt
        fi
    """)


def _render_test_file(expected: dict, fields: list[str]) -> str:
    """Generate pytest test file that validates the report."""
    expected_json = json.dumps(expected, indent=2)
    checks = []
    for f in fields:
        checks.append(
            f'    assert report["{f}"] == expected["{f}"], "{f} mismatch"'
        )
    checks_str = "\n".join(checks)

    return (
        '"""Auto-generated test for log analysis task."""\n'
        "\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"EXPECTED = {expected_json}\n"
        "\n"
        "\n"
        "def test_report_exists():\n"
        '    assert Path("/app/report.json").exists(), "report.json not found"\n'
        "\n"
        "\n"
        "def test_report_valid_json():\n"
        '    text = Path("/app/report.json").read_text()\n'
        "    json.loads(text)  # raises on invalid JSON\n"
        "\n"
        "\n"
        "def test_report_values():\n"
        '    text = Path("/app/report.json").read_text()\n'
        "    report = json.loads(text)\n"
        "    expected = EXPECTED\n"
        f"{checks_str}\n"
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@register_factory
class LogAnalysisFactory(TaskFactory):
    """Generates log-parsing → JSON-report tasks."""

    @staticmethod
    def factory_name() -> str:
        return "log_analysis"

    def param_space(self) -> dict[str, list]:
        return {
            "log_format": ["nginx_combined", "apache_common", "json_structured"],
            "num_lines": [50, 200, 500],
            "analysis_group": [0, 1, 2],
            "difficulty": ["easy", "medium", "hard"],
            "seed": list(range(1, 11)),
        }

    def generate(self, params: dict, seed: int) -> GeneratedTask:
        log_format: str = params["log_format"]
        num_lines: int = params["num_lines"]
        group_idx: int = params["analysis_group"]
        difficulty: str = params["difficulty"]

        group = _ANALYSIS_FIELD_GROUPS[group_idx]
        fields: list[str] = group["fields"]
        fields_key: str = group["key"]

        rng = random.Random(seed)

        # Generate deterministic log data
        generator = _LOG_GENERATORS[log_format]
        log_lines = generator(rng, num_lines)

        # Compute expected answer
        entries = _parse_log_entries(log_format, log_lines)
        expected = _compute_expected(entries, fields)

        # Build task name
        name = f"log-{log_format.replace('_', '-')}-{num_lines}L-{fields_key}-{difficulty}-s{seed}"

        # Render all files
        instruction = _render_instruction(log_format, fields, difficulty, num_lines)
        task_toml = _render_task_toml(difficulty, log_format, fields_key)
        dockerfile_content, log_data = _render_dockerfile(log_lines)
        test_script = _render_test_script()
        test_file = _render_test_file(expected, fields)

        return GeneratedTask(
            name=name,
            task_toml=task_toml,
            instruction=instruction,
            dockerfile=dockerfile_content,
            test_script=test_script,
            test_files={
                "test_outputs.py": test_file,
                "../environment/access.log": log_data,
            },
        )
