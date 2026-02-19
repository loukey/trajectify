"""BugFixFactory — generates code-with-bugs tasks for agent debugging practice."""

from __future__ import annotations

import json
import random
import textwrap

from trajectify.task_factory import register_factory
from trajectify.task_factory.base import GeneratedTask, TaskFactory

# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, dict] = {}


def _register_scenario(name: str):
    """Decorator to register a scenario dict builder."""
    def decorator(fn):
        _SCENARIOS[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Scenario 1: number_stats
# ---------------------------------------------------------------------------

@_register_scenario("number_stats")
def _scenario_number_stats():
    correct_source = textwrap.dedent('''\
        """Read numbers from /data/input.txt and compute statistics."""
        import json
        import math

        def compute_stats(numbers):
            if not numbers:
                return {"count": 0, "sum": 0, "mean": 0.0, "min": None, "max": None, "median": 0.0}
            count = len(numbers)
            total = sum(numbers)
            mean = total / count
            sorted_nums = sorted(numbers)
            if count % 2 == 1:
                median = sorted_nums[count // 2]
            else:
                median = (sorted_nums[count // 2 - 1] + sorted_nums[count // 2]) / 2
            return {
                "count": count,
                "sum": total,
                "mean": round(mean, 4),
                "min": min(numbers),
                "max": max(numbers),
                "median": round(median, 4),
            }

        def main():
            with open("/data/input.txt") as f:
                numbers = [float(line.strip()) for line in f if line.strip()]
            result = compute_stats(numbers)
            with open("/app/report.json", "w") as f:
                json.dump(result, f, indent=2)

        if __name__ == "__main__":
            main()
    ''')

    mutations = [
        {"type": "wrong_operator", "find": "mean = total / count",
         "replace": "mean = total // count", "difficulty": "easy"},
        {"type": "off_by_one", "find": "median = sorted_nums[count // 2]",
         "replace": "median = sorted_nums[count // 2 + 1]", "difficulty": "easy"},
        {"type": "missing_guard", "find": "    if not numbers:\n        return {\"count\": 0, \"sum\": 0, \"mean\": 0.0, \"min\": None, \"max\": None, \"median\": 0.0}",
         "replace": "    # BUG: missing empty check", "difficulty": "medium"},
        {"type": "wrong_operator", "find": "(sorted_nums[count // 2 - 1] + sorted_nums[count // 2]) / 2",
         "replace": "(sorted_nums[count // 2 - 1] + sorted_nums[count // 2]) // 2", "difficulty": "easy"},
        {"type": "wrong_function", "find": "total = sum(numbers)",
         "replace": "total = len(numbers)", "difficulty": "medium"},
    ]

    def generate_input(rng: random.Random, num_items: int) -> str:
        numbers = [round(rng.uniform(-100, 100), 2) for _ in range(num_items)]
        return "\n".join(str(n) for n in numbers)

    def compute_expected(input_data: str) -> dict:
        numbers = [float(line.strip()) for line in input_data.strip().split("\n") if line.strip()]
        count = len(numbers)
        total = sum(numbers)
        mean = total / count
        sorted_nums = sorted(numbers)
        if count % 2 == 1:
            median = sorted_nums[count // 2]
        else:
            median = (sorted_nums[count // 2 - 1] + sorted_nums[count // 2]) / 2
        return {
            "count": count,
            "sum": round(total, 4),
            "mean": round(mean, 4),
            "min": min(numbers),
            "max": max(numbers),
            "median": round(median, 4),
        }

    return {
        "correct_source": correct_source,
        "mutations": mutations,
        "generate_input": generate_input,
        "compute_expected": compute_expected,
        "description": "a script that reads numbers from a file and computes statistics (count, sum, mean, min, max, median)",
        "input_path": "/data/input.txt",
        "script_path": "/app/solution.py",
        "output_path": "/app/report.json",
    }


# ---------------------------------------------------------------------------
# Scenario 2: word_counter
# ---------------------------------------------------------------------------

@_register_scenario("word_counter")
def _scenario_word_counter():
    correct_source = textwrap.dedent('''\
        """Read text from /data/input.txt and compute word statistics."""
        import json
        from collections import Counter

        def analyze_text(text):
            lines = text.strip().split("\\n")
            words = text.split()
            word_counts = Counter(w.lower() for w in words)
            top_words = word_counts.most_common(5)
            return {
                "line_count": len(lines),
                "word_count": len(words),
                "unique_words": len(word_counts),
                "char_count": len(text),
                "top_words": [{"word": w, "count": c} for w, c in top_words],
            }

        def main():
            with open("/data/input.txt") as f:
                text = f.read()
            result = analyze_text(text)
            with open("/app/report.json", "w") as f:
                json.dump(result, f, indent=2)

        if __name__ == "__main__":
            main()
    ''')

    mutations = [
        {"type": "case_bug", "find": "Counter(w.lower() for w in words)",
         "replace": "Counter(w for w in words)", "difficulty": "easy"},
        {"type": "wrong_method", "find": "words = text.split()",
         "replace": "words = text.split(\"\\n\")", "difficulty": "easy"},
        {"type": "off_by_one", "find": "top_words = word_counts.most_common(5)",
         "replace": "top_words = word_counts.most_common(4)", "difficulty": "easy"},
        {"type": "wrong_field", "find": "\"char_count\": len(text)",
         "replace": "\"char_count\": len(words)", "difficulty": "medium"},
        {"type": "wrong_split", "find": "lines = text.strip().split(\"\\n\")",
         "replace": "lines = text.split(\"\\n\")", "difficulty": "medium"},
    ]

    _WORDS = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "python", "code", "data", "test", "file",
        "read", "write", "open", "close", "run", "fast", "slow",
        "big", "small", "red", "blue", "green", "dark", "light",
    ]

    def generate_input(rng: random.Random, num_items: int) -> str:
        lines = []
        for _ in range(num_items):
            n_words = rng.randint(3, 12)
            line = " ".join(rng.choice(_WORDS) for _ in range(n_words))
            lines.append(line)
        return "\n".join(lines)

    def compute_expected(input_data: str) -> dict:
        text = input_data
        lines = text.strip().split("\n")
        words = text.split()
        word_counts = Counter(w.lower() for w in words)
        top_words = word_counts.most_common(5)
        return {
            "line_count": len(lines),
            "word_count": len(words),
            "unique_words": len(word_counts),
            "char_count": len(text),
            "top_words": [{"word": w, "count": c} for w, c in top_words],
        }

    return {
        "correct_source": correct_source,
        "mutations": mutations,
        "generate_input": generate_input,
        "compute_expected": compute_expected,
        "description": "a script that reads text from a file and computes word statistics (line count, word count, unique words, most common words)",
        "input_path": "/data/input.txt",
        "script_path": "/app/solution.py",
        "output_path": "/app/report.json",
    }


# ---------------------------------------------------------------------------
# Scenario 3: csv_aggregator
# ---------------------------------------------------------------------------

@_register_scenario("csv_aggregator")
def _scenario_csv_aggregator():
    correct_source = textwrap.dedent('''\
        """Read CSV from /data/input.csv and compute per-category aggregates."""
        import csv
        import json
        from collections import defaultdict

        def aggregate(filepath):
            groups = defaultdict(list)
            with open(filepath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cat = row["category"]
                    score = float(row["score"])
                    groups[cat].append(score)
            result = {}
            for cat in sorted(groups.keys()):
                scores = groups[cat]
                result[cat] = {
                    "count": len(scores),
                    "total": round(sum(scores), 2),
                    "average": round(sum(scores) / len(scores), 2),
                    "min": min(scores),
                    "max": max(scores),
                }
            return result

        def main():
            result = aggregate("/data/input.csv")
            with open("/app/report.json", "w") as f:
                json.dump(result, f, indent=2)

        if __name__ == "__main__":
            main()
    ''')

    mutations = [
        {"type": "wrong_operator", "find": "round(sum(scores) / len(scores), 2)",
         "replace": "round(sum(scores) // len(scores), 2)", "difficulty": "easy"},
        {"type": "wrong_field", "find": "score = float(row[\"score\"])",
         "replace": "score = int(row[\"score\"])", "difficulty": "easy"},
        {"type": "wrong_sort", "find": "for cat in sorted(groups.keys()):",
         "replace": "for cat in groups.keys():", "difficulty": "medium"},
        {"type": "wrong_init", "find": "groups = defaultdict(list)",
         "replace": "groups = {}", "difficulty": "medium"},
        {"type": "off_by_one", "find": "\"total\": round(sum(scores), 2)",
         "replace": "\"total\": round(sum(scores) - scores[0], 2)", "difficulty": "hard"},
    ]

    _CATEGORIES = ["alpha", "beta", "gamma", "delta"]

    def generate_input(rng: random.Random, num_items: int) -> str:
        lines = ["name,category,score"]
        for i in range(num_items):
            name = f"item_{i:03d}"
            cat = rng.choice(_CATEGORIES)
            score = round(rng.uniform(1, 100), 2)
            lines.append(f"{name},{cat},{score}")
        return "\n".join(lines)

    def compute_expected(input_data: str) -> dict:
        lines = input_data.strip().split("\n")
        groups: dict[str, list[float]] = {}
        for line in lines[1:]:  # skip header
            parts = line.split(",")
            cat = parts[1]
            score = float(parts[2])
            groups.setdefault(cat, []).append(score)
        result = {}
        for cat in sorted(groups.keys()):
            scores = groups[cat]
            result[cat] = {
                "count": len(scores),
                "total": round(sum(scores), 2),
                "average": round(sum(scores) / len(scores), 2),
                "min": min(scores),
                "max": max(scores),
            }
        return result

    return {
        "correct_source": correct_source,
        "mutations": mutations,
        "generate_input": generate_input,
        "compute_expected": compute_expected,
        "description": "a script that reads CSV data and computes per-category aggregates (count, total, average, min, max)",
        "input_path": "/data/input.csv",
        "script_path": "/app/solution.py",
        "output_path": "/app/report.json",
    }


# ---------------------------------------------------------------------------
# Scenario 4: json_transformer
# ---------------------------------------------------------------------------

@_register_scenario("json_transformer")
def _scenario_json_transformer():
    correct_source = textwrap.dedent('''\
        """Read JSON records from /data/input.json, filter and transform."""
        import json

        def transform(records, min_score, sort_key):
            filtered = [r for r in records if r.get("score", 0) >= min_score]
            filtered.sort(key=lambda r: r.get(sort_key, ""), reverse=True)
            for r in filtered:
                r["grade"] = "A" if r["score"] >= 90 else "B" if r["score"] >= 70 else "C" if r["score"] >= 50 else "F"
            return {
                "total_input": len(records),
                "total_filtered": len(filtered),
                "records": filtered,
            }

        def main():
            with open("/data/input.json") as f:
                data = json.load(f)
            result = transform(data["records"], data["min_score"], data["sort_key"])
            with open("/app/report.json", "w") as f:
                json.dump(result, f, indent=2)

        if __name__ == "__main__":
            main()
    ''')

    mutations = [
        {"type": "wrong_comparison", "find": "if r.get(\"score\", 0) >= min_score",
         "replace": "if r.get(\"score\", 0) > min_score", "difficulty": "easy"},
        {"type": "wrong_sort_order", "find": "reverse=True",
         "replace": "reverse=False", "difficulty": "easy"},
        {"type": "wrong_threshold", "find": "\"A\" if r[\"score\"] >= 90 else \"B\" if r[\"score\"] >= 70 else \"C\" if r[\"score\"] >= 50 else \"F\"",
         "replace": "\"A\" if r[\"score\"] >= 90 else \"B\" if r[\"score\"] >= 80 else \"C\" if r[\"score\"] >= 50 else \"F\"",
         "difficulty": "medium"},
        {"type": "wrong_default", "find": "r.get(\"score\", 0)",
         "replace": "r.get(\"score\", 100)", "difficulty": "medium"},
        {"type": "missing_key", "find": "r.get(sort_key, \"\")",
         "replace": "r[sort_key]", "difficulty": "hard"},
    ]

    _NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

    def generate_input(rng: random.Random, num_items: int) -> str:
        records = []
        for i in range(num_items):
            records.append({
                "name": rng.choice(_NAMES) + f"_{i}",
                "score": rng.randint(20, 100),
                "department": rng.choice(["engineering", "marketing", "sales", "hr"]),
            })
        data = {
            "records": records,
            "min_score": rng.choice([40, 50, 60]),
            "sort_key": "score",
        }
        return json.dumps(data, indent=2)

    def compute_expected(input_data: str) -> dict:
        data = json.loads(input_data)
        records = data["records"]
        min_score = data["min_score"]
        sort_key = data["sort_key"]
        filtered = [r for r in records if r.get("score", 0) >= min_score]
        filtered.sort(key=lambda r: r.get(sort_key, ""), reverse=True)
        for r in filtered:
            r["grade"] = "A" if r["score"] >= 90 else "B" if r["score"] >= 70 else "C" if r["score"] >= 50 else "F"
        return {
            "total_input": len(records),
            "total_filtered": len(filtered),
            "records": filtered,
        }

    return {
        "correct_source": correct_source,
        "mutations": mutations,
        "generate_input": generate_input,
        "compute_expected": compute_expected,
        "description": "a script that reads JSON records, filters by minimum score, sorts, assigns letter grades, and outputs the result",
        "input_path": "/data/input.json",
        "script_path": "/app/solution.py",
        "output_path": "/app/report.json",
    }


# ---------------------------------------------------------------------------
# Scenario 5: matrix_ops
# ---------------------------------------------------------------------------

@_register_scenario("matrix_ops")
def _scenario_matrix_ops():
    correct_source = textwrap.dedent('''\
        """Read a matrix from /data/input.csv and perform operations."""
        import csv
        import json

        def read_matrix(path):
            matrix = []
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    matrix.append([float(x) for x in row])
            return matrix

        def transpose(m):
            rows, cols = len(m), len(m[0])
            return [[m[r][c] for r in range(rows)] for c in range(cols)]

        def row_sums(m):
            return [round(sum(row), 4) for row in m]

        def col_sums(m):
            return row_sums(transpose(m))

        def flatten_sum(m):
            return round(sum(sum(row) for row in m), 4)

        def main():
            m = read_matrix("/data/input.csv")
            t = transpose(m)
            result = {
                "rows": len(m),
                "cols": len(m[0]),
                "row_sums": row_sums(m),
                "col_sums": col_sums(m),
                "total_sum": flatten_sum(m),
                "transposed": t,
            }
            with open("/app/report.json", "w") as f:
                json.dump(result, f, indent=2)

        if __name__ == "__main__":
            main()
    ''')

    mutations = [
        {"type": "swap_indices", "find": "return [[m[r][c] for r in range(rows)] for c in range(cols)]",
         "replace": "return [[m[c][r] for r in range(rows)] for c in range(cols)]", "difficulty": "easy"},
        {"type": "wrong_function", "find": "def col_sums(m):\n    return row_sums(transpose(m))",
         "replace": "def col_sums(m):\n    return row_sums(m)", "difficulty": "easy"},
        {"type": "off_by_one", "find": "\"cols\": len(m[0])",
         "replace": "\"cols\": len(m[0]) - 1", "difficulty": "easy"},
        {"type": "wrong_rounding", "find": "return round(sum(sum(row) for row in m), 4)",
         "replace": "return int(sum(sum(row) for row in m))", "difficulty": "medium"},
        {"type": "wrong_cast", "find": "matrix.append([float(x) for x in row])",
         "replace": "matrix.append([int(x) for x in row])", "difficulty": "medium"},
    ]

    def generate_input(rng: random.Random, num_items: int) -> str:
        rows = max(3, num_items // 5)
        cols = max(3, num_items // 8)
        lines = []
        for _ in range(rows):
            row = [str(round(rng.uniform(-50, 50), 2)) for _ in range(cols)]
            lines.append(",".join(row))
        return "\n".join(lines)

    def compute_expected(input_data: str) -> dict:
        matrix = []
        for line in input_data.strip().split("\n"):
            matrix.append([float(x) for x in line.split(",")])
        rows, cols = len(matrix), len(matrix[0])
        t = [[matrix[r][c] for r in range(rows)] for c in range(cols)]
        return {
            "rows": rows,
            "cols": cols,
            "row_sums": [round(sum(row), 4) for row in matrix],
            "col_sums": [round(sum(matrix[r][c] for r in range(rows)), 4) for c in range(cols)],
            "total_sum": round(sum(sum(row) for row in matrix), 4),
            "transposed": t,
        }

    return {
        "correct_source": correct_source,
        "mutations": mutations,
        "generate_input": generate_input,
        "compute_expected": compute_expected,
        "description": "a script that reads a numeric matrix from CSV and computes transpose, row sums, column sums, and total sum",
        "input_path": "/data/input.csv",
        "script_path": "/app/solution.py",
        "output_path": "/app/report.json",
    }


# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

_DIFFICULTY_CONFIG = {
    "easy": {"timeout_agent": 300, "timeout_verifier": 120, "expert_min": 5, "junior_min": 15},
    "medium": {"timeout_agent": 600, "timeout_verifier": 300, "expert_min": 10, "junior_min": 30},
    "hard": {"timeout_agent": 900, "timeout_verifier": 300, "expert_min": 15, "junior_min": 45},
}

# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _apply_mutations(
    source: str,
    mutations: list[dict],
    difficulty: str,
    mutation_count: int,
    rng: random.Random,
) -> tuple[str, list[str]]:
    """Apply mutations to source code. Returns (buggy_source, list_of_bug_descriptions)."""
    eligible = [m for m in mutations if _difficulty_rank(m["difficulty"]) <= _difficulty_rank(difficulty)]
    if not eligible:
        eligible = mutations[:1]

    rng.shuffle(eligible)
    selected = eligible[:mutation_count]

    buggy = source
    descriptions = []
    for m in selected:
        if m["find"] in buggy:
            buggy = buggy.replace(m["find"], m["replace"], 1)
            descriptions.append(f"{m['type']}: {m['find'][:50]}...")

    return buggy, descriptions


def _difficulty_rank(d: str) -> int:
    return {"easy": 1, "medium": 2, "hard": 3}.get(d, 2)


def _render_instruction(scenario: dict, difficulty: str, mutation_count: int) -> str:
    desc = scenario["description"]
    script_path = scenario["script_path"]
    output_path = scenario["output_path"]
    input_path = scenario["input_path"]

    hint = ""
    if difficulty == "easy":
        hint = f"\n**Hint:** There is {mutation_count} bug in the code. Read the code carefully and trace the logic.\n"
    elif difficulty == "medium":
        hint = f"\n**Hint:** There are {mutation_count} bug(s) in the code.\n"

    return (
        f"You are given a Python script at `{script_path}` that is supposed to be {desc}.\n"
        f"\n"
        f"The script reads input from `{input_path}` and writes output to `{output_path}`.\n"
        f"\n"
        f"However, the script contains **bug(s)** that cause it to produce incorrect results.\n"
        f"The test suite is already provided and defines the expected behavior.\n"
        f"\n"
        f"Your task:\n"
        f"1. Read and understand the script at `{script_path}`\n"
        f"2. Identify and fix the bug(s)\n"
        f"3. Run the script to generate the output: `python3 {script_path}`\n"
        f"4. Verify the output is correct by examining `{output_path}`\n"
        f"{hint}\n"
        f"The test suite will automatically validate your fix.\n"
    )


def _render_test_file(expected: dict) -> str:
    expected_json = json.dumps(expected, indent=2)
    return (
        '"""Auto-generated test for bug-fix task."""\n'
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
        "    json.loads(text)\n"
        "\n"
        "\n"
        "def _approx_equal(a, b, tol=1e-2):\n"
        '    """Recursively compare with tolerance for floats."""\n'
        "    if isinstance(a, float) and isinstance(b, (int, float)):\n"
        "        return abs(a - b) < tol\n"
        "    if isinstance(a, dict) and isinstance(b, dict):\n"
        "        return a.keys() == b.keys() and all(_approx_equal(a[k], b[k], tol) for k in a)\n"
        "    if isinstance(a, list) and isinstance(b, list):\n"
        "        return len(a) == len(b) and all(_approx_equal(x, y, tol) for x, y in zip(a, b))\n"
        "    return a == b\n"
        "\n"
        "\n"
        "def test_report_values():\n"
        '    text = Path("/app/report.json").read_text()\n'
        "    report = json.loads(text)\n"
        "    for key in EXPECTED:\n"
        '        assert key in report, f"missing key: {key}"\n'
        "        assert _approx_equal(report[key], EXPECTED[key]), (\n"
        '            f"{key} mismatch: got {report[key]!r}, expected {EXPECTED[key]!r}"\n'
        "        )\n"
    )


def _render_task_toml(difficulty: str, scenario_name: str) -> str:
    cfg = _DIFFICULTY_CONFIG[difficulty]
    tags = json.dumps(["bug-fix", "debugging", scenario_name])
    return textwrap.dedent(f"""\
        version = "1.0"

        [metadata]
        author_name = "generated"
        author_email = "task-factory@trajectify"
        difficulty = "{difficulty}"
        category = "debugging"
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


def _render_dockerfile() -> str:
    return textwrap.dedent("""\
        FROM python:3.13-slim

        WORKDIR /app
        RUN mkdir -p /data

        COPY input_data /data/input.txt
        COPY solution.py /app/solution.py
    """)


def _render_test_script() -> str:
    return textwrap.dedent("""\
        #!/bin/bash

        apt-get update > /dev/null 2>&1
        apt-get install -y curl > /dev/null 2>&1
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@register_factory
class BugFixFactory(TaskFactory):
    """Generates code-debugging tasks: working code with injected bugs."""

    @staticmethod
    def factory_name() -> str:
        return "bug_fix"

    def param_space(self) -> dict[str, list]:
        return {
            "scenario": list(_SCENARIOS.keys()),
            "mutation_count": [1, 2, 3],
            "num_items": [20, 50, 100],
            "difficulty": ["easy", "medium", "hard"],
            "seed": list(range(1, 11)),
        }

    def generate(self, params: dict, seed: int) -> GeneratedTask:
        scenario_name: str = params["scenario"]
        mutation_count: int = params["mutation_count"]
        num_items: int = params["num_items"]
        difficulty: str = params["difficulty"]

        scenario = _SCENARIOS[scenario_name]()
        rng = random.Random(seed)

        # Generate input data
        input_data = scenario["generate_input"](rng, num_items)

        # Compute expected output from correct code
        expected = scenario["compute_expected"](input_data)

        # Apply mutations to create buggy code
        buggy_source, _bug_descs = _apply_mutations(
            scenario["correct_source"],
            scenario["mutations"],
            difficulty,
            mutation_count,
            random.Random(seed + 1000),  # separate rng for mutation selection
        )

        # Determine file extension for input
        input_path = scenario["input_path"]
        input_ext = input_path.rsplit(".", 1)[-1] if "." in input_path else "txt"

        # Build task name
        name = f"bugfix-{scenario_name}-{mutation_count}mut-{num_items}n-{difficulty}-s{seed}"

        # Render Dockerfile based on input file type
        dockerfile = (
            "FROM python:3.13-slim\n"
            "\n"
            "WORKDIR /app\n"
            "RUN mkdir -p /data\n"
            "\n"
            f"COPY input_data {input_path}\n"
            f"COPY solution.py {scenario['script_path']}\n"
        )

        return GeneratedTask(
            name=name,
            task_toml=_render_task_toml(difficulty, scenario_name),
            instruction=_render_instruction(scenario, difficulty, mutation_count),
            dockerfile=dockerfile,
            test_script=_render_test_script(),
            test_files={
                "test_outputs.py": _render_test_file(expected),
                "../environment/input_data": input_data,
                "../environment/solution.py": buggy_source,
            },
        )
