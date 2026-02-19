"""CodeRemovalFactory — generates tasks where function bodies are removed."""

from __future__ import annotations

import json
import random
import textwrap

from trajectify.task_factory import register_factory
from trajectify.task_factory.base import GeneratedTask, TaskFactory

# ---------------------------------------------------------------------------
# Module definitions — each module has multiple independent functions
# ---------------------------------------------------------------------------

_MODULES: dict[str, dict] = {}


def _register_module(name: str):
    def decorator(fn):
        _MODULES[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Module 1: string_utils
# ---------------------------------------------------------------------------

@_register_module("string_utils")
def _module_string_utils():
    functions = [
        {
            "name": "reverse_words",
            "signature": "def reverse_words(text: str) -> str:",
            "docstring": '    """Reverse the order of words in a string, preserving spacing."""',
            "body": textwrap.dedent("""\
                    words = text.split()
                    return " ".join(reversed(words))
            """),
            "test_gen": lambda rng: _gen_string_test(rng, "reverse_words",
                lambda t: " ".join(reversed(t.split()))),
        },
        {
            "name": "count_vowels",
            "signature": "def count_vowels(text: str) -> int:",
            "docstring": '    """Count the number of vowels (a, e, i, o, u) in text, case-insensitive."""',
            "body": textwrap.dedent("""\
                    return sum(1 for c in text.lower() if c in "aeiou")
            """),
            "test_gen": lambda rng: _gen_string_test(rng, "count_vowels",
                lambda t: sum(1 for c in t.lower() if c in "aeiou")),
        },
        {
            "name": "title_case",
            "signature": "def title_case(text: str) -> str:",
            "docstring": '    """Convert text to title case (capitalize first letter of each word)."""',
            "body": textwrap.dedent("""\
                    return " ".join(w.capitalize() for w in text.split())
            """),
            "test_gen": lambda rng: _gen_string_test(rng, "title_case",
                lambda t: " ".join(w.capitalize() for w in t.split())),
        },
        {
            "name": "is_palindrome",
            "signature": "def is_palindrome(text: str) -> bool:",
            "docstring": '    """Check if text is a palindrome (ignoring case and spaces)."""',
            "body": textwrap.dedent("""\
                    cleaned = text.lower().replace(" ", "")
                    return cleaned == cleaned[::-1]
            """),
            "test_gen": lambda rng: _gen_palindrome_test(rng),
        },
        {
            "name": "char_frequency",
            "signature": "def char_frequency(text: str) -> dict:",
            "docstring": '    """Return a dict of character frequencies (lowercase, excluding spaces)."""',
            "body": textwrap.dedent("""\
                    freq = {}
                    for c in text.lower():
                        if c != " ":
                            freq[c] = freq.get(c, 0) + 1
                    return dict(sorted(freq.items()))
            """),
            "test_gen": lambda rng: _gen_string_test(rng, "char_frequency",
                lambda t: dict(sorted({c: t.lower().count(c) for c in set(t.lower()) if c != " "}.items()))),
        },
    ]
    return {"functions": functions, "description": "string utility functions"}


_WORDS = ["hello", "world", "python", "racecar", "code", "level", "radar",
          "test", "data", "madam", "civic", "refer", "noon", "kayak"]


def _gen_string_test(rng: random.Random, fn_name: str, expected_fn):
    words = [rng.choice(_WORDS) for _ in range(rng.randint(2, 6))]
    text = " ".join(words)
    expected = expected_fn(text)
    return {"input": text, "expected": expected, "fn_name": fn_name}


def _gen_palindrome_test(rng: random.Random):
    if rng.random() < 0.5:
        # Generate a palindrome
        base = rng.choice(["racecar", "level", "madam", "civic", "radar", "noon", "kayak"])
        return {"input": base, "expected": True, "fn_name": "is_palindrome"}
    else:
        text = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 4)))
        cleaned = text.lower().replace(" ", "")
        return {"input": text, "expected": cleaned == cleaned[::-1], "fn_name": "is_palindrome"}


# ---------------------------------------------------------------------------
# Module 2: list_utils
# ---------------------------------------------------------------------------

@_register_module("list_utils")
def _module_list_utils():
    functions = [
        {
            "name": "flatten",
            "signature": "def flatten(nested: list) -> list:",
            "docstring": '    """Flatten a nested list into a single-level list."""',
            "body": textwrap.dedent("""\
                    result = []
                    for item in nested:
                        if isinstance(item, list):
                            result.extend(flatten(item))
                        else:
                            result.append(item)
                    return result
            """),
        },
        {
            "name": "unique_sorted",
            "signature": "def unique_sorted(items: list) -> list:",
            "docstring": '    """Return sorted unique elements from the list."""',
            "body": textwrap.dedent("""\
                    return sorted(set(items))
            """),
        },
        {
            "name": "chunk",
            "signature": "def chunk(items: list, size: int) -> list:",
            "docstring": '    """Split list into chunks of given size. Last chunk may be smaller."""',
            "body": textwrap.dedent("""\
                    return [items[i:i + size] for i in range(0, len(items), size)]
            """),
        },
        {
            "name": "interleave",
            "signature": "def interleave(a: list, b: list) -> list:",
            "docstring": '    """Interleave two lists. If lengths differ, append remaining elements."""',
            "body": textwrap.dedent("""\
                    result = []
                    for i in range(max(len(a), len(b))):
                        if i < len(a):
                            result.append(a[i])
                        if i < len(b):
                            result.append(b[i])
                    return result
            """),
        },
        {
            "name": "running_average",
            "signature": "def running_average(numbers: list) -> list:",
            "docstring": '    """Compute running average: result[i] = mean of numbers[0..i]."""',
            "body": textwrap.dedent("""\
                    result = []
                    total = 0
                    for i, n in enumerate(numbers):
                        total += n
                        result.append(round(total / (i + 1), 4))
                    return result
            """),
        },
    ]
    return {"functions": functions, "description": "list utility functions"}


# ---------------------------------------------------------------------------
# Module 3: math_utils
# ---------------------------------------------------------------------------

@_register_module("math_utils")
def _module_math_utils():
    functions = [
        {
            "name": "gcd",
            "signature": "def gcd(a: int, b: int) -> int:",
            "docstring": '    """Compute greatest common divisor using Euclidean algorithm."""',
            "body": textwrap.dedent("""\
                    while b:
                        a, b = b, a % b
                    return a
            """),
        },
        {
            "name": "lcm",
            "signature": "def lcm(a: int, b: int) -> int:",
            "docstring": '    """Compute least common multiple."""',
            "body": textwrap.dedent("""\
                    return a * b // gcd(a, b)
            """),
        },
        {
            "name": "is_prime",
            "signature": "def is_prime(n: int) -> bool:",
            "docstring": '    """Check if n is a prime number."""',
            "body": textwrap.dedent("""\
                    if n < 2:
                        return False
                    for i in range(2, int(n ** 0.5) + 1):
                        if n % i == 0:
                            return False
                    return True
            """),
        },
        {
            "name": "fibonacci",
            "signature": "def fibonacci(n: int) -> list:",
            "docstring": '    """Return the first n Fibonacci numbers."""',
            "body": textwrap.dedent("""\
                    if n <= 0:
                        return []
                    if n == 1:
                        return [0]
                    fibs = [0, 1]
                    for _ in range(2, n):
                        fibs.append(fibs[-1] + fibs[-2])
                    return fibs
            """),
        },
        {
            "name": "prime_factors",
            "signature": "def prime_factors(n: int) -> list:",
            "docstring": '    """Return the prime factorization of n as a sorted list."""',
            "body": textwrap.dedent("""\
                    factors = []
                    d = 2
                    while d * d <= n:
                        while n % d == 0:
                            factors.append(d)
                            n //= d
                        d += 1
                    if n > 1:
                        factors.append(n)
                    return factors
            """),
        },
    ]
    return {"functions": functions, "description": "math utility functions"}


# ---------------------------------------------------------------------------
# Module 4: dict_utils
# ---------------------------------------------------------------------------

@_register_module("dict_utils")
def _module_dict_utils():
    functions = [
        {
            "name": "invert_dict",
            "signature": "def invert_dict(d: dict) -> dict:",
            "docstring": '    """Swap keys and values. Values must be hashable."""',
            "body": textwrap.dedent("""\
                    return {v: k for k, v in d.items()}
            """),
        },
        {
            "name": "merge_dicts",
            "signature": "def merge_dicts(a: dict, b: dict) -> dict:",
            "docstring": '    """Merge two dicts. For shared keys, values from b take priority."""',
            "body": textwrap.dedent("""\
                    result = dict(a)
                    result.update(b)
                    return result
            """),
        },
        {
            "name": "group_by",
            "signature": "def group_by(items: list, key: str) -> dict:",
            "docstring": '    """Group a list of dicts by the value of a given key."""',
            "body": textwrap.dedent("""\
                    groups = {}
                    for item in items:
                        k = item[key]
                        groups.setdefault(k, []).append(item)
                    return groups
            """),
        },
        {
            "name": "flatten_dict",
            "signature": "def flatten_dict(d: dict, prefix: str = '') -> dict:",
            "docstring": '    """Flatten a nested dict with dot-separated keys."""',
            "body": textwrap.dedent("""\
                    result = {}
                    for k, v in d.items():
                        new_key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, dict):
                            result.update(flatten_dict(v, new_key))
                        else:
                            result[new_key] = v
                    return result
            """),
        },
        {
            "name": "pick_keys",
            "signature": "def pick_keys(d: dict, keys: list) -> dict:",
            "docstring": '    """Return a new dict containing only the specified keys."""',
            "body": textwrap.dedent("""\
                    return {k: d[k] for k in keys if k in d}
            """),
        },
    ]
    return {"functions": functions, "description": "dictionary utility functions"}


# ---------------------------------------------------------------------------
# Difficulty config
# ---------------------------------------------------------------------------

_DIFFICULTY_CONFIG = {
    "easy": {"timeout_agent": 300, "timeout_verifier": 120,
             "expert_min": 3, "junior_min": 15, "removal": [1, 2]},
    "medium": {"timeout_agent": 600, "timeout_verifier": 300,
               "expert_min": 8, "junior_min": 25, "removal": [2, 3]},
    "hard": {"timeout_agent": 900, "timeout_verifier": 300,
             "expert_min": 15, "junior_min": 40, "removal": [3, 4]},
}


# ---------------------------------------------------------------------------
# Code generation helpers
# ---------------------------------------------------------------------------

def _build_full_source(module: dict) -> str:
    """Build the complete source file with all functions."""
    lines = ['"""Utility module. Implement the missing function bodies."""', ""]
    for fn in module["functions"]:
        lines.append(fn["signature"])
        lines.append(fn["docstring"])
        for body_line in fn["body"].rstrip().split("\n"):
            lines.append("    " + body_line if body_line.strip() else "")
        lines.append("")
    return "\n".join(lines)


def _build_removed_source(module: dict, removed_indices: list[int]) -> str:
    """Build source with selected function bodies replaced by NotImplementedError."""
    lines = ['"""Utility module. Implement the missing function bodies."""', ""]
    for i, fn in enumerate(module["functions"]):
        lines.append(fn["signature"])
        lines.append(fn["docstring"])
        if i in removed_indices:
            lines.append('    raise NotImplementedError("TODO: implement this function")')
        else:
            for body_line in fn["body"].rstrip().split("\n"):
                lines.append("    " + body_line if body_line.strip() else "")
        lines.append("")
    return "\n".join(lines)


def _generate_test_cases(module: dict, rng: random.Random) -> list[dict]:
    """Generate test cases for all functions based on module type."""
    module_name = None
    for name, builder in _MODULES.items():
        if builder() is module:
            module_name = name
            break

    # Use generic test generation based on function signatures
    return _generate_generic_tests(module, rng)


def _generate_generic_tests(module: dict, rng: random.Random) -> list[dict]:
    """Generate test cases by inspecting function signatures."""
    test_cases = []
    fns = module["functions"]

    for fn in fns:
        name = fn["name"]
        # Generate test based on function name patterns
        if "test_gen" in fn:
            for _ in range(3):
                test_cases.append(fn["test_gen"](rng))
        else:
            # Use the function body directly to compute expected values
            test_cases.append({"fn_name": name, "skip": True})

    return test_cases


def _build_test_code_for_module(module_name: str, module: dict, rng: random.Random) -> str:
    """Build pytest test code for the module."""
    lines = [
        '"""Auto-generated tests for code removal task."""',
        "",
        "import sys",
        "sys.path.insert(0, '/app')",
        "",
        f"from solution import *",
        "",
    ]

    fns = module["functions"]

    if module_name == "string_utils":
        _add_string_utils_tests(lines, rng)
    elif module_name == "list_utils":
        _add_list_utils_tests(lines, rng)
    elif module_name == "math_utils":
        _add_math_utils_tests(lines, rng)
    elif module_name == "dict_utils":
        _add_dict_utils_tests(lines, rng)

    return "\n".join(lines) + "\n"


def _add_string_utils_tests(lines: list[str], rng: random.Random):
    # reverse_words
    words = [rng.choice(_WORDS) for _ in range(4)]
    text = " ".join(words)
    expected = " ".join(reversed(words))
    lines.extend([
        f"def test_reverse_words():",
        f'    assert reverse_words("{text}") == "{expected}"',
        f'    assert reverse_words("hello") == "hello"',
        f'    assert reverse_words("") == ""',
        "",
    ])
    # count_vowels
    text2 = " ".join(rng.choice(_WORDS) for _ in range(3))
    expected_v = sum(1 for c in text2.lower() if c in "aeiou")
    lines.extend([
        f"def test_count_vowels():",
        f'    assert count_vowels("{text2}") == {expected_v}',
        f'    assert count_vowels("xyz") == 0',
        f'    assert count_vowels("aeiou") == 5',
        "",
    ])
    # title_case
    text3 = " ".join(rng.choice(_WORDS) for _ in range(3))
    expected_t = " ".join(w.capitalize() for w in text3.split())
    lines.extend([
        f"def test_title_case():",
        f'    assert title_case("{text3}") == "{expected_t}"',
        f'    assert title_case("hello world") == "Hello World"',
        "",
    ])
    # is_palindrome
    lines.extend([
        f"def test_is_palindrome():",
        f'    assert is_palindrome("racecar") is True',
        f'    assert is_palindrome("hello") is False',
        f'    assert is_palindrome("A man a plan a canal Panama".replace(" ", "")) is True',
        "",
    ])
    # char_frequency
    text4 = rng.choice(_WORDS)
    freq = {}
    for c in text4.lower():
        if c != " ":
            freq[c] = freq.get(c, 0) + 1
    freq = dict(sorted(freq.items()))
    lines.extend([
        f"def test_char_frequency():",
        f'    assert char_frequency("{text4}") == {freq}',
        "",
    ])


def _add_list_utils_tests(lines: list[str], rng: random.Random):
    lines.extend([
        "def test_flatten():",
        "    assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]",
        "    assert flatten([]) == []",
        "    assert flatten([[1], [2], [3]]) == [1, 2, 3]",
        "",
        "def test_unique_sorted():",
        f"    assert unique_sorted([3, 1, 2, 1, 3]) == [1, 2, 3]",
        f"    assert unique_sorted([]) == []",
        "",
        "def test_chunk():",
        f"    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
        f"    assert chunk([1, 2, 3], 5) == [[1, 2, 3]]",
        "",
        "def test_interleave():",
        f"    assert interleave([1, 2, 3], ['a', 'b', 'c']) == [1, 'a', 2, 'b', 3, 'c']",
        f"    assert interleave([1, 2], ['a']) == [1, 'a', 2]",
        "",
    ])
    nums = [rng.randint(1, 50) for _ in range(5)]
    running = []
    total = 0
    for i, n in enumerate(nums):
        total += n
        running.append(round(total / (i + 1), 4))
    lines.extend([
        "def test_running_average():",
        f"    assert running_average({nums}) == {running}",
        f"    assert running_average([10]) == [10.0]",
        "",
    ])


def _add_math_utils_tests(lines: list[str], rng: random.Random):
    a, b = rng.randint(10, 100), rng.randint(10, 100)
    import math
    g = math.gcd(a, b)
    l = a * b // g
    lines.extend([
        f"def test_gcd():",
        f"    assert gcd({a}, {b}) == {g}",
        f"    assert gcd(12, 8) == 4",
        f"    assert gcd(7, 13) == 1",
        "",
        f"def test_lcm():",
        f"    assert lcm({a}, {b}) == {l}",
        f"    assert lcm(4, 6) == 12",
        "",
        "def test_is_prime():",
        "    assert is_prime(2) is True",
        "    assert is_prime(17) is True",
        "    assert is_prime(1) is False",
        "    assert is_prime(15) is False",
        "",
        "def test_fibonacci():",
        "    assert fibonacci(1) == [0]",
        "    assert fibonacci(5) == [0, 1, 1, 2, 3]",
        "    assert fibonacci(8) == [0, 1, 1, 2, 3, 5, 8, 13]",
        "    assert fibonacci(0) == []",
        "",
    ])
    n = rng.choice([12, 18, 28, 30, 42, 56, 60, 84, 90, 100])
    factors = []
    temp = n
    d = 2
    while d * d <= temp:
        while temp % d == 0:
            factors.append(d)
            temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    lines.extend([
        "def test_prime_factors():",
        f"    assert prime_factors({n}) == {factors}",
        "    assert prime_factors(2) == [2]",
        "    assert prime_factors(12) == [2, 2, 3]",
        "",
    ])


def _add_dict_utils_tests(lines: list[str], rng: random.Random):
    lines.extend([
        "def test_invert_dict():",
        '    assert invert_dict({"a": 1, "b": 2}) == {1: "a", 2: "b"}',
        "    assert invert_dict({}) == {}",
        "",
        "def test_merge_dicts():",
        '    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}',
        '    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}',
        "",
        "def test_group_by():",
        "    items = [",
        '        {"name": "alice", "dept": "eng"},',
        '        {"name": "bob", "dept": "eng"},',
        '        {"name": "carol", "dept": "hr"},',
        "    ]",
        '    result = group_by(items, "dept")',
        '    assert len(result["eng"]) == 2',
        '    assert len(result["hr"]) == 1',
        "",
        "def test_flatten_dict():",
        '    assert flatten_dict({"a": {"b": 1, "c": {"d": 2}}}) == {"a.b": 1, "a.c.d": 2}',
        '    assert flatten_dict({"x": 1}) == {"x": 1}',
        "",
        "def test_pick_keys():",
        '    assert pick_keys({"a": 1, "b": 2, "c": 3}, ["a", "c"]) == {"a": 1, "c": 3}',
        '    assert pick_keys({"a": 1}, ["b"]) == {}',
        "",
    ])


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_instruction(module_name: str, module: dict, removed_names: list[str], difficulty: str) -> str:
    desc = module["description"]
    fn_list = "\n".join(f"- `{name}()`" for name in removed_names)

    hint = ""
    if difficulty == "easy":
        hint = "\n**Hint:** Each function is independent — you can implement them one at a time. Use the docstrings as specification.\n"

    return (
        f"You are given a Python module at `/app/solution.py` containing {desc}.\n"
        f"\n"
        f"Some function bodies have been replaced with `raise NotImplementedError`.\n"
        f"Your task is to implement the missing functions:\n"
        f"\n"
        f"{fn_list}\n"
        f"\n"
        f"Each function has a docstring describing its expected behavior.\n"
        f"The other functions in the module are already implemented and can be used as reference.\n"
        f"{hint}\n"
        f"After implementing the functions, verify they work:\n"
        f"```bash\n"
        f"python3 -c \"import solution; print('OK')\"\n"
        f"```\n"
        f"\n"
        f"The test suite will validate your implementations.\n"
    )


def _render_task_toml(difficulty: str, module_name: str) -> str:
    cfg = _DIFFICULTY_CONFIG[difficulty]
    tags = json.dumps(["code-implementation", "function-completion", module_name])
    return textwrap.dedent(f"""\
        version = "1.0"

        [metadata]
        author_name = "generated"
        author_email = "task-factory@trajectify"
        difficulty = "{difficulty}"
        category = "software-engineering"
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
class CodeRemovalFactory(TaskFactory):
    """Generates tasks where function bodies are removed and agent must implement them."""

    @staticmethod
    def factory_name() -> str:
        return "code_removal"

    def param_space(self) -> dict[str, list]:
        return {
            "module": list(_MODULES.keys()),
            "removal_count": [1, 2, 3],
            "difficulty": ["easy", "medium", "hard"],
            "seed": list(range(1, 11)),
        }

    def generate(self, params: dict, seed: int) -> GeneratedTask:
        module_name: str = params["module"]
        removal_count: int = params["removal_count"]
        difficulty: str = params["difficulty"]

        module = _MODULES[module_name]()
        rng = random.Random(seed)
        fns = module["functions"]

        # Select which functions to remove
        indices = list(range(len(fns)))
        rng.shuffle(indices)
        actual_removal = min(removal_count, len(fns) - 1)  # keep at least 1 implemented
        removed_indices = sorted(indices[:actual_removal])
        removed_names = [fns[i]["name"] for i in removed_indices]

        # Build source code with removals
        removed_source = _build_removed_source(module, removed_indices)

        # Build test code
        test_code = _build_test_code_for_module(module_name, module, rng)

        # Task name
        name = f"impl-{module_name}-{removal_count}rm-{difficulty}-s{seed}"

        # Dockerfile
        dockerfile = (
            "FROM python:3.13-slim\n"
            "\n"
            "WORKDIR /app\n"
            "\n"
            "COPY solution.py /app/solution.py\n"
        )

        return GeneratedTask(
            name=name,
            task_toml=_render_task_toml(difficulty, module_name),
            instruction=_render_instruction(module_name, module, removed_names, difficulty),
            dockerfile=dockerfile,
            test_script=_render_test_script(),
            test_files={
                "test_outputs.py": test_code,
                "../environment/solution.py": removed_source,
            },
        )
