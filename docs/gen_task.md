# Task 生成系统：完整执行流程

## 概述

Trajectify 的 Task 生成系统通过 **TaskFactory 框架** 批量生成 Terminal-Bench 2.0 格式的 task 目录。核心思想是：定义**参数空间**，遍历其**笛卡尔积**，每个参数组合调用 `generate()` 生成一个完整的 task。

整个过程**不涉及 LLM 调用**，全部是纯 Python 代码执行，因此速度极快（数千个 task 在几秒内完成）。

---

## 1. 系统架构

```
CLI (--generate)
    │
    ▼
Registry (create_factory)
    │
    ▼
TaskFactory.generate_all()
    │
    ├── param_space() → 参数空间定义
    │
    ├── itertools.product(*values) → 笛卡尔积遍历
    │
    ├── generate(params, seed) → GeneratedTask
    │   ├── 生成输入数据（确定性随机）
    │   ├── 计算期望答案（程序化）
    │   ├── 渲染模板文件
    │   └── 返回所有文件内容
    │
    └── _write_task() → 写入 TB 2.0 目录结构
```

---

## 2. 关键概念

### 2.1 GeneratedTask

一个 task 的所有文件内容，以内存数据结构表示：

```python
@dataclass
class GeneratedTask:
    name: str                  # kebab-case 目录名，如 "bugfix-number_stats-1mut-20n-easy-s1"
    task_toml: str             # task.toml 文件内容
    instruction: str           # instruction.md 文件内容
    dockerfile: str            # Dockerfile 文件内容
    test_script: str           # test.sh 文件内容
    test_files: dict[str, str] # 额外文件 {相对路径: 内容}
```

`test_files` 的 key 支持 `../environment/xxx` 格式，用于将数据文件放入 `environment/` 目录（被 Dockerfile COPY 使用）。

### 2.2 参数空间（param_space）

每个 factory 定义一个字典，key 是维度名，value 是候选值列表：

```python
{
    "log_format": ["nginx_combined", "apache_common", "json_structured"],  # 3
    "num_lines": [50, 200, 500],                                           # 3
    "difficulty": ["easy", "medium", "hard"],                              # 3
    "seed": [1, 2, 3, ..., 10],                                           # 10
}
# 笛卡尔积: 3 × 3 × 3 × 10 = 270 个 task
```

`seed` 是特殊维度——它从 `params` 中 pop 出来，作为 `generate(params, seed)` 的第二个参数传入。同一组 params + 不同 seed 产生不同的随机数据，但 task 结构相同。

### 2.3 确定性

所有随机操作必须使用 `random.Random(seed)` 实例，**不使用全局随机状态**。这保证：

- 同一 `(params, seed)` 永远生成完全相同的 task
- 不同进程、不同机器上结果一致
- 可复现、可审计

---

## 3. 执行流程详解

### 3.1 CLI 入口

```bash
uv run trajectify --generate <factory_name|all> --output-dir <dir> [--max-count N]
```

CLI 解析到 `--generate` 参数后，进入 `_run_generate()` 分支（不加载 LLM 配置）：

```python
# cli.py
async def _main() -> None:
    args = _parse_args()
    if args.generate or args.list_factories:
        _run_generate(args)    # ← 走这条路，不需要 LLM
        return
    # ... 正常的 agent 执行流程
```

`_run_generate()` 的逻辑：
1. `--list-factories` → 打印所有已注册 factory 名称并退出
2. `--generate all` → 遍历所有 factory
3. `--generate <name>` → 只运行指定 factory
4. 对每个 factory 调用 `factory.generate_all(output_dir, max_count)`

### 3.2 Registry 加载

采用与 agent registry 相同的**装饰器 + 懒加载**模式：

```python
# task_factory/__init__.py
_FACTORY_REGISTRY: dict[str, type[TaskFactory]] = {}

def _ensure_builtins_loaded():
    if _FACTORY_REGISTRY:
        return
    import trajectify.task_factory.log_analysis   # @register_factory 触发注册
    import trajectify.task_factory.bug_fix
    import trajectify.task_factory.code_removal
```

新增 factory 时需要：
1. 创建 `task_factory/<name>.py`，用 `@register_factory` 装饰类
2. 在 `_ensure_builtins_loaded()` 中添加 `import` 语句

### 3.3 generate_all() — 笛卡尔积遍历

```python
# base.py
def generate_all(self, output_dir, max_count=None):
    space = self.param_space()
    keys = list(space.keys())         # ["log_format", "num_lines", "difficulty", "seed"]
    values = [space[k] for k in keys] # [["nginx_combined", ...], [50, 200, 500], ...]

    for i, combo in enumerate(itertools.product(*values)):
        if max_count is not None and i >= max_count:
            break
        params = dict(zip(keys, combo))  # {"log_format": "nginx_combined", "num_lines": 50, ...}
        seed = params.pop("seed", 0)     # seed 单独取出
        task = self.generate(params, seed)
        self._write_task(task, output_dir)
```

遍历顺序：按 `param_space()` 字典的 key 顺序做笛卡尔积。`max_count` 截断前 N 个组合（用于测试时快速生成少量 task）。

### 3.4 generate() — 单个 task 生成

以 `BugFixFactory.generate()` 为例，展示典型的生成流程：

```
输入: params={"scenario": "number_stats", "mutation_count": 1, "num_items": 20, "difficulty": "easy"}, seed=1
                                │
                                ▼
Step 1: 加载 scenario 定义
        scenario = _SCENARIOS["number_stats"]()
        → 返回 correct_source, mutations, generate_input, compute_expected
                                │
                                ▼
Step 2: 生成确定性输入数据
        rng = random.Random(seed=1)
        input_data = scenario["generate_input"](rng, num_items=20)
        → "42.5\n-17.3\n88.1\n..."  (20行随机数字)
                                │
                                ▼
Step 3: 计算期望答案
        expected = scenario["compute_expected"](input_data)
        → {"count": 20, "sum": 234.5, "mean": 11.725, "min": -42.1, ...}
                                │
                                ▼
Step 4: 注入 bug
        buggy_source = _apply_mutations(
            correct_source,
            mutations,             # 候选 bug 列表
            difficulty="easy",     # 过滤适合 easy 的 mutation
            mutation_count=1,      # 注入 1 个 bug
            rng=Random(seed+1000), # 独立的随机源选择 bug
        )
        → 把 "sorted_nums[count // 2]" 替换成 "sorted_nums[count // 2 + 1]"
                                │
                                ▼
Step 5: 渲染所有文件
        → GeneratedTask(
            name="bugfix-number_stats-1mut-20n-easy-s1",
            task_toml=...,      # 元数据 (difficulty, timeout, tags)
            instruction=...,    # "给你一个有 bug 的脚本，请修复..."
            dockerfile=...,     # FROM python:3.13-slim + COPY 数据
            test_script=...,    # uvx pytest runner
            test_files={
                "test_outputs.py": ...,          # 嵌入 expected 的 pytest 断言
                "../environment/input_data": ..., # 输入数据文件
                "../environment/solution.py": ..., # 带 bug 的源码
            },
        )
```

### 3.5 _write_task() — 写入磁盘

将 `GeneratedTask` 写入标准 TB 2.0 目录结构：

```
output_dir/
└── bugfix-number_stats-1mut-20n-easy-s1/
    ├── task.toml                    ← task.task_toml
    ├── instruction.md               ← task.instruction
    ├── environment/
    │   ├── Dockerfile               ← task.dockerfile
    │   ├── input_data               ← test_files["../environment/input_data"]
    │   └── solution.py              ← test_files["../environment/solution.py"]
    └── tests/
        ├── test.sh                  ← task.test_script
        └── test_outputs.py          ← test_files["test_outputs.py"]
```

`test_files` 中以 `../environment/` 开头的 key 会被写入 `environment/` 目录而非 `tests/` 目录。

---

## 4. 现有 Factory 详解

### 4.1 log_analysis（810 变体）

**任务类型**：解析日志文件 → 输出 JSON 统计报告

| 维度 | 候选值 | 说明 |
|------|--------|------|
| `log_format` | nginx_combined, apache_common, json_structured | 日志格式 |
| `num_lines` | 50, 200, 500 | 日志行数 |
| `analysis_group` | group_a, group_b, group_c | 需要统计的字段组合 |
| `difficulty` | easy, medium, hard | 影响 timeout 和 instruction 提示 |
| `seed` | 1-10 | 控制日志内容（IP、路径、状态码等） |

**生成流程**：
1. `_generate_nginx_combined(rng, num_lines)` → 生成 nginx 格式日志行
2. `_parse_log_entries()` → 解析为统一的 entry 列表
3. `_compute_expected(entries, fields)` → 程序化计算 total_requests, unique_ips, status_codes 等
4. 期望答案嵌入 `test_outputs.py` 的 `EXPECTED` 常量

### 4.2 bug_fix（1,350 变体）

**任务类型**：修复有 bug 的 Python 脚本

| 维度 | 候选值 | 说明 |
|------|--------|------|
| `scenario` | number_stats, word_counter, csv_aggregator, json_transformer, matrix_ops | 5 种基础程序 |
| `mutation_count` | 1, 2, 3 | 注入的 bug 数量 |
| `num_items` | 20, 50, 100 | 输入数据规模 |
| `difficulty` | easy, medium, hard | 过滤可选的 mutation 类型 |
| `seed` | 1-10 | 控制输入数据 + bug 选择位置 |

**生成流程**：
1. 正确源码是手写的模板（每个 scenario 约 30 行）
2. `generate_input(rng, num_items)` → 生成随机输入数据
3. `compute_expected(input_data)` → 用正确逻辑计算期望输出
4. `_apply_mutations()` → 从 mutation 列表中选择并注入 bug
5. 期望答案嵌入测试，buggy 代码放入容器

**Mutation 类型**：
| 类型 | 示例 | 难度 |
|------|------|------|
| wrong_operator | `/ → //`, `>= → >` | easy |
| off_by_one | `count // 2` → `count // 2 + 1` | easy |
| missing_guard | 删除 `if not numbers: return ...` | medium |
| wrong_function | `sum(numbers)` → `len(numbers)` | medium |
| wrong_cast | `float(x)` → `int(x)` | medium |

### 4.3 code_removal（360 变体）

**任务类型**：实现被删除的函数体

| 维度 | 候选值 | 说明 |
|------|--------|------|
| `module` | string_utils, list_utils, math_utils, dict_utils | 4 种函数模块 |
| `removal_count` | 1, 2, 3 | 删除的函数数量 |
| `difficulty` | easy, medium, hard | 影响 timeout 和 instruction 提示 |
| `seed` | 1-10 | 控制删除哪些函数 + 测试数据 |

**生成流程**：
1. 每个 module 包含 5 个独立函数（手写，含 signature + docstring + body）
2. 根据 seed 随机选择 removal_count 个函数
3. 被选中的函数体替换为 `raise NotImplementedError("TODO: implement this function")`
4. 生成针对所有函数的 pytest 测试（含具体的输入输出用例）

---

## 5. 生成的 Task 如何被执行

生成的 task 目录是标准的 TB 2.0 格式，可以直接被现有 pipeline 加载和执行，**无需任何改动**：

```bash
# 生成
uv run trajectify --generate bug_fix --output-dir tasks/generated/bug-fix --max-count 5

# 执行（与手工编写的 task 完全相同的方式）
uv run trajectify --task tasks/generated/bug-fix/bugfix-number_stats-1mut-20n-easy-s1 --agent terminus
```

执行流程：
```
CLI --task
    │
    ▼
TerminalBenchLoader.load()
    → 读取 task.toml + instruction.md
    → 返回 Task 对象
    │
    ▼
Runner
    → DockerEnvironment.start()
        → docker build (用 environment/Dockerfile + 同目录的数据文件)
        → docker compose up
    → TerminusAgent.run()
        → 把 instruction.md 内容发给 LLM
        → LLM 在 tmux 中执行命令（读代码、修 bug、运行脚本等）
        → 循环直到 agent 报告完成或超时
    → Verifier
        → 上传 tests/ 到容器
        → 执行 test.sh
        → 读取 /logs/verifier/reward.txt (0 或 1)
    → Exporter
        → 导出 trajectory.json (ATIF/SFT/Rollout 格式)
```

---

## 6. 验证机制

所有 factory 生成的 test 都遵循相同模式：

### 6.1 test.sh — 测试执行器

```bash
#!/bin/bash
# 安装 uv + pytest
uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 \
    pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

# 根据 pytest 退出码写入 reward
if [ $? -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
```

### 6.2 test_outputs.py — 断言逻辑

**log_analysis / bug_fix**：期望答案在生成时计算好，嵌入为 `EXPECTED` 常量：

```python
EXPECTED = {"total_requests": 50, "unique_ips": 44, ...}

def test_report_values():
    report = json.loads(Path("/app/report.json").read_text())
    for key in EXPECTED:
        assert report[key] == EXPECTED[key]
```

**code_removal**：直接对函数做单元测试：

```python
from solution import *

def test_reverse_words():
    assert reverse_words("hello world") == "world hello"

def test_is_prime():
    assert is_prime(17) is True
    assert is_prime(15) is False
```

### 6.3 浮点精度处理

`BugFixFactory` 的测试使用 `_approx_equal()` 递归比较，容忍 1e-2 的浮点误差：

```python
def _approx_equal(a, b, tol=1e-2):
    if isinstance(a, float) and isinstance(b, (int, float)):
        return abs(a - b) < tol
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_approx_equal(a[k], b[k], tol) for k in a)
    ...
```

---

## 7. 如何新增一个 Factory

### 7.1 创建文件

创建 `trajectify/task_factory/<name>.py`：

```python
from trajectify.task_factory import register_factory
from trajectify.task_factory.base import GeneratedTask, TaskFactory

@register_factory
class MyFactory(TaskFactory):
    @staticmethod
    def factory_name() -> str:
        return "my_factory"

    def param_space(self) -> dict[str, list]:
        return {
            "variant": ["a", "b", "c"],
            "difficulty": ["easy", "medium", "hard"],
            "seed": list(range(1, 11)),
        }

    def generate(self, params: dict, seed: int) -> GeneratedTask:
        rng = random.Random(seed)
        # 1. 生成输入数据
        # 2. 计算期望答案
        # 3. 渲染所有文件
        return GeneratedTask(name=..., task_toml=..., instruction=..., ...)
```

### 7.2 注册

在 `task_factory/__init__.py` 的 `_ensure_builtins_loaded()` 中添加：

```python
import trajectify.task_factory.<name>  # noqa: F401
```

### 7.3 设计检查清单

- [ ] `factory_name()` 返回唯一的 snake_case 标识
- [ ] `param_space()` 包含 `seed` 维度
- [ ] `generate()` 中所有随机操作使用 `random.Random(seed)`，不用全局 random
- [ ] 期望答案由 Python 程序化计算，不依赖外部服务
- [ ] instruction 不泄露测试逻辑或期望答案
- [ ] Dockerfile 使用 `python:3.13-slim` 基础镜像（除非任务需要其他运行时）
- [ ] test.sh 使用标准的 uvx + pytest 模式
- [ ] task name 格式：`<prefix>-<param1>-<param2>-...-s<seed>`

---

## 8. 扩展到大规模

### 8.1 当前规模

| Factory | 参数组合 | 变体数 |
|---------|---------|--------|
| log_analysis | 3×3×3×3×10 | 810 |
| bug_fix | 5×3×3×3×10 | 1,350 |
| code_removal | 4×3×3×10 | 360 |
| **合计** | | **2,520** |

### 8.2 扩展杠杆

**杠杆 1：扩大 seed 范围**

```python
"seed": list(range(1, 101))  # 10 → 100
```
所有 factory 变体数 ×10。当前 2,520 → **25,200**。

**杠杆 2：增加参数维度**

以 `log_analysis` 为例，可新增：
- `error_ratio`: [0.01, 0.1, 0.3, 0.5]（错误日志比例）
- `ip_distribution`: ["uniform", "zipf", "single_dominant"]（IP 分布模式）
- `output_format`: ["json", "csv", "yaml"]（输出格式要求）

**杠杆 3：增加 scenario/module 模板数量**

当前 BugFixFactory 有 5 个 scenario（手写）。可以：
- 继续手写更多 scenario
- 用 LLM 批量生成 scenario 模板，人工审核后加入
- 从开源代码库抽取小型脚本作为 scenario

**杠杆 4：新增 Factory**

适合 factory 化的域（数据可参数化 + 答案可程序化计算）：

| 候选 Factory | 预估变体数 |
|-------------|-----------|
| etl_pipeline（CSV/JSON 转换） | ~1,800 |
| database_queries（SQL 查询） | ~2,160 |
| regex_text（正则提取/替换） | ~1,440 |
| algorithm_impl（经典算法） | ~2,700 |
| file_conversion（格式互转） | ~1,620 |
| git_operations（分支/冲突） | ~1,350 |

### 8.3 到 10 万条的路径

```
  现有 3 个 factory × seed 扩到 100           = ~25,000
+ 新增 6 个 factory (各 ~2,000) × seed 100   = ~120,000
+ 参数维度细化                                 = 额外增量
──────────────────────────────────────────────
  总计 > 100,000
```

---

## 9. 完整命令参考

```bash
# 列出所有已注册的 factory
uv run trajectify --list-factories

# 从指定 factory 生成全部变体
uv run trajectify --generate log_analysis --output-dir tasks/generated/log-analysis

# 限制生成数量（用于测试）
uv run trajectify --generate bug_fix --output-dir tasks/generated/bug-fix --max-count 10

# 从所有 factory 生成
uv run trajectify --generate all --output-dir tasks/generated

# 执行生成的 task（与手工 task 完全相同）
uv run trajectify --task tasks/generated/bug-fix/bugfix-number_stats-1mut-20n-easy-s1 --agent terminus

# 批量执行（通过 YAML 配置）
uv run trajectify config.yaml
```

---

## 10. 目录结构

```
trajectify/
├── task_factory/
│   ├── __init__.py          # registry + 懒加载
│   ├── base.py              # TaskFactory ABC + GeneratedTask dataclass
│   ├── log_analysis.py      # 日志分析 factory (810 variants)
│   ├── bug_fix.py           # Bug 修复 factory (1,350 variants)
│   └── code_removal.py      # 函数实现 factory (360 variants)
├── cli.py                   # --generate / --list-factories 入口
└── ...

tasks/generated/             # 生成的 task 输出目录
├── log-analysis/
│   ├── log-nginx-combined-50L-group_a-easy-s1/
│   │   ├── task.toml
│   │   ├── instruction.md
│   │   ├── environment/
│   │   │   ├── Dockerfile
│   │   │   └── access.log
│   │   └── tests/
│   │       ├── test.sh
│   │       └── test_outputs.py
│   ├── log-nginx-combined-50L-group_a-easy-s2/
│   └── ...  (810 个目录)
├── bug-fix/
│   └── ...  (1,350 个目录)
└── code-removal/
    └── ...  (360 个目录)
```
