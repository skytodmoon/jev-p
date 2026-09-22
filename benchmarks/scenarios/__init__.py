"""基准测试/演示场景注册表。

每个场景模块暴露统一接口：
- NAME / DESC：名称与说明；
- build_questions() -> Dict[str, Question]：该场景的决策问题集（原子化判断）；
- frames() -> List[Any]：结构化状态列表（每项是一帧/一个快照）；
- describe(frame, idx) -> str：打印用的人类可读摘要。

用法：
    python benchmarks/benchmark.py --scenario mario
    python benchmarks/benchmark.py --scenario ticket --fields 4,16
"""
from . import guardrail, mario, ticket

SCENARIOS = {
    "mario": mario,
    "guardrail": guardrail,
    "ticket": ticket,
}


def load(name: str):
    try:
        return SCENARIOS[name]
    except KeyError:
        raise SystemExit(f"未知场景: {name!r}，可选: {', '.join(SCENARIOS)}")