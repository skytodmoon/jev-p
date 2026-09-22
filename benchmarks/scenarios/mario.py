"""Super Mario Bros. 1-1 关卡（示意合成快照）——"快系统"演示场景。

背景：System One 模型的典型应用是把低频**宏观动作决策**从高频确定性控制中
剥离出来（参考社区项目 typesafe-mario 的玩法：读模拟器结构化状态 → 决策
→ 输出动作）。本场景用合成快照演示决策层；真机接入模拟器（FCEUX/Libretro）
只需把每帧内存/视觉采样成同样的结构化状态（详见 README 马里奥小节）。

注意：tile 坐标为示意性 1-1 布局（16px/tile，地面行 y=12），非像素级复刻。
"""
from typing import Any, Dict, List

from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion

NAME = "mario"
DESC = "Super Mario Bros. 1-1（示意合成快照）：宏观动作决策，低延迟快系统演示"

# 动作 -> 模拟器按钮映射（真机接入时用）
ACTION_BUTTONS = {
    "run": ["RIGHT"],
    "jump": ["A"],
    "crouch": ["DOWN"],
    "wait": [],
    "turn_back": ["LEFT"],
}


def build_questions() -> Dict[str, Any]:
    """该场景的决策问题集：每个问题应是"懂行者几秒内的直觉判断"（原子化）。"""
    return {
        "action": ChoiceQuestion(
            instructions="马里奥下一步应该采取哪个宏观动作？",
            criteria={
                "run": "保持/加速向前奔跑",
                "jump": "起跳：越过敌人、坑或顶出问号块",
                "crouch": "蹲下：闪避或安全通过低矮障碍",
                "wait": "原地等待：前方暂时安全，无需动作",
                "turn_back": "回头/减速：前进风险过高",
            },
        ),
        "obstacle_ahead": NoulQuestion(
            instructions="马里奥前方 3 格内是否存在需要应对的障碍或敌人？",
            criteria={"true": "敌人/砖块/坑在 3 格内且需要动作", "false": "前方 3 格内通畅"},
        ),
        "gap_ahead": NoulQuestion(
            instructions="马里奥前方 5 格内是否存在必须跳过的坑？",
            criteria={"true": "存在无地面缺口", "false": "前方地面连续"},
        ),
        "danger": ScoreQuestion(
            instructions="当前画面整体危险程度（敌人距离/坑/速度综合）？",
            criteria=["低：通畅", "中：有障碍需处理", "高：敌人逼近或临坑"],
        ),
    }


def frames() -> List[Dict[str, Any]]:
    """4 帧示意快照：开局 → 问号块 → 蘑菇怪逼近 → 坑前。"""
    return [
        {
            "frame": 0,
            "mario": {"tile_x": 14.0, "tile_y": 12.0, "vx": 1.8, "vy": 0.0,
                      "status": "small", "on_ground": True},
            "terrain_scan": {"ground_ahead": 6, "gap_ahead": None,
                             "block_ahead": {"tile_x": 16, "type": "question_block"},
                             "stairs_ahead": None},
            "enemies": [],
            "pickups": [{"type": "coin", "tile_x": 16, "tile_y": 9}],
        },
        {
            "frame": 1,
            "mario": {"tile_x": 17.2, "tile_y": 12.0, "vx": 1.8, "vy": 0.0,
                      "status": "small", "on_ground": True},
            "terrain_scan": {"ground_ahead": 4, "gap_ahead": None,
                             "block_ahead": {"tile_x": 19, "type": "brick"},
                             "stairs_ahead": None},
            "enemies": [{"type": "goomba", "tile_x": 21.0, "tile_y": 12.0, "dx": -0.5}],
            "pickups": [],
        },
        {
            "frame": 2,
            "mario": {"tile_x": 19.1, "tile_y": 12.0, "vx": 1.8, "vy": 0.0,
                      "status": "small", "on_ground": True},
            "terrain_scan": {"ground_ahead": 3, "gap_ahead": None,
                             "block_ahead": None, "stairs_ahead": None},
            "enemies": [{"type": "goomba", "tile_x": 21.6, "tile_y": 12.0, "dx": -0.5}],
            "pickups": [],
        },
        {
            "frame": 3,
            "mario": {"tile_x": 24.0, "tile_y": 12.0, "vx": 1.8, "vy": 0.0,
                      "status": "small", "on_ground": True},
            "terrain_scan": {"ground_ahead": 1, "gap_ahead": {"start": 26, "width": 2},
                             "block_ahead": None, "stairs_ahead": None},
            "enemies": [{"type": "goomba", "tile_x": 29.0, "tile_y": 12.0, "dx": -0.5}],
            "pickups": [],
        },
    ]


def describe(frame: Dict[str, Any], idx: int) -> str:
    m = frame["mario"]
    enemies = ",".join(f"{e['type']}@{e['tile_x']:.0f}" for e in frame["enemies"]) or "无"
    gap = frame["terrain_scan"].get("gap_ahead")
    gap_txt = f"坑@{gap['start']}(宽{gap['width']})" if gap else "无坑"
    return (f"帧{idx:02d}: Mario(x={m['tile_x']:.1f}, vx={m['vx']:.1f}) | "
            f"前方地面={frame['terrain_scan']['ground_ahead']}格 | {gap_txt} | 敌人: {enemies}")


def expected(frame: Dict[str, Any]) -> Dict[str, Any]:
    """可辩解金标准（仅对主决策 action 判分）。

    帧0 顶问号块 -> jump；帧1/2 敌人 2.5~4 格外 -> run；帧3 临坑 -> jump。
    """
    exp = {"action": "jump"}
    idx = frame["frame"]
    if idx in (1, 2):
        exp["action"] = "run"
    return exp