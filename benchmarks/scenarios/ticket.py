"""工单分级/路由场景（原始基准）：客服工单三问 + 可选多部门 choice 扩展。"""
from typing import Any, Dict, List

from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion

NAME = "ticket"
DESC = "客服工单：路由/紧急度/情绪（可扩展字段数以测吞吐）"

STATE = (
    "客服工单：客户报告 Stripe 支付回调整整 3 天没有成功，"
    "订单 499 美元已支付，要求尽快处理并退款。"
)

CHOICE_CRITERIA = {
    "billing": "账单、退款、发票相关",
    "technical": "产品故障、报错、无法使用",
    "account": "账号与登录",
    "sales": "售前咨询、报价",
    "other": "其他",
}


def build_questions(n_choice: int = 1) -> Dict[str, Any]:
    """n_choice 个 choice + 1 个 noul + 1 个 score ≈ n_choice+2 个字段。"""
    qs: Dict[str, Any] = {}
    for i in range(n_choice):
        qs[f"dept_{i}"] = ChoiceQuestion(
            instructions=f"工单 {i}：应该路由到哪个部门？",
            criteria=CHOICE_CRITERIA,
        )
    qs["urgent"] = NoulQuestion(instructions="这条工单是否紧急、需要立即干预？")
    qs["frustration"] = ScoreQuestion(
        instructions="客户的不满程度如何？",
        criteria=["平静", "不满", "非常愤怒"],
    )
    return qs


def frames(n_choice: int = 1) -> List[Any]:
    return [STATE]  # 单状态；多帧在需要时扩展


def describe(frame: Any, idx: int) -> str:
    return f"工单快照 {idx}"