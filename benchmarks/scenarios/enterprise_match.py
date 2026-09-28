"""业务场景：企业-产品匹配评价（潜在客户挖掘 → 打分排序）。

定位：System One 体系的批量精排应用。recall 阶段（企业库/招投标库检索候选）
由外部系统完成，本场景对每个候选企业做 6 个原子化判断，产出 match_score 排序。

- 产品输入：任意产品/方案（示例为"工业装备预测性维护平台 PHM"）；
- 企业档案（state）：基本信息 + 招投标 + 官网战略关键词 + 上下游关系；
- 金标准：match_level = 0(无关)/1(低)/2(中)/3(高/极高)，供离线评测判分。

中文 criteria 面向 LLM 后端（qwen3.8-27b 等中英均衡），判别式英文后端会弱，
README §3.12 有说明（粗排用 27B；英文场景可切 Laya/AgentJev）。
"""
from typing import Any, Dict, List

from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion

NAME = "enterprise_match"
DESC = "企业-产品匹配评价：潜在客户挖掘（6 问打分 → 排序）"

PRODUCT = {
    "name": "Titan-PHM 工业装备预测性维护平台",
    "description": ("面向离散/流程制造业的设备健康管理 SaaS：振动/温度/电流传感接入、"
                    "AI 故障预测、维修工单联动；目标客户为拥有核心生产设备且重视"
                    "停机损失与设备全生命周期成本的企业。"),
}


def build_questions() -> Dict[str, Any]:
    return {
        "industry_fit": NoulQuestion(
            instructions="该企业的主营行业是否与我们产品的目标行业（制造业主）、或相邻重资产行业一致？",
            criteria={"true": "行业匹配或相邻（制造业/流程/能源等重资产）",
                      "false": "行业无关（纯互联网/金融/零售轻资产等）"},
        ),
        "demand_signal": NoulQuestion(
            instructions="企业近期（12 个月内招投标/采购公告/新闻）是否出现该类产品（设备健康/预测性维护/智能运维/数字化转型）的需求信号？",
            criteria={"true": "有招投标/采购/转型信号", "false": "无可见需求信号"},
        ),
        "budget_scale": ScoreQuestion(
            instructions="企业的体量与采购能力是否支撑该类工业软件的采购与落地？",
            criteria=["小：微型/个体，无力采购", "中：中小规模，预算有限", "大：中大企业，预算充足"],
        ),
        "strategic_fit": ScoreQuestion(
            instructions="企业官网/年报的战略方向（数字化转型/智能制造/设备全生命周期管理）与我们的价值主张契合度？",
            criteria=["低：传统经营，无数字化战略", "中：有数字化表述但非重点", "高：智能制造/设备管理是战略重点"],
        ),
        "urgency": NoulQuestion(
            instructions="是否存在近期可触发的采购窗口（正在招标、产线刚建成、有明确痛点表述）？",
            criteria={"true": "近期有明确采购窗口", "false": "无近期窗口"},
        ),
        "match_score": ScoreQuestion(
            instructions="综合评估：该公司作为我们潜在客户的匹配度等级？",
            criteria=["0 无关", "1 低匹配", "2 中匹配", "3 高匹配"],
        ),
    }


def frames() -> List[Dict[str, Any]]:
    """8 家合成候选企业档案（覆盖 4 档），字段：基本信息/招投标/官网/上下游。"""
    return [
        {
            "name": "华银钢铁集团",
            "basic": {"industry": "钢铁冶炼", "scale": "大型/上市", "location": "河北",
                      "employees": "2.4万", "since": 1992},
            "bidding": [{"year": 2026, "title": "轧钢产线设备状态监测系统招标", "amount": 860},
                        {"year": 2025, "title": "全厂设备全生命周期管理平台建设"}],
            "website": {"strategy": "『数字化转型标杆工厂』『设备全生命周期成本最优』",
                        "news": "新建 5G+工业互联网平台"},
            "supply_chain": {"customers": "汽车/家电制造商", "suppliers": "冶金设备/传感器厂商",
                             "leverage": "设备供应商生态可作渠道"},
        },
        {
            "name": "新能电池科技",
            "basic": {"industry": "新能源锂电", "scale": "大型/新投产", "location": "江苏",
                      "employees": "1.1万", "since": 2021},
            "bidding": [{"year": 2026, "title": "智能工厂产线设备联网与预测性维护采购"},
                        {"year": 2025, "title": "新建 3 期产线整线自动化集成"}],
            "website": {"strategy": "『灯塔工厂』『全流程智能制造』『零非计划停机』",
                        "news": "3 期基地投产在即，产线设备集中到货"},
            "supply_chain": {"customers": "头部车企/储能集成商", "suppliers": "卷绕机/涂布机厂商",
                             "leverage": "设备新投+产线爬坡，现场痛点密集"},
        },
        {
            "name": "环球化工集团",
            "basic": {"industry": "基础化工", "scale": "大型/央企背景", "location": "山东",
                      "employees": "1.8万", "since": 1987},
            "bidding": [{"year": 2025, "title": "化工装置预测性维护与安全联锁升级"},
                        {"year": 2024, "title": "重点设备健康监测平台一期"}],
            "website": {"strategy": "『本质安全』『智能工厂』『降本增效』",
                        "news": "安全环保技改年度投入超 10 亿"},
            "supply_chain": {"customers": "下游精细化工/新材料园区", "suppliers": "泵阀/压缩机厂商",
                             "leverage": "安全驱动采购意愿强"},
        },
        {
            "name": "锐驰汽车零部件",
            "basic": {"industry": "汽车零部件", "scale": "中型", "location": "浙江",
                      "employees": 3800, "since": 2005},
            "bidding": [{"year": 2026, "title": "机加工中心设备刀具寿命预测试点"}],
            "website": {"strategy": "『精益制造』，数字化提及有限",
                        "news": "获主机厂定点，订单增长 40%"},
            "supply_chain": {"customers": "比亚迪/吉利体系", "suppliers": "数控机床厂商",
                             "leverage": "试点预算小，但行业示范价值"},
        },
        {
            "name": "绿源建材股份",
            "basic": {"industry": "建材水泥", "scale": "中型", "location": "河南",
                      "employees": 2200, "since": 1998},
            "bidding": [{"year": 2025, "title": "水泥窑协同处置技改（侧重环保）"}],
            "website": {"strategy": "传统经营，官网无数字化战略表述",
                        "news": "无"},
            "supply_chain": {"customers": "地产/基建", "suppliers": "破碎机/磨机厂商",
                             "leverage": "有大体量设备但预算与意愿存疑"},
        },
        {
            "name": "味全食品工业",
            "basic": {"industry": "食品加工", "scale": "中型", "location": "广东",
                      "employees": 1500, "since": 2010},
            "bidding": [{"year": 2026, "title": "产线自动化改造（以包装为主）"}],
            "website": {"strategy": "『品质管控』，无设备管理数字化重点",
                        "news": "无"},
            "supply_chain": {"customers": "商超/餐饮连锁", "suppliers": "灌装/杀菌设备厂商",
                             "leverage": "设备复杂度低，匹配度弱"},
        },
        {
            "name": "云购电商科技",
            "basic": {"industry": "电商零售", "scale": "中型", "location": "上海",
                      "employees": 900, "since": 2015},
            "bidding": [],
            "website": {"strategy": "『全渠道零售』『数据驱动增长』",
                        "news": "无"},
            "supply_chain": {"customers": "C 端消费者", "suppliers": "仓储机器人/物流服务商",
                             "leverage": "无生产设备，行业无关"},
        },
        {
            "name": "北方工业大学装备研究院",
            "basic": {"industry": "高校/科研院所", "scale": "非企业", "location": "辽宁",
                      "employees": 600, "since": 1952},
            "bidding": [{"year": 2025, "title": "实验台架状态监测科研采购（单台小额度）"}],
            "website": {"strategy": "科研方向：故障诊断与 PHM 算法研究",
                        "news": "招生与课题为主"},
            "supply_chain": {"customers": "科研合作企业", "suppliers": "试验台厂商",
                             "leverage": "可作产学研合作而非商业客户"},
        },
    ]


def expected(frame: Dict[str, Any]) -> Dict[str, Any]:
    """规则金标准（可辩解）：综合档位 0-3。"""
    name = frame["name"]
    level = {
        "华银钢铁集团": 3, "新能电池科技": 3, "环球化工集团": 3,
        "锐驰汽车零部件": 2, "绿源建材股份": 2, "味全食品工业": 1,
        "云购电商科技": 0, "北方工业大学装备研究院": 0,
    }[name]
    return {"match_score": level}


def describe(frame: Dict[str, Any], idx: int) -> str:
    return (f"{frame['name']}（{frame['basic']['industry']}，{frame['basic']['scale']}）")