"""国际（英文）企业-产品匹配场景：海外企业候选，与中文场景同一套 6 问与 0-3 金标准。

配合 `enterprise_match`（中文）构成**中英双语评测集**：
- 同一产品（Titan-PHM 预测性维护平台），同一评分问题结构；
- state/criteria 全英文（海外企业数据）；金标准 match_level 0-3 同口径；
- 用途：双语评测（`benchmarks/eval_bilingual_enterprise.py`）与未来中英兼顾模型的
  持续评测基准。

示例产品（英文版描述）：
    Titan-PHM Reliability SaaS — vibration/temp/current sensing, AI failure prediction,
    maintenance-work-order loop for discrete & process manufacturing.
"""
from typing import Any, Dict, List

from jevp.protocol import ChoiceQuestion, NoulQuestion, ScoreQuestion

NAME = "enterprise_match_en"
DESC = "国际（英文）企业-产品匹配：海外企业候选，双语评测集英文侧"

PRODUCT_EN = {
    "name": "Titan-PHM Predictive Maintenance Platform",
    "description": ("Reliability SaaS for discrete & process manufacturing: sensing, "
                    "AI failure prediction, maintenance-work-order loop. Target: companies "
                    "with critical production equipment and high downtime/asset-lifecycle cost."),
}


def build_questions() -> Dict[str, Any]:
    return {
        "industry_fit": NoulQuestion(
            instructions=("Is the company's primary industry aligned with or adjacent to our "
                          "target (manufacturing-heavy, or asset-heavy energy & utilities)?"),
            criteria={"true": "Manufacturing or asset-heavy industry (steel/chemical/energy/auto parts etc.)",
                      "false": "Unrelated (pure e-commerce/finance/light retail)"},
        ),
        "demand_signal": NoulQuestion(
            instructions=("Any demand signal in the last 12 months (bids / procurement / news) "
                          "for equipment-health, predictive-maintenance, or smart O&M?"),
            criteria={"true": "A bid / procurement / digitalization signal exists",
                      "false": "No visible demand signal"},
        ),
        "budget_scale": ScoreQuestion(
            instructions="Does the company's size support procuring and deploying industrial software?",
            criteria=["Small: unlikely budget", "Medium: mid-size budget", "Large: strong budget"],
        ),
        "strategic_fit": ScoreQuestion(
            instructions=("How well does the company's stated strategy (digitalization / smart "
                          "factory / asset lifecycle) match our value proposition?"),
            criteria=["Low: traditional, no digital strategy", "Medium: some digital, not core",
                      "High: smart factory / equipment management is strategic"],
        ),
        "urgency": NoulQuestion(
            instructions="Is there a near-term procurement window (active bid, new plant, stated pain)?",
            criteria={"true": "Clear near-term procurement window", "false": "No near-term window"},
        ),
        "match_score": ScoreQuestion(
            instructions="Overall: how strong is this company as a prospective customer?",
            criteria=["0 Irrelevant", "1 Low", "2 Medium", "3 High"],
        ),
    }


def frames() -> List[Dict[str, Any]]:
    """6 家海外合成企业（英文档案，覆盖 0-3 档）。"""
    return [
        {
            "name": "VulcanSteel Corp.",
            "basic": {"industry": "Integrated steelmaking", "scale": "Large / listed",
                      "location": "Pittsburgh, USA", "employees": "32k", "since": 1901},
            "bidding": [{"year": 2026, "title": "Blast-furnace line condition-monitoring retrofit", "amount": 1200},
                        {"year": 2025, "title": "Rolling-mill asset-health analytics platform"}],
            "website": {"strategy": "'reliability engineering', 'digital twin of the mill'",
                        "news": "Opened new EAF line Q3"},
            "supply_chain": {"customers": "Auto/construction OEMs", "suppliers": "Mill equipment vendors",
                             "leverage": "Vendor ecosystem can be a channel"},
        },
        {
            "name": "RheinChem",
            "basic": {"industry": "Chemicals", "scale": "Large / blue-chip", "location": "Leverkusen, DE",
                      "employees": "21k", "since": 1863},
            "bidding": [{"year": 2025, "title": "Predictive maintenance on rotating equipment fleet"},
                        {"year": 2024, "title": "Asset integrity management tooling"}],
            "website": {"strategy": "'Industry 4.0', 'asset integrity', 'net-zero operations'",
                        "news": "Annual safety & reliability capex > EUR 400M"},
            "supply_chain": {"customers": "Specialty chemical downstream", "suppliers": "Pump/compressor OEMs",
                             "leverage": "Safety-driven procurement appetite"},
        },
        {
            "name": "BrightGrid Energy",
            "basic": {"industry": "Power generation / utility", "scale": "Large", "location": "Austin, USA",
                      "employees": "9k", "since": 1998},
            "bidding": [{"year": 2026, "title": "Turbine health monitoring across 3 plants"},
                        {"year": 2025, "title": "O&M digitalization of gas fleet"}],
            "website": {"strategy": "'grid reliability', 'O&M cost reduction', 'asset analytics'",
                        "news": "Retiring 2 coal units, adding CCGT"},
            "supply_chain": {"customers": "Regional grid / utilities", "suppliers": "Turbine OEMs",
                             "leverage": "Asset-intensive, data-ready"},
        },
        {
            "name": "NordaAuto",
            "basic": {"industry": "Automotive parts", "scale": "Mid-size", "location": "Munich, DE",
                      "employees": "3.2k", "since": 1994},
            "bidding": [{"year": 2026, "title": "CNC tool-life prediction pilot"}, ],
            "website": {"strategy": "'lean manufacturing'; limited digital mention",
                        "news": "New OEM framework contract"},
            "supply_chain": {"customers": "German premium OEMs", "suppliers": "CNC machine vendors",
                             "leverage": "Small pilot budget, reference value"},
        },
        {
            "name": "PacificFresh Foods",
            "basic": {"industry": "Food processing", "scale": "Mid-size", "location": "Auckland, NZ",
                      "employees": "1.4k", "since": 2008},
            "bidding": [{"year": 2026, "title": "Packaging-line automation (focussed on wrapping)"}],
            "website": {"strategy": "'quality & traceability'; no equipment-health focus",
                        "news": "None"},
            "supply_chain": {"customers": "Retail chains", "suppliers": "Filling/sterilisation vendors",
                             "leverage": "Low equipment complexity"},
        },
        {
            "name": "ShopNova",
            "basic": {"industry": "E-commerce", "scale": "Mid-size", "location": "London, UK",
                      "employees": "800", "since": 2016},
            "bidding": [],
            "website": {"strategy": "'omnichannel retail', 'data-driven growth'",
                        "news": "None"},
            "supply_chain": {"customers": "Consumers", "suppliers": "Logistics/robotics vendors",
                             "leverage": "No production equipment; industry irrelevant"},
        },
    ]


def expected(frame: Dict[str, Any]) -> Dict[str, Any]:
    name = frame["name"]
    level = {
        "VulcanSteel Corp.": 3,
        "RheinChem": 3,
        "BrightGrid Energy": 3,
        "NordaAuto": 2,
        "PacificFresh Foods": 1,
        "ShopNova": 0,
    }[name]
    return {"match_score": level}


def describe(frame: Dict[str, Any], idx: int) -> str:
    return f"{frame['name']}（{frame['basic']['industry']}，{frame['basic']['scale']}）"