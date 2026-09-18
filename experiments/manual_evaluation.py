"""Step 12.3 human evaluation framework for A/B marketing strategies."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for module_path in (PROJECT_ROOT, APP_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import product_analyzer  # noqa: E402
from evaluation_test import (  # noqa: E402
    PRODUCT,
    STRATEGY_FIELDS,
    TOP_K,
    generate_without_rag,
)
from marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    generate_marketing_strategy,
)
from marketing_strategy_workflow import (  # noqa: E402
    build_retrieval_query,
    retrieve_knowledge,
)
from product_analyzer import ProductAnalysis  # noqa: E402


MANUAL_CRITERIA: tuple[dict[str, str], ...] = (
    {
        "key": "product_relevance",
        "name": "商品相关性",
        "description": "策略是否准确围绕当前商品及已确认卖点展开。",
    },
    {
        "key": "audience_fit",
        "name": "目标人群匹配度",
        "description": "目标人群是否与当前商品和使用场景合理匹配。",
    },
    {
        "key": "scenario_fit",
        "name": "场景匹配度",
        "description": "营销策略是否与已确认的登山、徒步、户外运动场景匹配。",
    },
    {
        "key": "user_needs_reasonableness",
        "name": "用户需求合理性",
        "description": "提出的用户需求是否能够从当前商品特点和场景合理推导。",
    },
    {
        "key": "marketing_direction_reasonableness",
        "name": "营销方向合理性",
        "description": "营销方向是否清晰，是否能够连接商品、用户和场景。",
    },
    {
        "key": "channel_executability",
        "name": "渠道可执行性",
        "description": "渠道选择是否能够支持当前商品的营销目标，并具有实际执行价值。",
    },
    {
        "key": "content_executability",
        "name": "内容可执行性",
        "description": "内容方向是否可以进一步转化为具体营销内容。",
    },
    {
        "key": "risk_control_completeness",
        "name": "风险控制完整性",
        "description": "是否充分避免虚构属性、参数、认证、效果，以及从场景反推更强能力等问题。",
    },
)

SCORE_GUIDE = (
    "5分：非常符合，基本无需修改。",
    "4分：较符合，有少量修改空间。",
    "3分：基本可用，但存在明显优化空间。",
    "2分：问题较明显，需要较大修改。",
    "1分：明显不符合，基本不可用。",
)


def evaluate_strategy_manually(
    strategy: MarketingStrategy,
    responses: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create or validate a human-filled evaluation record without auto-scoring."""
    if not isinstance(strategy, MarketingStrategy):
        raise TypeError("人工评价对象必须是 MarketingStrategy")

    supplied = responses or {}
    evaluations: dict[str, dict[str, Any]] = {}
    for criterion in MANUAL_CRITERIA:
        response = supplied.get(criterion["key"], {})
        score = response.get("score")
        reason = response.get("reason", "")
        if score is not None and (not isinstance(score, int) or not 1 <= score <= 5):
            raise ValueError(f"{criterion['name']}的评分必须是 1～5 或 None")
        if not isinstance(reason, str):
            raise TypeError(f"{criterion['name']}的评价理由必须是字符串")
        evaluations[criterion["key"]] = {
            "score": score,
            "reason": reason,
        }

    return {
        "strategy": strategy.model_dump(),
        "evaluations": evaluations,
    }


def _print_product() -> None:
    print("【测试商品】")
    print(f"品牌：{PRODUCT['brand_name']}")
    print(f"商品：{PRODUCT['product_name']}")
    print(f"商品类型：{PRODUCT['product_type']}")
    print(f"已确认卖点：{PRODUCT['confirmed_features']}")
    print(f"使用场景：{PRODUCT['usage_scenarios']}")


def _print_strategy(strategy: MarketingStrategy) -> None:
    data = strategy.model_dump()
    for field in STRATEGY_FIELDS:
        print(f"{field}：{data.get(field)}")


def _print_criteria() -> None:
    print("\n========== 人工评价维度 ==========")
    for index, criterion in enumerate(MANUAL_CRITERIA, 1):
        print(f"{index}. {criterion['name']}")
        print(f"   评分：1-5")
        print(f"   评价说明：{criterion['description']}")
    print("\n统一评分标准：")
    for line in SCORE_GUIDE:
        print(line)


def _print_manual_template(
    title: str,
    evaluation: Mapping[str, Any],
) -> None:
    print(f"\n【{title}】")
    records = evaluation["evaluations"]
    for criterion in MANUAL_CRITERIA:
        record = records[criterion["key"]]
        score = record["score"] if record["score"] is not None else "____"
        reason = record["reason"] or "________________________________"
        print(f"\n{criterion['name']}：")
        print(f"评分：{score}")
        print(f"评价：{reason}")


def _print_method_note() -> None:
    print("\n========== 实验方法说明 ==========")
    print(
        "本实验采用人工评价方式，对无 RAG 与有 RAG 两套营销策略进行独立评价。\n\n"
        "评价维度包括商品相关性、目标人群匹配度、场景匹配度、用户需求合理性、"
        "营销方向合理性、渠道可执行性、内容可执行性和风险控制完整性。\n\n"
        "人工评价用于辅助判断策略质量，不将单次人工评分视为统计意义上的模型性能结论。"
    )


def run_manual_evaluation() -> int:
    print("========== Step 12.3 人工质量评估 ==========")
    _print_product()

    product_analysis = product_analyzer.analyze_product(**PRODUCT)
    if not isinstance(product_analysis, ProductAnalysis):
        raise TypeError("商品分析未返回 ProductAnalysis")

    without_rag = generate_without_rag(product_analysis)
    if not isinstance(without_rag, MarketingStrategy):
        raise TypeError("无 RAG 方案未返回 MarketingStrategy")

    query = build_retrieval_query(product_analysis)
    documents = retrieve_knowledge(query, top_k=TOP_K)
    with_rag = generate_marketing_strategy(product_analysis, documents)
    if not isinstance(with_rag, MarketingStrategy):
        raise TypeError("RAG 方案未返回 MarketingStrategy")

    print("\n========== 方案 A：无 RAG ==========")
    _print_strategy(without_rag)
    print("\n========== 方案 B：RAG ==========")
    _print_strategy(with_rag)

    evaluation_a = evaluate_strategy_manually(without_rag)
    evaluation_b = evaluate_strategy_manually(with_rag)
    _print_criteria()
    _print_manual_template("方案 A：无 RAG", evaluation_a)
    _print_manual_template("方案 B：RAG", evaluation_b)
    _print_method_note()
    return 0


def main() -> int:
    try:
        return run_manual_evaluation()
    except Exception as exc:
        print("\n========== 人工质量评估实验失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
