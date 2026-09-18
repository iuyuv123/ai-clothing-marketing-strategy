"""End-to-end acceptance test for product analysis, RAG, and strategy generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# TODO: This legacy test references the removed app.workflow module and requires migration to the current marketing_strategy_workflow module.

# Make the existing app modules (which use project-local absolute imports)
# resolvable when this test is launched directly from PyCharm.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    generate_marketing_strategy,
)
# workflow.py imports this project-local module by its absolute name; reuse the
# same class object so isinstance checks do not split across duplicate modules.
from product_analyzer import ProductAnalysis  # noqa: E402
from app.workflow import analyze_product, retrieve_knowledge  # noqa: E402


PRODUCT = {
    "品牌名称": "AAA",
    "商品类型": "户外三合一冲锋衣",
    "商品卖点": "三合一设计、防风、防水",
    "目标人群": "待根据当前商品动态分析",
    "营销场景": "登山、徒步、户外运动",
    "用户需求": "待根据商品特点和使用场景分析",
}

RETRIEVAL_QUERIES = (
    "户外冲锋衣适合哪些使用场景",
    "如何根据商品特点分析用户需求",
    "如何制定服装营销方案",
)


def _print_analysis(analysis: Any) -> None:
    if hasattr(analysis, "model_dump_json"):
        print(analysis.model_dump_json(ensure_ascii=False, indent=2))
    else:
        print(json.dumps(analysis, ensure_ascii=False, indent=2, default=str))


def _print_documents(documents: list[Any]) -> None:
    for index, document in enumerate(documents, 1):
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        content = (getattr(document, "page_content", "") or "").strip()
        print(f"\n知识{index}：")
        print(f"source：{source}")
        print(f"内容：{content[:200]}")


def _print_strategy(strategy: MarketingStrategy) -> None:
    labels = (
        ("产品定位", "product_positioning"),
        ("核心卖点", "core_selling_points"),
        ("目标人群", "target_audience"),
        ("使用场景", "usage_scenarios"),
        ("用户需求", "user_needs"),
        ("消费者价值", "consumer_value"),
        ("核心营销方向", "core_marketing_direction"),
        ("渠道策略", "channel_strategy"),
        ("内容方向", "content_direction"),
        ("执行建议", "execution_suggestions"),
        ("风险控制", "risk_control"),
    )
    for label, field_name in labels:
        value = getattr(strategy, field_name)
        print(f"{label}：{value if isinstance(value, str) else '；'.join(value)}")


def run_acceptance_test() -> int:
    # The current project exposes analysis and retrieval through workflow.py;
    # this test reuses those functions instead of recreating either component.
    product_analysis = analyze_product(PRODUCT, trusted_knowledge=[])
    if not isinstance(product_analysis, ProductAnalysis):
        raise TypeError("商品分析结果不是 ProductAnalysis")

    documents: list[Any] = []
    for query in RETRIEVAL_QUERIES:
        documents.extend(retrieve_knowledge(PRODUCT, top_k=3, query=query))

    strategy = generate_marketing_strategy(product_analysis, documents)
    if not isinstance(strategy, MarketingStrategy):
        raise TypeError("generate_marketing_strategy() 未返回 MarketingStrategy")

    required_fields = (
        "product_positioning",
        "core_selling_points",
        "target_audience",
        "usage_scenarios",
        "user_needs",
        "consumer_value",
        "core_marketing_direction",
        "channel_strategy",
        "content_direction",
        "execution_suggestions",
        "risk_control",
    )
    missing = [field for field in required_fields if not hasattr(strategy, field)]
    if missing:
        raise ValueError(f"MarketingStrategy 缺少字段：{missing}")

    print("========== AI 服装营销策略助手验收 ==========")
    print("\n【1. 当前商品】")
    print(f"商品名称：{PRODUCT['商品类型']}")
    print(f"商品卖点：{PRODUCT['商品卖点']}")
    print(f"使用场景：{PRODUCT['营销场景']}")

    print("\n【2. 商品分析结果】")
    _print_analysis(product_analysis)

    print("\n【3. RAG 检索结果】")
    _print_documents(documents)

    print("\n【4. 最终营销策略】")
    _print_strategy(strategy)
    print("\n结构化结果：MarketingStrategy")
    print("验收通过：商品分析、RAG 检索、策略生成和字段校验均完成")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_acceptance_test())
    except Exception as exc:
        print("========== 验收失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        raise


