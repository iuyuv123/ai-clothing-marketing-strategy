"""Step 12.4 multi-product human evaluation for RAG A/B strategies."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for module_path in (PROJECT_ROOT, APP_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import product_analyzer  # noqa: E402
from evaluation_test import (  # noqa: E402
    STRATEGY_FIELDS,
    TOP_K,
    generate_without_rag,
)
from manual_evaluation import (  # noqa: E402
    MANUAL_CRITERIA,
    evaluate_strategy_manually,
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


PRODUCTS: tuple[dict[str, str], ...] = (
    {
        "brand_name": "AAA",
        "product_name": "户外三合一冲锋衣",
        "product_type": "户外服装",
        "confirmed_features": "三合一设计、防风、防水",
        "usage_scenarios": "登山、徒步、户外运动",
    },
    {
        "brand_name": "AAA",
        "product_name": "羽绒服",
        "product_type": "冬季服装",
        "confirmed_features": "羽绒填充、保暖、轻便",
        "usage_scenarios": "冬季通勤、日常出行、旅行",
    },
    {
        "brand_name": "AAA",
        "product_name": "防晒衣",
        "product_type": "户外服装",
        "confirmed_features": "防晒、轻薄、便于携带",
        "usage_scenarios": "户外出行、旅行、日常通勤",
    },
)


def _print_product(product: dict[str, str]) -> None:
    print("【商品输入】")
    print(f"品牌：{product['brand_name']}")
    print(f"商品名称：{product['product_name']}")
    print(f"商品类型：{product['product_type']}")
    print(f"已确认卖点：{product['confirmed_features']}")
    print(f"使用场景：{product['usage_scenarios']}")


def _print_product_analysis(product_analysis: ProductAnalysis) -> None:
    print("\n【商品理解结果】")
    for field, value in product_analysis.model_dump().items():
        print(f"{field}：{value}")


def _print_strategy(strategy: MarketingStrategy) -> None:
    strategy_data = strategy.model_dump()
    for field in STRATEGY_FIELDS:
        print(f"{field}：{strategy_data.get(field)}")


def _knowledge_sources(documents: list[Any]) -> list[str]:
    return sorted(
        {
            str((getattr(document, "metadata", {}) or {}).get("source"))
            for document in documents
            if (getattr(document, "metadata", {}) or {}).get("source")
        }
    )


def _score_text(score: Any) -> str:
    return "待人工评分" if score is None else str(score)


def _print_manual_table(
    evaluation_a: dict[str, Any],
    evaluation_b: dict[str, Any],
) -> None:
    print("\n---------- 人工评价 ----------")
    print("评分标准：1=很差，2=较差，3=一般，4=较好，5=很好")
    print("| 评价指标 | A：无RAG | B：有RAG | A评分理由 | B评分理由 |")
    print("| --- | --- | --- | --- | --- |")
    records_a = evaluation_a["evaluations"]
    records_b = evaluation_b["evaluations"]
    for criterion in MANUAL_CRITERIA:
        record_a = records_a[criterion["key"]]
        record_b = records_b[criterion["key"]]
        print(
            f"| {criterion['name']} "
            f"| {_score_text(record_a['score'])} "
            f"| {_score_text(record_b['score'])} "
            f"| {record_a['reason']} "
            f"| {record_b['reason']} |"
        )


def evaluate_product(product_index: int, product: dict[str, str]) -> None:
    print(f"\n========== 商品 {product_index}：{product['product_name']} ==========")
    _print_product(product)

    product_analysis = product_analyzer.analyze_product(**product)
    if not isinstance(product_analysis, ProductAnalysis):
        raise TypeError("商品分析未返回 ProductAnalysis")
    _print_product_analysis(product_analysis)

    without_rag = generate_without_rag(product_analysis)
    if not isinstance(without_rag, MarketingStrategy):
        raise TypeError("无 RAG 方案未返回 MarketingStrategy")
    print("\n---------- A：无 RAG ----------")
    _print_strategy(without_rag)

    query = build_retrieval_query(product_analysis)
    documents = retrieve_knowledge(query, top_k=TOP_K)
    with_rag = generate_marketing_strategy(product_analysis, documents)
    if not isinstance(with_rag, MarketingStrategy):
        raise TypeError("RAG 方案未返回 MarketingStrategy")
    print("\n---------- B：有 RAG ----------")
    print(f"动态 RAG Query：{query}")
    print(f"RAG 检索数量：{len(documents)}")
    print(f"检索到的知识来源：{_knowledge_sources(documents) or '无'}")
    _print_strategy(with_rag)

    evaluation_a = evaluate_strategy_manually(without_rag)
    evaluation_b = evaluate_strategy_manually(with_rag)
    _print_manual_table(evaluation_a, evaluation_b)


def _print_experiment_note() -> None:
    print("\n========== 多商品人工评估说明 ==========")
    print("本实验用于比较不同商品在无 RAG 与有 RAG 条件下生成的营销策略，并通过人工评价观察：")
    for index, criterion in enumerate(MANUAL_CRITERIA, 1):
        print(f"{index}. {criterion['name']}")
    print("\n人工评价不作为模型自动评分结果。")
    print(
        "本实验用于观察不同商品和不同知识条件下的策略差异，"
        "不直接据此证明 RAG 一定提升策略质量。"
    )


def main() -> int:
    try:
        for index, product in enumerate(PRODUCTS, 1):
            evaluate_product(index, product)
        _print_experiment_note()
        return 0
    except Exception as exc:
        print("\n========== 多商品人工评估执行失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
