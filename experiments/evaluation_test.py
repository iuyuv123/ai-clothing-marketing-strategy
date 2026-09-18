"""Step 12.2 reproducible A/B evaluation experiment for RAG effectiveness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for module_path in (PROJECT_ROOT, APP_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import product_analyzer  # noqa: E402
from evaluation_metrics import (  # noqa: E402
    evaluate_fact_boundary,
    evaluate_product_understanding,
    evaluate_rag_effectiveness,
    evaluate_strategy_completeness,
    evaluate_strategy_relevance,
)
from marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    _build_llm,
    generate_marketing_strategy,
)
from marketing_strategy_workflow import (  # noqa: E402
    build_retrieval_query,
    retrieve_knowledge,
)
from product_analyzer import ProductAnalysis  # noqa: E402


TOP_K = 5
PRODUCT = {
    "brand_name": "AAA",
    "product_name": "户外三合一冲锋衣",
    "product_type": "户外服装",
    "confirmed_features": "三合一设计、防风、防水",
    "usage_scenarios": "登山、徒步、户外运动",
}

STRATEGY_FIELDS = (
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


def _without_rag_prompt(product_analysis: ProductAnalysis) -> str:
    analysis_json = product_analysis.model_dump_json()
    return f"""你正在为“AI 服装营销策略助手”生成结构化营销策略。

本方案不使用知识库或 RAG。你只能根据【当前商品分析】生成策略。
不得补充商品分析中没有提供的属性、材质、参数、等级、测试数据、认证、专利、性能或使用能力。
使用场景不能反向证明商品具备更严苛环境下的能力。正向策略字段只描述当前商品事实及合理营销分析；
风险提醒集中写入 risk_control，不要混入其他正向字段。

只返回合法 JSON，不要 Markdown、代码围栏或额外说明。JSON 必须严格包含以下字段：
{{
  "product_positioning": "",
  "core_selling_points": [],
  "target_audience": [],
  "usage_scenarios": [],
  "user_needs": [],
  "consumer_value": [],
  "core_marketing_direction": [],
  "channel_strategy": [],
  "content_direction": [],
  "execution_suggestions": [],
  "risk_control": []
}}

【当前商品分析】
{analysis_json}
"""


def generate_without_rag(product_analysis: ProductAnalysis) -> MarketingStrategy:
    """Generate the baseline strategy without retrieval or RAG context."""
    response = _build_llm().invoke(_without_rag_prompt(product_analysis))
    try:
        data = json.loads(response.content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("无 RAG 方案模型返回结果不是合法 JSON") from exc
    return MarketingStrategy.model_validate(data)


def _status(passed: bool) -> str:
    return "[PASS]" if passed else "[FAIL]"


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


def _retrieved_sources(documents: list[Any]) -> list[str]:
    return sorted(
        {
            str((getattr(document, "metadata", {}) or {}).get("source"))
            for document in documents
            if (getattr(document, "metadata", {}) or {}).get("source")
        }
    )


def run_evaluation() -> int:
    print("========== Step 12.2 效果评估实验 ==========\n")
    _print_product()

    product_analysis = product_analyzer.analyze_product(**PRODUCT)
    analysis_type_ok = isinstance(product_analysis, ProductAnalysis)
    understanding = evaluate_product_understanding(
        product_analysis,
        product_name=PRODUCT["product_name"],
        product_type=PRODUCT["product_type"],
        confirmed_features=PRODUCT["confirmed_features"],
        usage_scenarios=PRODUCT["usage_scenarios"],
    )
    product_understanding_ok = analysis_type_ok and understanding["passed"]
    print("\n========== 商品理解 ==========")
    print(_status(product_understanding_ok))
    print(f"逐项检查：{understanding['checks']}")

    without_rag = generate_without_rag(product_analysis)
    without_rag_generated = isinstance(without_rag, MarketingStrategy)
    without_rag_completeness = evaluate_strategy_completeness(without_rag)
    without_rag_boundary = evaluate_fact_boundary(
        without_rag, PRODUCT["confirmed_features"]
    )
    without_rag_ok = (
        without_rag_generated
        and without_rag_completeness["passed"]
        and without_rag_boundary["passed"]
    )
    print("\n========== 方案 A：无 RAG ==========")
    print(f"策略生成：{_status(without_rag_generated)}")
    print(f"策略完整性：{_status(without_rag_completeness['passed'])}")
    print(f"完整字段数量：{without_rag_completeness['completed_fields']}/11")
    print(f"事实边界：{_status(without_rag_boundary['passed'])}")
    print(f"事实边界命中项：{without_rag_boundary['matched_unconfirmed_terms'] or '无'}")
    _print_strategy(without_rag)

    query = build_retrieval_query(product_analysis)
    documents = retrieve_knowledge(query, top_k=TOP_K)
    with_rag = generate_marketing_strategy(product_analysis, documents)
    with_rag_generated = isinstance(with_rag, MarketingStrategy)
    with_rag_relevance = evaluate_strategy_relevance(product_analysis, with_rag)
    with_rag_completeness = evaluate_strategy_completeness(with_rag)
    with_rag_boundary = evaluate_fact_boundary(
        with_rag, PRODUCT["confirmed_features"]
    )
    with_rag_ok = (
        with_rag_generated
        and with_rag_relevance["passed"]
        and with_rag_completeness["passed"]
        and with_rag_boundary["passed"]
    )
    sources = _retrieved_sources(documents)
    print("\n========== 方案 B：RAG ==========")
    print(f"RAG Query：{query}")
    print(f"RAG 检索数量：{len(documents)}")
    print(f"实际检索来源：{sources or '无'}")
    print(f"策略生成：{_status(with_rag_generated)}")
    print(f"策略相关性：{_status(with_rag_relevance['passed'])}")
    print(f"相关字段数量：{with_rag_relevance['matched_fields']}/{with_rag_relevance['total_fields']}")
    print(f"策略完整性：{_status(with_rag_completeness['passed'])}")
    print(f"完整字段数量：{with_rag_completeness['completed_fields']}/11")
    print(f"事实边界：{_status(with_rag_boundary['passed'])}")
    print(f"事实边界命中项：{with_rag_boundary['matched_unconfirmed_terms'] or '无'}")
    _print_strategy(with_rag)

    rag_effectiveness = evaluate_rag_effectiveness(
        without_rag,
        with_rag,
        documents,
    )
    rag_checks = rag_effectiveness["checks"]
    print("\n========== RAG 效果对比 ==========")
    print(f"两套策略是否存在差异：{'是' if rag_checks['strategies_differ'] else '否'}")
    print(f"不同字段数量：{len(rag_effectiveness['different_fields'])}")
    print(f"不同字段：{rag_effectiveness['different_fields'] or '无'}")
    print(f"RAG 检索是否为空：{'否' if rag_checks['retrieval_non_empty'] else '是'}")
    print(f"检索文档是否有效：{_status(rag_checks['retrieved_documents_valid'])}")
    print(
        "RAG 知识参与证据："
        f"{rag_effectiveness['knowledge_overlap_evidence'] or '未检测到文本重合证据'}"
    )
    print(f"RAG 知识是否参与：{_status(rag_checks['retrieved_knowledge_used'])}")
    print("说明：本实验验证可观察变化与知识依据，不声称 RAG 必然提升策略质量。")

    overall = (
        product_understanding_ok
        and without_rag_ok
        and with_rag_ok
        and rag_effectiveness["passed"]
    )
    print("\n========== 最终评估 ==========")
    print(f"商品理解：{_status(product_understanding_ok)}")
    print(f"无 RAG：{_status(without_rag_ok)}")
    print(f"有 RAG：{_status(with_rag_ok)}")
    print(f"RAG 有效性：{_status(rag_effectiveness['passed'])}")
    print(f"总体结论：{_status(overall)}")
    return 0 if overall else 1


def main() -> int:
    try:
        return run_evaluation()
    except Exception as exc:
        print("\n========== 评估实验失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
