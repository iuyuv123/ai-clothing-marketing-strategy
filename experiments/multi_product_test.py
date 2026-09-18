"""Validate that the marketing strategy pipeline works for multiple products."""

from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import dashscope
from dotenv import load_dotenv
from langchain_chroma import Chroma

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    generate_marketing_strategy,
)
from product_analyzer import ProductAnalysis, analyze_product  # noqa: E402
from rag_context import build_rag_context  # noqa: E402
from vector_store import (  # noqa: E402
    DashScopeEmbeddings,
    retrieve_marketing_knowledge,
)


PERSIST_DIR = PROJECT_ROOT / "vector_store" / "marketing_strategy_kb"
COLLECTION_NAME = "marketing_strategy_kb"
TOP_K = 5

PRODUCTS: tuple[dict[str, str], ...] = (
    {
        "品牌": "AAA",
        "商品名称": "户外三合一冲锋衣",
        "商品类型": "户外服装",
        "已确认商品特征": "三合一设计、防风、防水",
        "使用场景": "登山、徒步、户外运动",
    },
    {
        "品牌": "AAA",
        "商品名称": "羽绒服",
        "商品类型": "冬季服装",
        "已确认商品特征": "羽绒填充、保暖、轻便",
        "使用场景": "冬季通勤、日常出行、旅行",
    },
    {
        "品牌": "AAA",
        "商品名称": "防晒衣",
        "商品类型": "户外服装",
        "已确认商品特征": "防晒、轻薄、便于携带",
        "使用场景": "户外出行、旅行、日常通勤",
    },
)


def _load_embedding_model() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    if not model:
        raise RuntimeError("DASHSCOPE_EMBEDDING_MODEL 未设置")
    dashscope.api_key = api_key
    return model


def _build_store() -> Chroma:
    if not PERSIST_DIR.is_dir():
        raise FileNotFoundError("正式营销策略知识库不存在，请先完成知识库构建。")
    model = _load_embedding_model()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=DashScopeEmbeddings(model=model),
    )
    return store


def build_dynamic_rag_query(product_analysis: ProductAnalysis) -> str:
    """Build a product-specific query from the current analysis."""
    fields = (
        product_analysis.product_name,
        product_analysis.product_type,
        *product_analysis.confirmed_features,
        *product_analysis.core_selling_points,
        *product_analysis.target_audience,
        *product_analysis.usage_scenarios,
        *product_analysis.user_needs,
    )
    return "；".join(str(item).strip() for item in fields if str(item).strip())


def retrieve_knowledge(store: Chroma, product_analysis: ProductAnalysis) -> list[Any]:
    query = build_dynamic_rag_query(product_analysis)
    return retrieve_marketing_knowledge(store, query, top_k=TOP_K)


def _print_items(label: str, value: Any) -> None:
    print(f"{label}：")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value, 1):
            print(f"{index}. {item}")
    else:
        print(value)


def print_product_analysis(analysis: ProductAnalysis) -> None:
    _print_items("商品定位/商品类型", analysis.product_type)
    _print_items("确认商品特征", analysis.confirmed_features)
    _print_items("核心卖点", analysis.core_selling_points)
    _print_items("目标人群", analysis.target_audience)
    _print_items("使用场景", analysis.usage_scenarios)
    _print_items("用户需求", analysis.user_needs)
    _print_items("消费者价值", analysis.consumer_value)


def print_rag_results(documents: list[Any]) -> None:
    print("\n【2. RAG 检索】")
    print(f"RAG 检索数量：{len(documents)}")
    candidate_count = (
        (getattr(documents[0], "metadata", {}) or {}).get("retrieval_candidate_count", 0)
        if documents
        else 0
    )
    print("\n【RAG 检索策略】")
    print(f"候选结果数量：{candidate_count}")
    print(f"最终结果数量：{len(documents)}")
    if not documents:
        print("检索到的知识来源：无")
        return
    category_counts: Counter[str] = Counter()
    print("\n【RAG 检索详情】")
    for index, document in enumerate(documents[:TOP_K], 1):
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        filename = metadata.get("filename") or Path(str(source)).name or "unknown"
        category = _source_category(metadata, source)
        score = metadata.get("retrieval_score")
        category_counts[category] += 1
        content = (getattr(document, "page_content", "") or "").strip()
        print(f"Rank {index}")
        print(f"source：{source}")
        print(f"文件名：{filename}")
        print(f"category：{category}")
        print(f"score：{score if score is not None else 'unknown'}")
        print(f"content：{content[:150]}")
    print("\n【RAG 来源分布】")
    for category in ("brand", "marketing_rules", "marketing_cases", "platform_rules", "other"):
        print(f"{category}：{category_counts.get(category, 0)}")


def _source_category(metadata: dict[str, Any], source: str) -> str:
    category = str(metadata.get("category") or "").strip()
    if category in {"brand", "marketing_rules", "marketing_cases", "platform_rules"}:
        return category
    normalized = source.replace("\\", "/")
    for candidate in ("brand", "marketing_rules", "marketing_cases", "platform_rules"):
        if f"/{candidate}/" in f"/{normalized}/":
            return candidate
    return "other"


def print_strategy(strategy: MarketingStrategy) -> None:
    print("\n【3. 最终营销策略】")
    _print_items("产品定位", strategy.product_positioning)
    _print_items("核心营销方向", strategy.core_marketing_direction)
    _print_items("渠道策略", strategy.channel_strategy)
    _print_items("内容方向", strategy.content_direction)
    _print_items("执行建议", strategy.execution_suggestions)
    _print_items("风险控制", strategy.risk_control)


def run_single_product_test(product: dict[str, str], store: Chroma) -> dict[str, Any]:
    name = product["商品名称"]
    print(f"\n# ============================== 测试商品：{name}")
    try:
        analysis = analyze_product(
            brand_name=product["品牌"],
            product_name=product["商品名称"],
            product_type=product["商品类型"],
            confirmed_features=product["已确认商品特征"],
            usage_scenarios=product["使用场景"],
        )
        if not isinstance(analysis, ProductAnalysis):
            raise TypeError("商品分析未返回 ProductAnalysis")
        print("\n【1. 商品分析】")
        print_product_analysis(analysis)

        query = build_dynamic_rag_query(analysis)
        print("\n【RAG 动态查询】")
        print(query)
        documents = retrieve_marketing_knowledge(store, query, top_k=TOP_K)
        print_rag_results(documents)
        if not documents:
            print("[FAIL] RAG 未返回知识")

        context = build_rag_context(analysis, documents)
        strategy = generate_marketing_strategy(analysis, documents)
        if not isinstance(strategy, MarketingStrategy):
            raise TypeError("营销策略生成未返回 MarketingStrategy")
        print_strategy(strategy)
        return {
            "input": product,
            "analysis": analysis,
            "query": query,
            "documents": documents,
            "context": context,
            "strategy": strategy,
            "error": None,
        }
    except Exception as exc:
        print(f"[ERROR] 商品：{name}")
        print(f"原因：{type(exc).__name__}: {exc}")
        return {"input": product, "error": exc}


def _value_signature(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "；".join(str(item).strip() for item in value)
    return str(value).strip()


def _check(label: str, passed: bool, reason: str) -> bool:
    print(f"[{ 'PASS' if passed else 'FAIL' }] {label}：{reason}")
    return passed


def _query_terms(query: str) -> set[str]:
    """Split the actual Chinese/English query separators without NLP dependencies."""
    return {
        term.strip("：:;；，,。！？!?、/|()（）[]【】\"'")
        for term in re.split(r"[；;，,。！？!?、/|\s]+", query)
        if term.strip("：:;；，,。！？!?、/|()（）[]【】\"'")
    }


def print_query_diagnostics(results: list[dict[str, Any]]) -> None:
    successful = [result for result in results if result.get("error") is None]
    if not successful:
        return
    print("\n【Query 差异检查】")
    query_terms = []
    for index, result in enumerate(successful, 1):
        query = result.get("query", "")
        terms = _query_terms(query)
        query_terms.append(terms)
        print(f"商品{index} query：{query}")
    common = set.intersection(*query_terms) if query_terms else set()
    print(f"三个 query 公共词：{'、'.join(sorted(common)) or '无'}")
    for index, terms in enumerate(query_terms, 1):
        other_terms = [other for other_index, other in enumerate(query_terms) if other_index != index]
        others_union = set.union(*other_terms) if other_terms else set()
        unique = terms - others_union
        print(f"商品{index} 特有词：{'、'.join(sorted(unique)) or '无'}")


def _document_signature(document: Any) -> tuple[str, str]:
    metadata = getattr(document, "metadata", {}) or {}
    source = str(metadata.get("source") or "unknown")
    content = (getattr(document, "page_content", "") or "").strip()
    return source, content[:200]


def _product_terms(product: dict[str, str], analysis: ProductAnalysis) -> set[str]:
    raw_values = [
        product["商品名称"],
        product["商品类型"],
        product["已确认商品特征"],
        product["使用场景"],
        *analysis.confirmed_features,
        *analysis.usage_scenarios,
    ]
    return {
        term
        for value in raw_values
        for term in _query_terms(str(value))
        if len(term) >= 2
    }


def has_product_relevant_knowledge(result: dict[str, Any]) -> bool:
    """Use transparent substring evidence only for this test's diagnostics."""
    terms = _product_terms(result["input"], result["analysis"])
    for document in result.get("documents", []):
        metadata = getattr(document, "metadata", {}) or {}
        source = str(metadata.get("source") or "unknown")
        content = (getattr(document, "page_content", "") or "").strip()
        haystack = f"{source} {content}"
        if any(term in haystack for term in terms):
            return True
    return False


def print_retrieval_differences(results: list[dict[str, Any]]) -> None:
    successful = [result for result in results if result.get("error") is None]
    if len(successful) < 2:
        return
    print("\n【商品间检索差异】")
    for left_index in range(len(successful)):
        for right_index in range(left_index + 1, len(successful)):
            left = {_document_signature(document) for document in successful[left_index]["documents"]}
            right = {_document_signature(document) for document in successful[right_index]["documents"]}
            left_sources = {source for source, _ in left}
            right_sources = {source for source, _ in right}
            shared_sources = left_sources & right_sources
            different_sources = left_sources ^ right_sources
            different_documents = len(left ^ right)
            print(f"商品{left_index + 1} vs 商品{right_index + 1}")
            print(f"共同文档数量：{len(left & right)}")
            print(f"不同文档数量：{different_documents}")
            print(f"共同检索来源数量：{len(shared_sources)}")
            print(f"不同检索来源数量：{len(different_sources)}")


def print_rag_acceptance(results: list[dict[str, Any]]) -> bool:
    successful = [result for result in results if result.get("error") is None]
    if not successful:
        print("\n【RAG 多商品通用性验收】\n[RAG FAIL]")
        return False

    queries = [result.get("query", "") for result in successful]
    query_is_unique = [queries.count(query) == 1 for query in queries]
    result_sets = [
        {_document_signature(document) for document in result.get("documents", [])}
        for result in successful
    ]
    result_changed = [
        any(result_sets[index] != other for other_index, other in enumerate(result_sets) if other_index != index)
        for index in range(len(result_sets))
    ]
    relevance = [has_product_relevant_knowledge(result) for result in successful]

    print("\n【RAG 多商品通用性验收】")
    for index, (query_ok, changed, relevant) in enumerate(zip(query_is_unique, result_changed, relevance), 1):
        print(f"商品{index}：")
        print(f"Query 是否不同：{'是' if query_ok else '否'}")
        print(f"检索结果是否变化：{'是' if changed else '否'}")
        print(f"是否存在商品相关知识：{'是' if relevant else '否'}")

    query_diff = len(set(queries)) > 1
    at_least_one_result_diff = any(
        left != right
        for index, left in enumerate(result_sets)
        for right in result_sets[index + 1 :]
    )
    relevant_evidence = any(relevance)
    passed = query_diff and at_least_one_result_diff and relevant_evidence
    print(f"\n{'[RAG PASS]' if passed else '[RAG FAIL]'}")
    return passed


def print_summary(results: list[dict[str, Any]]) -> bool:
    print("\n# ======================================== 多商品通用性测试结论")
    successful = [result for result in results if result.get("error") is None]
    failed = [result for result in results if result.get("error") is not None]
    print(f"成功商品数量：{len(successful)}")
    print(f"失败商品数量：{len(failed)}")
    print_query_diagnostics(results)
    print_retrieval_differences(results)
    rag_acceptance = print_rag_acceptance(results)

    checks: list[bool] = []
    input_names = [product["商品名称"] for product in PRODUCTS]
    checks.append(_check("商品名称不同", len(set(input_names)) == len(input_names), str(input_names)))

    if len(successful) == len(PRODUCTS):
        analyses = [result["analysis"] for result in successful]
        strategies = [result["strategy"] for result in successful]
        checks.append(
            _check(
                "商品分析名称与输入一致",
                all(a.product_name == r["input"]["商品名称"] for a, r in zip(analyses, successful)),
                "ProductAnalysis.product_name 已逐项核对",
            )
        )
        checks.append(
            _check(
                "商品分析是否随商品变化",
                len({_value_signature(a.core_selling_points) for a in analyses}) > 1,
                "核心卖点分析存在差异" if len({_value_signature(a.core_selling_points) for a in analyses}) > 1 else "核心卖点分析完全相同",
            )
        )
        for label, values in (
            ("目标人群是否动态变化", [a.target_audience for a in analyses]),
            ("使用场景是否动态变化", [a.usage_scenarios for a in analyses]),
            ("用户需求是否动态变化", [a.user_needs for a in analyses]),
            ("产品定位是否变化", [s.product_positioning for s in strategies]),
            ("最终营销方向是否变化", [s.core_marketing_direction for s in strategies]),
            ("渠道策略是否变化", [s.channel_strategy for s in strategies]),
            ("内容方向是否变化", [s.content_direction for s in strategies]),
        ):
            signatures = {_value_signature(value) for value in values}
            checks.append(_check(label, len(signatures) > 1, f"不同结果数：{len(signatures)}"))

        rag_ok = all(result["documents"] for result in successful)
        checks.append(_check("每个商品均检索到知识", rag_ok, "每个商品至少返回一条" if rag_ok else "存在空检索结果"))
        checks.append(
            _check(
                "RAG 检索结果随商品语义变化",
                rag_acceptance,
                "Query、实际文档集合和商品相关性均满足要求"
                if rag_acceptance
                else "Query 或实际文档集合未体现商品语义差异，或缺少相关知识",
            )
        )
        risk_values = [_value_signature(strategy.risk_control) for strategy in strategies]
        checks.append(_check("风险控制保持合规边界", all(risk_values), "三个策略均有风险控制内容"))
        checks.append(
            _check(
                "不存在明显冲锋衣硬编码",
                any("冲锋衣" not in _value_signature(a.product_name) for a in analyses[1:]),
                "羽绒服和防晒衣分析未被命名为冲锋衣" if any("冲锋衣" not in _value_signature(a.product_name) for a in analyses[1:]) else "存在冲锋衣名称硬编码",
            )
        )
    else:
        checks.append(_check("三个商品均成功完成流程", False, "存在失败商品，无法完成完整通用性比较"))

    overall = len(successful) == len(PRODUCTS) and all(checks)
    print(f"\n总体结论：{'[PASS]' if overall else '[FAIL]'}")
    if failed:
        print("存在失败商品，整体不能判定为 PASS。")
    return overall


def run_all_tests() -> int:
    store = _build_store()
    results = [run_single_product_test(product, store) for product in PRODUCTS]
    return 0 if print_summary(results) else 1


if __name__ == "__main__":
    raise SystemExit(run_all_tests())
