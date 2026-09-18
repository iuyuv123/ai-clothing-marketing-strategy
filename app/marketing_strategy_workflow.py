"""Main workflow for the AI clothing marketing strategy assistant."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import dashscope
from dotenv import load_dotenv
from langchain_chroma import Chroma

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    generate_marketing_strategy,
)
from app.rag_context import build_rag_context  # noqa: E402
import product_analyzer  # noqa: E402
from product_analyzer import ProductAnalysis  # noqa: E402
from vector_store import DashScopeEmbeddings  # noqa: E402


PERSIST_DIR = PROJECT_ROOT / "vector_store" / "marketing_strategy_kb"
COLLECTION_NAME = "marketing_strategy_kb"
TOP_K = 5


def build_product_input() -> dict[str, str]:
    """Return the current runtime product input used by this demonstration."""
    return {
        "品牌名称": "AAA",
        "商品名称": "户外三合一冲锋衣",
        "商品类型": "户外服装",
        "已确认商品事实": "三合一设计、防风、防水",
        "使用场景": "登山、徒步、户外运动",
    }


def _analyze_product(product: dict[str, str]) -> ProductAnalysis:
    """Delegate product analysis to the existing product_analyzer module."""
    analysis = product_analyzer.analyze_product(
        brand_name=product["品牌名称"],
        product_name=product["商品名称"],
        product_type=product["商品类型"],
        confirmed_features=product["已确认商品事实"],
        usage_scenarios=product["使用场景"],
    )
    if not isinstance(analysis, ProductAnalysis):
        raise TypeError("商品分析模块未返回 ProductAnalysis")
    return analysis


def _embedding_model() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置，请检查项目 .env 配置。")
    if not model:
        raise RuntimeError("DASHSCOPE_EMBEDDING_MODEL 未设置，请检查项目 .env 配置。")
    dashscope.api_key = api_key
    return model


def build_retrieval_query(product_analysis: ProductAnalysis) -> str:
    """Build a query from the actual analysis instead of a fixed example string."""
    fields = (
        product_analysis.product_name,
        product_analysis.product_type,
        *product_analysis.confirmed_features,
        *product_analysis.core_selling_points,
        *product_analysis.target_audience,
        *product_analysis.usage_scenarios,
        *product_analysis.user_needs,
        "营销策略",
        "用户需求",
        "场景营销",
    )
    return "；".join(str(item).strip() for item in fields if str(item).strip())


def retrieve_knowledge(query: str, top_k: int = TOP_K) -> list[Any]:
    """Load the formal Chroma store and retrieve strategy knowledge."""
    if not PERSIST_DIR.is_dir():
        raise FileNotFoundError("正式营销策略知识库不存在，请先完成知识库构建。")

    model = _embedding_model()
    try:
        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(PERSIST_DIR),
            embedding_function=DashScopeEmbeddings(model=model),
        )
        return list(
            store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            ).invoke(query)
        )
    except Exception as exc:
        raise RuntimeError(f"RAG 检索失败：{exc}") from exc


def _print_value(label: str, value: Any) -> None:
    print(f"\n========== {label} ==========")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value, 1):
            print(f"{index}. {item}")
    else:
        print(value)


def print_product_analysis(product_analysis: ProductAnalysis) -> None:
    print("========== 1. 商品分析 ==========")
    fields = (
        ("品牌", "brand_name"),
        ("商品名称", "product_name"),
        ("商品类型", "product_type"),
        ("已确认商品事实", "confirmed_features"),
        ("核心卖点", "core_selling_points"),
        ("目标人群", "target_audience"),
        ("使用场景", "usage_scenarios"),
        ("用户需求", "user_needs"),
        ("消费者价值", "consumer_value"),
    )
    for label, field_name in fields:
        _print_value(label, getattr(product_analysis, field_name))


def print_retrieved_knowledge(documents: list[Any]) -> None:
    print("\n========== 2. RAG 知识检索 ==========")
    print(f"检索结果数量：{len(documents)}")
    if not documents:
        print("未检索到相关知识，将基于当前商品分析生成策略。")
        return
    for index, document in enumerate(documents[:TOP_K], 1):
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        content = (getattr(document, "page_content", "") or "").strip()
        print(f"\n知识{index}：")
        print(f"来源：{source}")
        print(f"内容：{content[:500]}")


def print_strategy(strategy: MarketingStrategy) -> None:
    print("\n========== 4. AI 营销策略 ==========")
    fields = (
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
    for label, field_name in fields:
        _print_value(label, getattr(strategy, field_name))


def main() -> None:
    product = build_product_input()
    try:
        product_analysis = _analyze_product(product)
    except Exception as exc:
        raise RuntimeError(f"商品分析失败：{exc}") from exc
    print_product_analysis(product_analysis)

    query = build_retrieval_query(product_analysis)
    documents = retrieve_knowledge(query, top_k=TOP_K)
    print_retrieved_knowledge(documents)

    context = build_rag_context(product_analysis, documents)
    print("\n========== 3. RAG Context ==========")
    print("是否生成成功：是")
    print(f"RAG Context 长度：{len(context)}")
    print(f"RAG Context 预览：{context[:500]}")

    try:
        strategy = generate_marketing_strategy(product_analysis, documents)
    except Exception as exc:
        raise RuntimeError(f"LLM 营销策略生成失败：{exc}") from exc
    if not isinstance(strategy, MarketingStrategy):
        raise TypeError("营销策略生成器未返回 MarketingStrategy")
    print_strategy(strategy)


if __name__ == "__main__":
    main()
