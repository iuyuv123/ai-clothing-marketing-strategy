"""Compare structured strategy generation with and without retrieved context."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import dashscope
from dotenv import load_dotenv
from langchain_chroma import Chroma

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    _build_llm,
    generate_marketing_strategy,
)
from product_analyzer import ProductAnalysis, parse_response  # noqa: E402
from rag_context import build_rag_context  # noqa: E402
from vector_store import DashScopeEmbeddings  # noqa: E402


PROJECT_ROOT = APP_DIR.parent
PERSIST_DIR = PROJECT_ROOT / "vector_store" / "marketing_strategy_kb"
COLLECTION_NAME = "marketing_strategy_kb"
RETRIEVAL_TOP_K = 3


PRODUCT = {
    "品牌名称": "AAA",
    "商品类型": "户外三合一冲锋衣",
    "商品卖点": "三合一设计、防风、防水",
    "目标人群": "待根据当前商品动态分析",
    "营销场景": "登山、徒步、户外运动",
    "用户需求": "待根据商品特点和使用场景分析",
}

RETRIEVAL_QUERIES = (
    "户外服装适合哪些使用场景",
    "如何根据商品特点分析用户需求",
    "如何制定服装营销方案",
)

COMPARISON_FIELDS = (
    ("产品定位", "product_positioning"),
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


def _without_rag_prompt(product_analysis: ProductAnalysis) -> str:
    analysis = product_analysis.model_dump_json(ensure_ascii=False)
    return f"""你正在为“AI 服装营销策略助手”生成结构化营销策略。

以下只有【当前商品分析】，它是用户本次输入商品的实时信息，具有最高优先级，只能作为当前商品事实和分析依据。
不得补充当前商品未提供的材质、参数、功能、数据、认证、专利、性能或效果。
target_audience 必须根据当前商品动态分析，不能复制任何历史案例人群。

只返回合法 JSON，禁止 Markdown、```json、解释文字或前后额外说明。严格包含以下字段：
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
{analysis}
"""


def _analyze_product(product: dict[str, str]) -> ProductAnalysis:
    """Create the shared runtime product analysis without the legacy workflow."""
    prompt = f"""你是商品分析助手。请只根据用户提供的当前商品信息进行分析，不得编造商品属性。
品牌名称：{product['品牌名称']}
商品类型：{product['商品类型']}
商品卖点：{product['商品卖点']}
使用场景：{product['营销场景']}

目标人群需要根据当前商品特点和使用场景动态分析，不要复制历史案例中的人群。
不得补充材质、参数、重量、性能、数据、认证、专利或未提供的功能。
只输出合法 JSON，不要 Markdown 或额外解释，格式为：
{{"brand_name":"品牌名称","product_type":"商品类型","core_selling_points":["..."],"consumer_benefits":["..."],"target_audience":["..."],"marketing_scenarios":["..."],"marketing_directions":["..."]}}"""
    response = _build_llm().invoke(prompt)
    return parse_response(str(response.content))


def _embedding_model() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    if not model:
        raise RuntimeError("DASHSCOPE_EMBEDDING_MODEL 未设置")
    dashscope.api_key = api_key
    return model


def _build_retriever() -> Any:
    if not PERSIST_DIR.is_dir():
        raise FileNotFoundError(f"正式向量库目录不存在：{PERSIST_DIR}")

    model = _embedding_model()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=DashScopeEmbeddings(model=model),
    )
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_TOP_K},
    )


def _retrieve_documents(product: dict[str, str]) -> list[Any]:
    retriever = _build_retriever()
    product_facts = (
        f"商品类型：{product['商品类型']}；"
        f"商品卖点：{product['商品卖点']}；"
        f"使用场景：{product['营销场景']}"
    )
    documents: list[Any] = []
    for topic in RETRIEVAL_QUERIES:
        query = f"{product_facts}；检索主题：{topic}"
        documents.extend(list(retriever.invoke(query)))
    return documents


def generate_without_rag(product_analysis: ProductAnalysis) -> MarketingStrategy:
    """Generate a strategy from the same analysis without retrieval/context."""
    response = _build_llm().invoke(_without_rag_prompt(product_analysis))
    try:
        data = json.loads(response.content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("无 RAG 方案模型返回结果不是合法 JSON") from exc
    return MarketingStrategy.model_validate(data)


def _display_value(value: Any) -> str:
    return value if isinstance(value, str) else "；".join(str(item) for item in value)


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
        print(f"{label}：{_display_value(getattr(strategy, field_name))}")


def _print_documents(documents: list[Any]) -> None:
    print(f"检索结果数量：{len(documents)}")
    for index, document in enumerate(documents, 1):
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        content = (getattr(document, "page_content", "") or "").strip()
        print(f"\n知识{index}：")
        print(f"来源：{source}")
        print(f"内容摘要：{content[:200]}")


def run_comparison() -> int:
    product_analysis = _analyze_product(PRODUCT)
    if not isinstance(product_analysis, ProductAnalysis):
        raise TypeError("商品分析结果不是 ProductAnalysis")

    without_rag = generate_without_rag(product_analysis)
    if not isinstance(without_rag, MarketingStrategy):
        raise TypeError("无 RAG 方案未返回 MarketingStrategy")

    retrieved_documents = _retrieve_documents(PRODUCT)

    print("\n========== RAG 检索知识 ==========")
    _print_documents(retrieved_documents)
    if not retrieved_documents:
        raise ValueError("RAG 检索未返回任何知识，无法进行 RAG 效果对比实验")

    rag_context = build_rag_context(product_analysis, retrieved_documents)
    print("\n========== RAG Context ==========")
    print("是否生成 RAG Context：是")
    print(f"RAG Context 长度：{len(rag_context)}")
    print(rag_context)

    # The existing generator rebuilds this same context from these two inputs
    # before composing its RAG prompt.
    with_rag = generate_marketing_strategy(product_analysis, retrieved_documents)
    if not isinstance(with_rag, MarketingStrategy):
        raise TypeError("RAG 方案未返回 MarketingStrategy")

    print("========== RAG 效果对比实验 ==========")
    print("\n【当前商品】")
    print(f"商品名称：{PRODUCT['商品类型']}")
    print(f"商品卖点：{PRODUCT['商品卖点']}")
    print(f"使用场景：{PRODUCT['营销场景']}")

    print("\n========== 方案 A：无 RAG ==========")
    _print_strategy(without_rag)
    print("\n========== 方案 B：RAG + LLM ==========")
    _print_strategy(with_rag)

    print("\n========== 结果差异 ==========")
    for label, field_name in COMPARISON_FIELDS:
        print(f"\n{label}：")
        print(f"无 RAG：{_display_value(getattr(without_rag, field_name))}")
        print(f"RAG：{_display_value(getattr(with_rag, field_name))}")

    print("\n========== RAG 价值观察 ==========")
    print("请根据两个结果重点观察：")
    for item in (
        "RAG 是否增加了新的场景洞察？",
        "RAG 是否增加了新的用户需求？",
        "RAG 是否让营销方向更加具体？",
        "RAG 是否改善渠道策略？",
        "RAG 是否改善内容方向？",
        "RAG 是否增加了合规风险控制？",
        "RAG 是否出现了错误迁移历史案例信息的问题？",
    ):
        print(item)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_comparison())
    except Exception as exc:
        print("========== 实验失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        raise
