"""Utilities for assembling the RAG context passed to a later strategy step."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any


def _analysis_as_mapping(product_analysis: Any) -> Mapping[str, Any]:
    """Convert a Pydantic model, mapping, or other value to displayable fields."""
    if product_analysis is None:
        return {"分析结果": "未提供当前商品分析结果。"}

    model_dump = getattr(product_analysis, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped

    if isinstance(product_analysis, Mapping):
        return product_analysis

    return {"分析结果": product_analysis}


def _format_value(value: Any) -> str:
    """Keep structured analysis readable without assuming a specific schema."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return "" if value is None else str(value)
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def build_rag_context(
    product_analysis: Any,
    retrieved_documents: Iterable[Any] | None,
) -> str:
    """Build a clearly separated context from runtime analysis and RAG documents.

    ``product_analysis`` is runtime input and is never persisted. Each retrieved
    LangChain ``Document`` contributes only its source metadata and page content.
    """
    lines = [
        "【当前商品分析】",
        "以下内容是当前用户商品的运行时分析结果，只能作为当前商品事实和分析依据。",
    ]
    for key, value in _analysis_as_mapping(product_analysis).items():
        lines.append(f"{key}：{_format_value(value)}")

    lines.extend(
        [
            "",
            "【RAG参考知识】",
            "以下内容来自知识库的通用方法、品牌资料、历史案例和风险规范，只能作为策略制定参考。",
            "历史案例中的具体商品属性不能直接套用到当前商品；当前商品未提供的功能、参数或效果不得据此新增。",
        ]
    )

    documents = list(retrieved_documents or [])
    if not documents:
        lines.append("未检索到相关知识。")
        return "\n".join(lines)

    for index, document in enumerate(documents, 1):
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or "unknown"
        content = getattr(document, "page_content", "") or ""
        lines.extend(
            [
                "",
                f"【知识{index}】",
                f"source：{source}",
                f"content：{content}",
            ]
        )

    return "\n".join(lines)
