"""Deterministic explanations for already retrieved RAG documents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


class RAGExplanation(BaseModel):
    """Structured explanation for one document in the current RAG result."""

    rank: int
    source: str
    filename: str
    category: str
    match_reason: str
    reference_role: str


_REFERENCE_ROLES = {
    "marketing_rules": "用于参考营销方法、分析框架和策略制定逻辑。",
    "marketing_cases": (
        "用于参考历史营销中的方法、场景拆分和内容组织方式，"
        "不迁移具体商品属性。"
    ),
    "brand": "用于参考品牌定位、品牌消费者和品牌营销原则。",
    "platform_rules": "用于参考营销表达边界、内容规范和风险控制要求。",
    "other": "作为与当前商品相关的补充参考知识。",
}

_TERM_SPLITTER = re.compile(r"[\s，,、；;|/\\]+")


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert a Pydantic model or mapping without changing the source object."""
    if isinstance(value, Mapping):
        return dict(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}

    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        dumped = legacy_dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}

    return {}


def _value_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if item is not None]
    if value is None:
        return []
    return [str(value)]


def _split_terms(value: Any) -> list[str]:
    """Extract readable phrases while preserving their original order."""
    terms: list[str] = []
    for item in _value_items(value):
        text = item.strip()
        if not text:
            continue
        parts = _TERM_SPLITTER.split(text)
        for part in parts:
            term = part.strip(" ：:。.!！?？()（）[]【】\"'")
            if ":" in term or "：" in term:
                term = re.split(r"[：:]", term, maxsplit=1)[-1].strip()
            if len(term) >= 2 and term not in terms:
                terms.append(term)
    return terms


def _field_terms(data: Mapping[str, Any], *field_names: str) -> list[str]:
    terms: list[str] = []
    for field_name in field_names:
        for term in _split_terms(data.get(field_name)):
            if term not in terms:
                terms.append(term)
    return terms


def _matching_terms(terms: list[str], text: str, limit: int = 3) -> list[str]:
    normalized_text = text.casefold()
    matches = [term for term in terms if term.casefold() in normalized_text]
    return matches[:limit]


def _display_terms(terms: list[str]) -> str:
    return "、".join(f"“{term}”" for term in terms)


def _safe_metadata(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _metadata_text(metadata: Mapping[str, Any], key: str, default: str) -> str:
    value = metadata.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _document_content(document: Any) -> str:
    content = getattr(document, "page_content", "")
    return content if isinstance(content, str) else str(content or "")


def _query_terms(query: str) -> list[str]:
    return _split_terms(query if isinstance(query, str) else "")


def _direct_match_description(
    analysis: Mapping[str, Any],
    document_content: str,
) -> str:
    feature_matches = _matching_terms(
        _field_terms(analysis, "confirmed_features", "core_selling_points"),
        document_content,
    )
    if feature_matches:
        return (
            f"该知识涉及当前商品分析中已有的{_display_terms(feature_matches)}，"
            "与已确认卖点或其营销表达直接相关。"
        )

    scenario_matches = _matching_terms(
        _field_terms(analysis, "usage_scenarios"),
        document_content,
    )
    if scenario_matches:
        return (
            f"该知识与当前商品已有的{_display_terms(scenario_matches)}使用场景"
            "存在直接文本关联。"
        )

    product_matches = _matching_terms(
        _field_terms(analysis, "product_name", "product_type"),
        document_content,
    )
    if product_matches:
        return (
            f"该知识提及当前商品名称或类型中的{_display_terms(product_matches)}，"
            "与本次商品分析对象存在直接关联。"
        )

    analysis_matches = _matching_terms(
        _field_terms(
            analysis,
            "target_audience",
            "user_needs",
            "consumer_value",
        ),
        document_content,
    )
    if analysis_matches:
        return (
            f"该知识与当前商品分析中的{_display_terms(analysis_matches)}存在直接文本关联，"
            "可辅助理解用户或消费价值。"
        )

    return ""


def _category_reason(
    category: str,
    analysis: Mapping[str, Any],
    direct_match: str,
    query_matches: list[str],
) -> str:
    if direct_match:
        relevance_evidence = direct_match
    elif query_matches:
        relevance_evidence = (
            f"该知识与本次动态 RAG Query 中的{_display_terms(query_matches)}"
            "存在直接文本关联。"
        )
    else:
        relevance_evidence = (
            "该知识由现有向量检索流程召回，但未发现足以进一步说明的直接文本重合，"
            "因此仅按其知识类别说明参考范围。"
        )

    if category == "marketing_rules":
        return (
            f"{relevance_evidence}该知识属于营销规则，可用于当前商品的营销方法、"
            "分析框架或策略制定参考。"
        )

    if category == "marketing_cases":
        return (
            f"{relevance_evidence}该知识属于历史营销案例，只能参考营销方法、内容结构、"
            "场景拆分方式和用户洞察方法；不能迁移案例中的商品属性、材质、参数、功能、"
            "性能、认证、目标人群或使用场景。"
        )

    if category == "brand":
        brand_name = str(analysis.get("brand_name") or "").strip()
        brand_context = f"当前商品品牌为“{brand_name}”；" if brand_name else ""
        return (
            f"{brand_context}{relevance_evidence}该知识属于品牌资料，可用于约束当前商品的"
            "品牌定位、消费者表达和营销方向。"
        )

    if category == "platform_rules":
        return (
            f"{relevance_evidence}该知识属于平台内容规范或营销风险规则，可用于检查"
            "营销表达边界和宣传风险。"
        )

    return (
        f"{relevance_evidence}该知识未归入已知的主要知识类别，"
        "仅作为本次营销策略生成的补充参考。"
    )


def explain_rag_results(
    product_analysis: Any,
    query: str,
    documents: list[Any],
) -> list[RAGExplanation]:
    """Explain retrieved documents in their existing order without retrieval calls."""
    analysis = _as_mapping(product_analysis)
    query_terms = _query_terms(query)
    explanations: list[RAGExplanation] = []

    for rank, document in enumerate(documents, start=1):
        metadata = _safe_metadata(document)
        source = _metadata_text(metadata, "source", "unknown")
        filename = _metadata_text(metadata, "filename", "unknown")
        category = _metadata_text(metadata, "category", "other")
        content = _document_content(document)

        direct_match = _direct_match_description(analysis, content)
        matched_query_terms = _matching_terms(query_terms, content)
        normalized_category = category.casefold()
        role = _REFERENCE_ROLES.get(normalized_category, _REFERENCE_ROLES["other"])

        explanations.append(
            RAGExplanation(
                rank=rank,
                source=source,
                filename=filename,
                category=category,
                match_reason=_category_reason(
                    normalized_category,
                    analysis,
                    direct_match,
                    matched_query_terms,
                ),
                reference_role=role,
            )
        )

    return explanations
