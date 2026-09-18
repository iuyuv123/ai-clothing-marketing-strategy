"""Reusable evaluation metrics for the AI clothing marketing assistant."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


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

POSITIVE_STRATEGY_FIELDS = STRATEGY_FIELDS[:-1]

DEFAULT_UNCONFIRMED_ATTRIBUTE_TERMS = (
    "透气",
    "保暖",
    "耐磨",
    "速干",
    "轻量化",
    "羽绒",
    "锦纶",
    "聚酯纤维",
    "UPF",
    "专利",
    "认证",
    "防水等级",
    "防风等级",
    "高海拔",
    "极端天气",
    "专业探险",
    "极地",
    "全天候",
    "所有户外环境",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        dumped = legacy_dict()
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _non_empty(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_non_empty(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return bool(value) and all(_non_empty(item) for item in value)
    return value is not None and bool(str(value).strip())


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return "；".join(_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return "；".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value)).lower()


def _terms(value: Any) -> set[str]:
    return {
        term.strip()
        for term in re.split(r"[；;，,。！？!?、/|\s]+", _text(value))
        if len(term.strip()) >= 2
    }


def _contains_all(actual: Any, expected: Any) -> bool:
    actual_text = _normalize(actual)
    return all(_normalize(term) in actual_text for term in _terms(expected))


def _has_text_overlap(left: Any, right: Any) -> bool:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if left_terms & right_terms:
        return True
    left_text = _normalize(left)
    right_text = _normalize(right)
    return any(term in right_text for term in left_terms) or any(
        term in left_text for term in right_terms
    )


def evaluate_product_understanding(
    product_analysis: Any,
    *,
    product_name: str,
    product_type: str,
    confirmed_features: str | Iterable[str],
    usage_scenarios: str | Iterable[str],
) -> dict[str, Any]:
    """检查商品分析是否忠实保留运行时输入，不补充或改写输入事实。"""
    analysis = _as_mapping(product_analysis)
    checks = {
        "product_name_preserved": _normalize(analysis.get("product_name"))
        == _normalize(product_name),
        "product_type_preserved": _normalize(analysis.get("product_type"))
        == _normalize(product_type),
        "confirmed_features_preserved": _contains_all(
            analysis.get("confirmed_features", []), confirmed_features
        ),
        "usage_scenarios_preserved": _contains_all(
            analysis.get("usage_scenarios", []), usage_scenarios
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_strategy_relevance(
    product_analysis: Any,
    strategy: Any,
) -> dict[str, Any]:
    """用结构化字段之间的文本关联判断策略是否围绕当前商品生成。"""
    analysis = _as_mapping(product_analysis)
    strategy_data = _as_mapping(strategy)
    positioning_basis = [
        analysis.get("product_name"),
        analysis.get("product_type"),
        analysis.get("confirmed_features"),
    ]
    checks = {
        "product_positioning": _has_text_overlap(
            strategy_data.get("product_positioning"), positioning_basis
        ),
        "core_selling_points": _has_text_overlap(
            strategy_data.get("core_selling_points"),
            [analysis.get("confirmed_features"), analysis.get("core_selling_points")],
        ),
        "target_audience": _has_text_overlap(
            strategy_data.get("target_audience"), analysis.get("target_audience")
        ),
        "usage_scenarios": _has_text_overlap(
            strategy_data.get("usage_scenarios"), analysis.get("usage_scenarios")
        ),
        "user_needs": _has_text_overlap(
            strategy_data.get("user_needs"), analysis.get("user_needs")
        ),
        "core_marketing_direction": _has_text_overlap(
            strategy_data.get("core_marketing_direction"), list(analysis.values())
        ),
    }
    return {
        "passed": all(checks.values()),
        "matched_fields": sum(checks.values()),
        "total_fields": len(checks),
        "checks": checks,
    }


def evaluate_strategy_completeness(strategy: Any) -> dict[str, Any]:
    """检查 MarketingStrategy 的 11 个字段是否存在且非空。"""
    strategy_data = _as_mapping(strategy)
    checks = {
        field: field in strategy_data and _non_empty(strategy_data.get(field))
        for field in STRATEGY_FIELDS
    }
    return {
        "passed": all(checks.values()),
        "completed_fields": sum(checks.values()),
        "total_fields": len(checks),
        "checks": checks,
    }


def evaluate_fact_boundary(
    strategy: Any,
    confirmed_features: str | Iterable[str],
    prohibited_terms: Iterable[str] = DEFAULT_UNCONFIRMED_ATTRIBUTE_TERMS,
) -> dict[str, Any]:
    """检查正向策略字段中的未确认属性；risk_control 明确排除在外。"""
    strategy_data = _as_mapping(strategy)
    positive_content = {
        field: strategy_data.get(field) for field in POSITIVE_STRATEGY_FIELDS
    }
    positive_text = _normalize(positive_content)
    confirmed_text = _normalize(confirmed_features)
    hits = sorted(
        {
            term
            for term in prohibited_terms
            if _normalize(term) in positive_text and _normalize(term) not in confirmed_text
        }
    )
    return {
        "passed": not hits,
        "checked_fields": list(POSITIVE_STRATEGY_FIELDS),
        "excluded_fields": ["risk_control"],
        "matched_unconfirmed_terms": hits,
    }


def _character_ngrams(value: Any, size: int = 4) -> set[str]:
    segments = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", _text(value).lower())
    return {
        segment[index : index + size]
        for segment in segments
        for index in range(len(segment) - size + 1)
    }


def evaluate_rag_effectiveness(
    without_rag_strategy: Any,
    with_rag_strategy: Any,
    retrieved_documents: Iterable[Any] | None,
) -> dict[str, Any]:
    """比较 A/B 策略，并用实际检索内容的文本重合提供 RAG 参与证据。"""
    without_rag = _as_mapping(without_rag_strategy)
    with_rag = _as_mapping(with_rag_strategy)
    field_differences = {
        field: without_rag.get(field) != with_rag.get(field)
        for field in STRATEGY_FIELDS
    }

    documents = list(retrieved_documents or [])
    valid_documents = [
        document
        for document in documents
        if _non_empty(getattr(document, "page_content", ""))
    ]
    sources = sorted(
        {
            str((getattr(document, "metadata", {}) or {}).get("source"))
            for document in valid_documents
            if (getattr(document, "metadata", {}) or {}).get("source")
        }
    )
    knowledge_ngrams = _character_ngrams(
        [getattr(document, "page_content", "") for document in valid_documents]
    )
    strategy_ngrams = _character_ngrams(
        {field: with_rag.get(field) for field in POSITIVE_STRATEGY_FIELDS}
    )
    overlap = sorted(knowledge_ngrams & strategy_ngrams)

    checks = {
        "strategies_differ": any(field_differences.values()),
        "retrieval_non_empty": bool(documents),
        "retrieved_documents_valid": len(valid_documents) == len(documents)
        and bool(valid_documents),
        "retrieved_knowledge_used": bool(overlap),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "different_fields": [
            field for field, different in field_differences.items() if different
        ],
        "retrieved_document_count": len(documents),
        "retrieved_sources": sources,
        "knowledge_overlap_evidence": overlap[:20],
    }
