"""Build deterministic reference traces for an existing marketing strategy."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel


class StrategyTrace(BaseModel):
    """Reference material that may support one existing strategy field."""

    strategy_field: str
    strategy_title: str
    strategy_content: str
    product_facts: list[str]
    usage_scenarios: list[str]
    rag_references: list[str]
    reference_notes: list[str]


# Dictionary insertion order is the required MarketingStrategy display order.
STRATEGY_FIELD_TITLES: dict[str, str] = {
    "product_positioning": "商品定位",
    "core_selling_points": "核心卖点",
    "target_audience": "目标人群",
    "usage_scenarios": "使用场景",
    "user_needs": "用户需求",
    "consumer_value": "消费者价值",
    "core_marketing_direction": "核心营销方向",
    "channel_strategy": "渠道策略",
    "content_direction": "内容方向",
    "execution_suggestions": "执行建议",
    "risk_control": "风险控制",
}


# These rules mean only "may serve as a reference". They do not represent a
# causal link between a retrieved document and generated strategy text.
RAG_REFERENCE_FIELDS: dict[str, frozenset[str]] = {
    "marketing_rules": frozenset(
        {
            "product_positioning",
            "target_audience",
            "usage_scenarios",
            "user_needs",
            "consumer_value",
            "core_marketing_direction",
            "channel_strategy",
            "content_direction",
            "execution_suggestions",
        }
    ),
    "marketing_cases": frozenset(
        {
            "core_marketing_direction",
            "content_direction",
            "execution_suggestions",
        }
    ),
    "brand": frozenset(
        {
            "product_positioning",
            "target_audience",
            "core_marketing_direction",
            "content_direction",
        }
    ),
    "platform_rules": frozenset(
        {
            "content_direction",
            "execution_suggestions",
            "risk_control",
        }
    ),
}


_REFERENCE_NOTES: dict[str, str] = {
    "marketing_rules": (
        "该知识作为营销方法和分析框架参考，用于辅助当前商品的策略组织。"
    ),
    "marketing_cases": (
        "该知识仅用于参考历史营销方法、场景拆分和内容组织方式，"
        "不迁移案例中的具体商品属性。"
    ),
    "brand": "该知识用于参考品牌定位、品牌消费者和品牌营销原则。",
    "platform_rules": "该知识用于参考内容规范、宣传边界和风险控制要求。",
}


# 使用场景可作为这些策略环节的已确认上下文，但不能反向证明商品能力。
_SCENARIO_REFERENCE_FIELDS = frozenset(
    {
        "product_positioning",
        "target_audience",
        "usage_scenarios",
        "user_needs",
        "consumer_value",
        "core_marketing_direction",
        "channel_strategy",
        "content_direction",
        "execution_suggestions",
        "risk_control",
    }
)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Read a Pydantic model or mapping without mutating the original value."""
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


def _string_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[str] = []
        for item in value:
            text = str(item).strip() if item is not None else ""
            if text:
                items.append(text)
        return items

    text = str(value).strip()
    return [text] if text else []


def _unique(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _product_facts(product_analysis: Mapping[str, Any]) -> list[str]:
    """Return facts exclusively from the current ProductAnalysis."""
    facts: list[str] = []
    for field_name in (
        "product_name",
        "product_type",
        "confirmed_features",
        "core_selling_points",
    ):
        facts.extend(_string_items(product_analysis.get(field_name)))
    return _unique(facts)


def _strategy_content(value: Any) -> str:
    """Serialize existing content for display without generating new wording."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(str(item) for item in value if item is not None)
    return str(value)


def _rag_references_for_field(
    strategy_field: str,
    rag_explanations: Sequence[Any],
) -> tuple[list[str], list[str]]:
    references: list[str] = []
    notes: list[str] = []
    seen_references: set[str] = set()

    for explanation in rag_explanations:
        data = _as_mapping(explanation)
        category = str(data.get("category") or "other").strip().casefold()
        if strategy_field not in RAG_REFERENCE_FIELDS.get(category, frozenset()):
            continue

        filename = str(data.get("filename") or "").strip()
        source = str(data.get("source") or "").strip()
        reference_name = filename or source or "unknown"
        if reference_name in seen_references:
            continue

        seen_references.add(reference_name)
        references.append(reference_name)
        note = _REFERENCE_NOTES.get(category)
        if note:
            notes.append(f"{reference_name}：{note}")

    return references, notes


def build_strategy_traceability(
    product_analysis: Any,
    strategy: Any,
    rag_explanations: list[Any] | None,
) -> list[StrategyTrace]:
    """Organize existing facts and references without claiming causal influence."""
    analysis_data = _as_mapping(product_analysis)
    strategy_data = _as_mapping(strategy)
    product_facts = _product_facts(analysis_data)
    usage_scenarios = _unique(_string_items(analysis_data.get("usage_scenarios")))
    explanations = rag_explanations or []

    traces: list[StrategyTrace] = []
    for strategy_field, strategy_title in STRATEGY_FIELD_TITLES.items():
        rag_references, reference_notes = _rag_references_for_field(
            strategy_field,
            explanations,
        )
        traces.append(
            StrategyTrace(
                strategy_field=strategy_field,
                strategy_title=strategy_title,
                strategy_content=_strategy_content(strategy_data.get(strategy_field)),
                product_facts=list(product_facts),
                usage_scenarios=(
                    list(usage_scenarios)
                    if strategy_field in _SCENARIO_REFERENCE_FIELDS
                    else []
                ),
                rag_references=rag_references,
                reference_notes=reference_notes,
            )
        )

    return traces
