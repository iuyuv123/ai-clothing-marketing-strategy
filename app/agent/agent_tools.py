"""Five business tools used by :class:`EcommerceAgent`.

The LangChain tools use explicit schemas and return serializable dictionaries.
Small state adapters remain for backward compatibility; the orchestrator is
responsible for writing Tool results into ``AgentState``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.llm_factory import create_chat_model
from app.marketing_strategy_generator import MarketingStrategy, generate_marketing_strategy
from app.product_analyzer import ProductAnalysis, analyze_product as analyze_product_core
from app.vector_store import retrieve_marketing_knowledge
from app.embedding_factory import create_embedding_model
from app.runtime_config import get_runtime_config
from langchain_chroma import Chroma


class XiaohongshuContent(BaseModel):
    """MVP content contract for the Xiaohongshu platform."""

    title: str = Field(min_length=1)
    body: str = Field(min_length=1)
    tags: list[str] = Field(min_length=1)
    call_to_action: str = Field(min_length=1)


class ContentReview(BaseModel):
    """Auditable result of the three review dimensions."""

    passed: bool
    fact_consistency: bool
    compliance: bool
    task_match: bool
    risk_items: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    checked_rule_categories: list[str] = Field(default_factory=list)


class KnowledgeRetrievalInput(BaseModel):
    """Explicit input contract for the RAG retrieval Tool."""

    query: str = Field(min_length=1, description="用于检索营销和平台知识的查询")
    top_k: int = Field(default=5, ge=1, le=20, description="最终返回的文档数量")
    candidate_k: int = Field(default=10, ge=1, le=50, description="候选文档数量")
    decision_reason: str = Field(min_length=3, description="当前任务为什么需要知识检索")


class MarketingStrategyInput(BaseModel):
    """Explicit input contract for the marketing-strategy Tool."""

    product_analysis: dict[str, Any] = Field(description="已完成的结构化商品分析")
    retrieved_knowledge: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Agent 已检索并传入的知识文档",
    )
    marketing_goal: str = Field(default="推广", description="用户营销目标")
    decision_reason: str = Field(min_length=3, description="当前任务为什么需要营销策略")


class ContentGenerationInput(BaseModel):
    """Explicit input contract for the content-generation Tool."""

    product_analysis: dict[str, Any] = Field(description="已完成的结构化商品分析")
    marketing_strategy: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent 已生成的营销策略",
    )
    retrieved_knowledge: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Agent 已检索并传入的平台/营销参考知识",
    )
    platform: str = Field(default="小红书", description="发布平台")
    marketing_goal: str = Field(default="推广", description="营销目标")
    correction_notes: list[str] = Field(default_factory=list, description="审核后的修正要求")
    decision_reason: str = Field(min_length=3, description="当前任务为什么需要内容生成")


class ContentReviewInput(BaseModel):
    """Explicit input contract for the content-review Tool."""

    content: dict[str, Any] | str = Field(description="待审核的结构化内容或正文")
    product_analysis: dict[str, Any] = Field(
        default_factory=dict,
        description="当前商品分析；用于读取已确认事实和场景",
    )
    confirmed_features: str = Field(default="", description="已确认商品事实")
    usage_scenarios: str = Field(default="", description="已确认使用场景")
    task_type: str = Field(default="promotion_plan", description="当前任务类型")
    retrieved_knowledge: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Agent 已传入的平台规则和营销规范参考",
    )
    decision_reason: str = Field(min_length=3, description="当前任务为什么需要内容审核")


class AnalyzeProductInput(BaseModel):
    """Explicit input contract for the standalone product-analysis tool."""

    product_name: str = Field(min_length=1, description="商品名称")
    product_info: str = Field(
        default="",
        description="商品详细信息；未单独提供卖点时也作为已确认商品事实",
    )
    product_selling_points: str = Field(
        default="",
        description="已由用户确认的商品卖点或商品事实",
    )
    brand_name: str = Field(default="AAA", description="品牌名称")
    product_type: str = Field(
        default="",
        description="商品类型；为空时使用商品名称作为保守默认值",
    )
    usage_scenarios: str = Field(default="", description="用户已确认的使用场景")
    decision_reason: str = Field(
        min_length=3,
        description="当前任务为什么需要商品分析",
    )

    @model_validator(mode="after")
    def require_product_facts(self) -> "AnalyzeProductInput":
        if not self.product_info.strip() and not self.product_selling_points.strip():
            raise ValueError("product_info 和 product_selling_points 至少提供一项")
        return self


UNCONFIRMED_ATTRIBUTE_TERMS = (
    "保暖", "透气", "轻量", "轻量化", "耐磨", "速干", "羽绒", "锦纶", "聚酯纤维",
    "防水等级", "防风等级", "高海拔", "极端天气", "专业探险", "极地", "全天候",
    "防护等级", "测试数据", "国家标准", "工艺参数", "认证", "专利", "面料", "重量",
)
RISK_TERMS = ("最好", "最强", "第一", "100%", "永远", "绝对")

TOOL_CONTRACTS: dict[str, dict[str, Any]] = {
    "analyze_product": {"reads": ["user_request", "product_info"], "writes": ["product_analysis", "target_user", "platform", "marketing_goal"]},
    "knowledge_retrieval_tool": {"reads": ["query"], "writes": ["retrieved_knowledge"]},
    "marketing_strategy_tool": {"reads": ["product_analysis", "marketing_goal", "retrieved_knowledge"], "writes": ["marketing_strategy"]},
    "content_generation_tool": {"reads": ["product_analysis", "platform", "marketing_goal", "retrieved_knowledge", "marketing_strategy", "review_result"], "writes": ["generated_content"]},
    "content_review_tool": {"reads": ["product_analysis", "retrieved_knowledge", "generated_content"], "writes": ["review_result"]},
}


def _field(data: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = data.get(name)
        if value is not None:
            return str(value).strip()
    return default


def _items(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[，,、；;\n]+", str(value or "")) if part.strip()]


def _json_response(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("模型返回不是合法 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("模型返回必须是 JSON 对象")
    return value


def _document_result(document: Any) -> dict[str, Any]:
    metadata = getattr(document, "metadata", {}) or {}
    source = str(metadata.get("source") or "")
    return {
        "source": source,
        "filename": str(metadata.get("filename") or source.replace("\\", "/").rsplit("/", 1)[-1]),
        "category": str(metadata.get("category") or ""),
        "content": str(getattr(document, "page_content", "") or ""),
        "metadata": dict(metadata),
    }


def _document_from_result(value: Any) -> Document:
    if isinstance(value, Document):
        return value
    if not isinstance(value, dict):
        raise TypeError("retrieved_knowledge 的每一项必须是文档字典")
    metadata = dict(value.get("metadata") or {})
    for key in ("source", "filename", "category"):
        if value.get(key) and not metadata.get(key):
            metadata[key] = value[key]
    return Document(
        page_content=str(value.get("content", value.get("page_content", "")) or ""),
        metadata=metadata,
    )


def _documents_from_results(values: Iterable[Any] | None) -> list[Document]:
    return [_document_from_result(value) for value in (values or [])]


def _analysis_from_result(value: Any) -> ProductAnalysis:
    if isinstance(value, ProductAnalysis):
        return value
    if not isinstance(value, dict):
        raise TypeError("product_analysis 必须是结构化字典")
    normalized = dict(value)
    normalized.setdefault("consumer_value", normalized.get("consumer_benefits", []))
    normalized.setdefault("usage_scenarios", normalized.get("marketing_scenarios", []))
    return ProductAnalysis.model_validate(normalized)


def _run_product_analysis(
    *,
    product_name: str,
    product_info: str,
    product_selling_points: str,
    brand_name: str,
    product_type: str,
    usage_scenarios: str,
) -> ProductAnalysis:
    confirmed_features = product_selling_points.strip() or product_info.strip()
    product_details = product_info.strip() if product_selling_points.strip() else ""
    return analyze_product_core(
        brand_name=brand_name.strip() or "AAA",
        product_name=product_name.strip(),
        product_type=product_type.strip() or product_name.strip(),
        confirmed_features=confirmed_features,
        usage_scenarios=usage_scenarios.strip(),
        product_details=product_details,
    )


def _product_analysis_result(analysis: ProductAnalysis) -> dict[str, Any]:
    data = analysis.model_dump()
    return {
        **data,
        "consumer_benefits": list(analysis.consumer_value),
        "marketing_scenarios": list(analysis.usage_scenarios),
        "marketing_directions": [
            f"围绕“{selling_point}”进行事实型卖点表达"
            for selling_point in analysis.core_selling_points
        ],
    }


@tool("analyze_product", args_schema=AnalyzeProductInput)
def analyze_product(
    product_name: str,
    product_info: str = "",
    product_selling_points: str = "",
    brand_name: str = "AAA",
    product_type: str = "",
    usage_scenarios: str = "",
    decision_reason: str = "",
) -> dict[str, Any]:
    """Analyze confirmed product facts and return structured marketing elements."""
    analysis = _run_product_analysis(
        product_name=product_name,
        product_info=product_info,
        product_selling_points=product_selling_points,
        brand_name=brand_name,
        product_type=product_type,
        usage_scenarios=usage_scenarios,
    )
    return _product_analysis_result(analysis)


def product_analysis_tool(state: Any) -> dict[str, Any]:
    """Backward-compatible state adapter for the product-analysis capability."""
    product = state.product_info
    selling_points = _field(
        product,
        "confirmed_features",
        "已确认商品事实",
        "已确认商品卖点",
    )
    product_info = _field(product, "product_details", "商品详细资料")
    analysis = _run_product_analysis(
        brand_name=_field(product, "brand_name", "品牌名称", default="AAA"),
        product_name=_field(product, "product_name", "商品名称"),
        product_type=_field(product, "product_type", "商品类型"),
        product_selling_points=selling_points,
        product_info=product_info,
        usage_scenarios=_field(product, "usage_scenarios", "使用场景"),
    )
    return {
        "product_analysis": analysis,
        "target_user": list(analysis.target_audience),
        "platform": _field(product, "platform", "平台", default="小红书"),
        "marketing_goal": _field(product, "marketing_goal", "运营目标", default="推广"),
    }


def _build_query(analysis: ProductAnalysis, task: str) -> str:
    values: list[str] = [analysis.product_name, analysis.product_type, task]
    for field in (
        "confirmed_features", "core_selling_points", "target_audience", "usage_scenarios",
        "user_needs", "consumer_value",
    ):
        values.extend(getattr(analysis, field, []))
    values.extend(["营销策略", "用户需求", "场景营销", "小红书", "平台规则"])
    return "；".join(value.strip() for value in values if str(value).strip())


def _retrieve_knowledge(query: str, top_k: int, candidate_k: int) -> list[Document]:
    config = get_runtime_config()
    if not config.chroma_persist_dir.is_dir():
        raise FileNotFoundError("正式营销策略知识库不存在，请先构建 Chroma")
    store = Chroma(
        collection_name=config.chroma_collection,
        persist_directory=str(config.chroma_persist_dir),
        embedding_function=create_embedding_model(),
    )
    return retrieve_marketing_knowledge(
        store,
        query.strip(),
        top_k=top_k,
        candidate_k=candidate_k,
    )


@tool("knowledge_retrieval_tool", args_schema=KnowledgeRetrievalInput)
def knowledge_retrieval_tool(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    decision_reason: str = "",
) -> dict[str, Any]:
    """Retrieve marketing knowledge from Chroma and return serializable documents."""
    documents = _retrieve_knowledge(query, top_k, candidate_k)
    return {
        "query": query.strip(),
        "documents": [_document_result(document) for document in documents],
        "candidate_k": candidate_k,
        "top_k": top_k,
        "retrieval_scope": "营销方法、案例与规则参考；不作为当前商品事实来源",
    }


def knowledge_retrieval_for_state(state: Any) -> dict[str, Any]:
    """Backward-compatible AgentState adapter for RAG retrieval."""
    analysis = state.product_analysis
    if isinstance(analysis, ProductAnalysis):
        query = _build_query(analysis, state.task)
    else:
        # Review-only tasks intentionally skip product analysis.  Their query
        # is still grounded in the caller's explicit product facts.
        product = state.product_info
        query = "；".join(
            value for value in (
                _field(product, "product_name", "商品名称"),
                _field(product, "product_type", "商品类型"),
                _field(product, "confirmed_features", "已确认商品事实", "已确认商品卖点"),
                _field(product, "usage_scenarios", "使用场景"),
                state.task,
                "平台规则", "营销风险表达",
            ) if value
        )
    result = knowledge_retrieval_tool.invoke({
        "query": query,
        "top_k": 5,
        "candidate_k": 10,
        "decision_reason": "兼容现有 AgentState 检索调用",
    })
    return {
        "query": result["query"],
        "retrieved_knowledge": result["documents"],
        "candidate_k": result["candidate_k"],
        "top_k": result["top_k"],
    }


def _strategy_result(strategy: MarketingStrategy) -> dict[str, Any]:
    data = strategy.model_dump()
    return {
        **data,
        "marketing_scenarios": list(strategy.usage_scenarios),
        "marketing_directions": list(strategy.core_marketing_direction),
        "content_strategy": list(strategy.content_direction),
        "topic_directions": list(strategy.content_direction),
    }


@tool("marketing_strategy_tool", args_schema=MarketingStrategyInput)
def marketing_strategy_tool(
    product_analysis: dict[str, Any],
    retrieved_knowledge: list[dict[str, Any]] | None = None,
    marketing_goal: str = "推广",
    decision_reason: str = "",
) -> dict[str, Any]:
    """Generate a strategy from Agent-provided analysis and retrieved knowledge."""
    analysis = _analysis_from_result(product_analysis)
    documents = _documents_from_results(retrieved_knowledge)
    strategy = generate_marketing_strategy(analysis, documents)
    return {**_strategy_result(strategy), "marketing_goal": marketing_goal}


def marketing_strategy_for_state(state: Any) -> dict[str, Any]:
    """Backward-compatible AgentState adapter for strategy generation."""
    analysis = state.product_analysis
    if not isinstance(analysis, ProductAnalysis):
        raise ValueError("营销策略前必须先完成商品分析")
    strategy = generate_marketing_strategy(
        analysis,
        _documents_from_results(state.retrieved_knowledge),
    )
    return {"marketing_strategy": strategy}


def _content_prompt(
    analysis: ProductAnalysis,
    strategy: MarketingStrategy | None,
    documents: Iterable[Document],
    platform: str,
    marketing_goal: str,
    correction_notes: Iterable[str] = (),
) -> str:
    strategy_data = strategy.model_dump() if strategy is not None else {}
    knowledge = "\n".join(
        f"- {document.page_content[:800]}" for document in documents
    ) or "未检索到知识"
    corrections = "；".join(str(item) for item in correction_notes if str(item).strip()) or "无"
    return f"""你是小红书内容生成工具。只输出合法 JSON，不要 Markdown。
字段必须是 title、body、tags、call_to_action，且全部非空。
当前商品事实（唯一事实来源）：{json.dumps(analysis.model_dump(), ensure_ascii=False)}
营销策略（仅作方向）：{json.dumps(strategy_data, ensure_ascii=False)}
平台：{platform or '小红书'}；运营目标：{marketing_goal or '推广'}
RAG 参考规则（只用于平台表达和方法，不得新增商品属性）：{knowledge}
禁止把未确认属性写成事实，尤其是保暖、透气、轻量、耐磨、速干、等级、参数或认证。
禁止使用最好、最强、第一、100%、永远、绝对等绝对化表达。只围绕【当前商品事实】中明确提供的
卖点和使用场景生成，不得扩展为更严苛的环境、未确认的性能或专业能力。
上一次审核修正要求：{corrections}
"""


def _generate_content(
    analysis: ProductAnalysis,
    strategy: MarketingStrategy | None,
    documents: list[Document],
    platform: str,
    marketing_goal: str,
    correction_notes: Iterable[str] = (),
) -> XiaohongshuContent:
    response = create_chat_model().invoke(_content_prompt(
        analysis,
        strategy,
        documents,
        platform,
        marketing_goal,
        correction_notes,
    ))
    try:
        return XiaohongshuContent.model_validate(_json_response(response.content))
    except (ValidationError, ValueError) as exc:
        raise ValueError(f"小红书内容生成结果无效：{exc}") from exc


def _content_result(content: XiaohongshuContent) -> dict[str, Any]:
    return {
        **content.model_dump(),
        "content": content.body,
        "hashtags": list(content.tags),
    }


@tool("content_generation_tool", args_schema=ContentGenerationInput)
def content_generation_tool(
    product_analysis: dict[str, Any],
    marketing_strategy: dict[str, Any] | None = None,
    retrieved_knowledge: list[dict[str, Any]] | None = None,
    platform: str = "小红书",
    marketing_goal: str = "推广",
    correction_notes: list[str] | None = None,
    decision_reason: str = "",
) -> dict[str, Any]:
    """Generate platform content only from Agent-provided state and knowledge."""
    analysis = _analysis_from_result(product_analysis)
    strategy = (
        MarketingStrategy.model_validate(marketing_strategy)
        if marketing_strategy
        else None
    )
    content = _generate_content(
        analysis,
        strategy,
        _documents_from_results(retrieved_knowledge),
        platform,
        marketing_goal,
        correction_notes or [],
    )
    return _content_result(content)


def content_generation_for_state(state: Any) -> dict[str, Any]:
    """Backward-compatible AgentState adapter for content generation."""
    analysis = state.product_analysis
    if not isinstance(analysis, ProductAnalysis):
        raise ValueError("内容生成前必须先完成商品分析")
    strategy = state.marketing_strategy
    if strategy is not None and not isinstance(strategy, MarketingStrategy):
        strategy = MarketingStrategy.model_validate(strategy)
    content = _generate_content(
        analysis,
        strategy,
        _documents_from_results(state.retrieved_knowledge),
        state.platform,
        state.marketing_goal,
        state.correction_notes,
    )
    return {"generated_content": content}


def _review_content(
    content: Any,
    *,
    confirmed_features: str,
    usage_scenarios: str,
    task_type: str,
    retrieved_knowledge: Iterable[Any] = (),
) -> ContentReview:
    if isinstance(content, BaseModel):
        content_data = content.model_dump()
    elif isinstance(content, dict):
        content_data = content
    elif isinstance(content, str) and content.strip():
        content_data = {"body": content.strip()}
    else:
        raise ValueError("内容审核前没有可审核内容")
    text = "；".join(str(value) for value in content_data.values())
    required_facts = _items(confirmed_features)
    required_scenarios = _items(usage_scenarios)
    unsupported = [
        term for term in UNCONFIRMED_ATTRIBUTE_TERMS
        if term in text and term not in confirmed_features
    ]
    risks = [term for term in RISK_TERMS if term in text]
    fact_consistency = not unsupported
    compliance = not risks
    is_review_only = str(task_type) == "content_review"
    task_match = bool(content_data.get("body")) if is_review_only else bool(
        content_data.get("title") and content_data.get("body") and content_data.get("tags")
    )
    if required_facts and not is_review_only:
        # At least one confirmed feature must appear in the body/title, while
        # absence of a feature is acceptable when the task is a review-only task.
        task_match = task_match and any(fact in text for fact in required_facts)
    if required_scenarios and not is_review_only:
        task_match = task_match and any(scenario in text for scenario in required_scenarios)
    suggestions: list[str] = []
    if unsupported:
        suggestions.append(f"删除未确认商品信息：{'、'.join(unsupported)}")
    if risks:
        suggestions.append(f"改写绝对化表达：{'、'.join(risks)}")
    if not task_match:
        suggestions.append("补充与当前商品、平台和推广目标直接相关的事实与场景")
    issues = []
    if unsupported:
        issues.append(f"未经确认的商品事实：{'、'.join(unsupported)}")
    if risks:
        issues.append(f"绝对化或高风险表达：{'、'.join(risks)}")
    if not task_match:
        issues.append("内容与当前商品事实、场景或任务结构不匹配")
    rule_categories = sorted({
        str((item.get("metadata") or {}).get("category") or item.get("category") or "")
        for item in retrieved_knowledge
        if isinstance(item, dict)
    } - {""})
    return ContentReview(
        passed=fact_consistency and compliance and task_match,
        fact_consistency=fact_consistency,
        compliance=compliance,
        task_match=task_match,
        risk_items=risks,
        unsupported_claims=unsupported,
        suggestions=suggestions,
        issues=issues,
        risk_level="high" if unsupported or risks else ("medium" if not task_match else "low"),
        checked_rule_categories=rule_categories,
    )


@tool("content_review_tool", args_schema=ContentReviewInput)
def content_review_tool(
    content: dict[str, Any] | str,
    product_analysis: dict[str, Any] | None = None,
    confirmed_features: str = "",
    usage_scenarios: str = "",
    task_type: str = "promotion_plan",
    retrieved_knowledge: list[dict[str, Any]] | None = None,
    decision_reason: str = "",
) -> dict[str, Any]:
    """Review content without modifying, regenerating, or re-reviewing it."""
    analysis = product_analysis or {}
    confirmed = confirmed_features or "；".join(
        str(item) for item in analysis.get("confirmed_features", [])
    )
    scenarios = usage_scenarios or "；".join(
        str(item) for item in analysis.get(
            "usage_scenarios",
            analysis.get("marketing_scenarios", []),
        )
    )
    return _review_content(
        content,
        confirmed_features=confirmed,
        usage_scenarios=scenarios,
        task_type=task_type,
        retrieved_knowledge=retrieved_knowledge or [],
    ).model_dump()


def content_review_for_state(state: Any) -> dict[str, Any]:
    """Backward-compatible AgentState adapter for deterministic review."""
    product = state.product_info
    result = _review_content(
        state.generated_content,
        confirmed_features=_field(
            product,
            "confirmed_features",
            "已确认商品事实",
            "已确认商品卖点",
        ),
        usage_scenarios=_field(product, "usage_scenarios", "使用场景"),
        task_type=str(state.task_type),
        retrieved_knowledge=state.retrieved_knowledge,
    )
    return {"review_result": result}
