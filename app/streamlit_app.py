"""Streamlit interface for the AI clothing marketing strategy assistant."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app.agent.ecommerce_agent import run_agent  # noqa: E402
from app.agent.agent_state import AgentState  # noqa: E402
from marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
)


PAGE_CSS = """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 { letter-spacing: 0; color: #17202a; }
    h1 { margin-bottom: 0.25rem; }
    .workflow-step { padding: 0.75rem 0.9rem; border: 1px solid #dfe4e8; border-radius: 8px; text-align: center; background: #fff; }
    .workflow-step.active { border-color: #2e7d62; background: #f1f8f4; }
    [data-testid="stCaptionContainer"] { color: #5f6b76; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dfe4e8;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(23, 32, 42, 0.04);
    }
    [data-testid="stForm"] { border: 0; padding: 0; }
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button { border-radius: 6px; font-weight: 600; }
    [data-testid="stAlert"] { border-radius: 8px; }
    hr { margin: 2.25rem 0; border-color: #e7eaed; }
</style>
"""


def _render_value(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            st.markdown(f"- {item}")
    else:
        st.write(value)


def _result_card(title: str, strategy: MarketingStrategy, field: str) -> None:
    with st.container():
        st.subheader(title)
        value = strategy.get(field) if isinstance(strategy, dict) else getattr(strategy, field, None)
        if value is not None:
            _render_value(value)


def _render_header() -> None:
    st.title("AI 电商运营助手")
    st.caption("基于 LLM + RAG + Agent + Tool Calling 的营销自动化工作流")


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value if isinstance(value, dict) else {}


def _render_workflow(state: AgentState) -> None:
    st.subheader("Agent 工作流")
    trace = set(state.tool_trace)
    steps = (
        ("商品理解", "product_analysis"),
        ("知识检索", "knowledge_retrieval"),
        ("营销策略生成", "marketing_strategy"),
        ("内容生成", "content_generation"),
        ("内容审核", "content_review"),
        ("最终结果", "final"),
    )
    columns = st.columns(len(steps))
    for index, (label, key) in enumerate(steps):
        active = key in trace or (key == "final" and state.final_result is not None)
        with columns[index]:
            st.markdown(
                f'<div class="workflow-step{" active" if active else ""}">'
                f'{"✓ " if active else ""}{label}</div>',
                unsafe_allow_html=True,
            )


def _render_agent_process(state: AgentState) -> None:
    with st.expander("查看 Agent 执行过程", expanded=False):
        if state.tool_trace:
            st.write("Tool 调用顺序：" + " → ".join(state.tool_trace))
        st.write(f"Tool 调用次数：{state.tool_call_count}")
        st.write(f"Agent 迭代次数：{state.iteration_count}")
        events = state.to_dict().get("events", [])
        tool_results = [
            event for event in events
            if event.get("type") in {"tool_result", "tool_error"}
        ]
        if tool_results:
            st.write("Tool 执行结果摘要")
            for event in tool_results:
                with st.container(border=True):
                    st.caption(f"{event.get('tool', 'unknown')} · {event.get('status', 'error')}")
                    st.write(str(event.get("result", event.get("error", "")))[:2000])
        if events:
            st.json(events)


def _render_retrieval(state: AgentState) -> None:
    documents = state.retrieved_knowledge
    if not documents:
        return
    with st.expander("查看知识检索结果", expanded=False):
        st.write(f"检索结果数量：{len(documents)}")
        for index, document in enumerate(documents, 1):
            if isinstance(document, dict):
                metadata = document.get("metadata") or {}
                content = document.get("content", document.get("page_content", ""))
            else:
                metadata = getattr(document, "metadata", {}) or {}
                content = getattr(document, "page_content", "")
            metadata = metadata if isinstance(metadata, dict) else {}
            source = metadata.get("source", document.get("source", "unknown") if isinstance(document, dict) else "unknown")
            st.markdown(f"**{index}. {source}**")
            details = {
                key: metadata[key]
                for key in ("filename", "category", "retrieval_score")
                if metadata.get(key) is not None
            }
            if details:
                st.json(details)
            st.write(str(content)[:1200])


def _render_content(content_value: Any) -> None:
    content = _as_dict(content_value)
    if not content:
        st.write(content_value)
        return
    if content.get("title"):
        st.subheader(content["title"])
    if content.get("body"):
        st.write(content["body"])
    if content.get("tags"):
        st.write("标签：" + " ".join(content["tags"]))
    if content.get("call_to_action"):
        st.write("行动引导：" + content["call_to_action"])


def _render_review(review_value: Any) -> None:
    review = _as_dict(review_value)
    if not review:
        st.write(review_value)
        return
    passed = review.get("passed")
    if passed is True:
        st.success("内容审核通过")
    elif passed is False:
        st.warning("内容审核未通过，需要根据审核意见处理")
    if "needs_human_review" in review:
        st.write(f"需要人工复核：{'是' if review['needs_human_review'] else '否'}")
    for key in ("correction_count", "risk_level", "suggestions", "issues", "unsupported_claims"):
        if review.get(key):
            st.write(f"{key}：{review[key]}")
    st.json(review)


def _render_product_input() -> tuple[dict[str, str], bool]:
    with st.container():
        st.header("商品信息")
        with st.form("product_form"):
            task = st.text_area(
                "运营任务",
                value="",
                placeholder="请输入运营任务",
                height=80,
            )
            brand_name = st.text_input(
                "品牌名称",
                placeholder="请输入品牌名称",
            )
            product_name = st.text_input(
                "商品名称",
                placeholder="请输入商品名称",
            )
            product_type = st.text_input(
                "商品类型",
                placeholder="例如：外套、裤装、羽绒服等",
            )
            confirmed_features = st.text_area(
                "已确认商品卖点",
                placeholder="请输入已经确认的商品特点、功能或卖点",
                height=100,
            )
            usage_scenarios = st.text_area(
                "使用场景",
                placeholder="请输入商品实际适用的使用场景",
                height=100,
            )
            product_details = st.text_area(
                "商品详细资料",
                placeholder="请输入面料、设计、规格等已经确认的商品资料",
                height=140,
            )
            draft_content = st.text_area(
                "待审核文案（仅事实/合规检查任务填写）",
                placeholder="粘贴需要检查的标题、正文或完整文案",
                height=120,
            )
            submitted = st.form_submit_button(
                "开始生成营销方案",
                type="primary",
                use_container_width=True,
            )
    return (
        {
            "task": task,
            "brand_name": brand_name,
            "product_name": product_name,
            "product_type": product_type,
            "confirmed_features": confirmed_features,
            "usage_scenarios": usage_scenarios,
            "product_details": product_details,
            "draft_content": draft_content,
        },
        submitted,
    )


def _render_marketing_strategy(strategy: MarketingStrategy) -> None:
    st.header("营销策略")
    fields = (
        ("商品定位", "product_positioning"),
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
    for title, field in fields:
        _result_card(title, strategy, field)


def _generate(
    task: str,
    brand_name: str,
    product_name: str,
    product_type: str,
    confirmed_features: str,
    usage_scenarios: str,
    product_details: str,
    draft_content: str,
) -> AgentState:
    return run_agent(
        task=task,
        product_info={
            "brand_name": brand_name,
            "product_name": product_name,
            "product_type": product_type,
            "confirmed_features": confirmed_features,
            "usage_scenarios": usage_scenarios,
            "product_details": product_details,
            "platform": "小红书" if "小红书" in task else "",
            "marketing_goal": "推广" if any(word in task for word in ("推广", "方案")) else "",
            "draft_content": draft_content,
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="AI 电商运营助手",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    _render_header()
    st.divider()
    product, submitted = _render_product_input()

    if submitted:
        st.session_state.pop("strategy_result", None)
        st.session_state.pop("agent_state", None)
        required = {
            "运营任务": product["task"],
            "商品名称": product["product_name"],
            "商品类型": product["product_type"],
            "已确认商品卖点": product["confirmed_features"],
        }
        missing = [label for label, value in required.items() if not value.strip()]
        if missing:
            st.error(f"请填写：{'、'.join(missing)}")
        else:
            try:
                with st.status("Agent 正在执行营销任务…", expanded=True) as execution:
                    st.write("商品理解 → 知识检索 → 营销策略生成 → 内容生成 → 内容审核 → 最终结果")
                    state = _generate(
                        product["task"].strip(),
                        product["brand_name"].strip(),
                        product["product_name"].strip(),
                        product["product_type"].strip(),
                        product["confirmed_features"].strip(),
                        product["usage_scenarios"].strip(),
                        product["product_details"].strip(),
                        product["draft_content"].strip(),
                    )
                    execution.update(label="Agent 执行完成", state="complete", expanded=False)
                st.session_state["agent_state"] = state
            except Exception as exc:
                st.error(f"执行失败：{type(exc).__name__}: {exc}")

    state = st.session_state.get("agent_state")
    if state is None:
        return

    final = _as_dict(state.final_result)
    if final.get("status") == "needs_human_review":
        st.warning("Agent 已结束，本次结果需要人工复核")
    elif state.errors:
        st.warning("Agent 已结束，执行过程包含错误，请查看执行详情")
    else:
        st.success("Agent 执行完成")
    _render_workflow(state)
    _render_agent_process(state)
    _render_retrieval(state)
    st.divider()
    if isinstance(state.final_result, dict) and state.final_result.get("answer"):
        st.header("最终营销方案")
        st.write(state.final_result["answer"])
    if state.marketing_strategy is not None:
        with st.expander("查看结构化营销策略", expanded=not bool(final.get("answer"))):
            _render_marketing_strategy(state.marketing_strategy)
    if state.generated_content is not None:
        st.header("小红书内容")
        _render_content(state.generated_content)
    if state.review_result is not None:
        st.header("内容审核")
        _render_review(state.review_result)
    result_data = state.to_dict()
    evaluation = (
        getattr(state, "evaluation_report", None)
        or getattr(state, "evaluation", None)
        or result_data.get("evaluation_report")
        or result_data.get("evaluation")
        or final.get("evaluation_report")
        or final.get("evaluation")
    )
    if evaluation:
        with st.expander("查看评估结果", expanded=False):
            st.json(evaluation)


if __name__ == "__main__":
    main()
