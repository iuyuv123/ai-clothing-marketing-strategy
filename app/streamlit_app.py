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

import product_analyzer  # noqa: E402
from marketing_strategy_generator import (  # noqa: E402
    MarketingStrategy,
    generate_marketing_strategy,
)
from marketing_strategy_workflow import (  # noqa: E402
    build_retrieval_query,
    retrieve_knowledge,
)


TOP_K = 5


PAGE_CSS = """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    h1, h2, h3 { letter-spacing: 0; color: #17202a; }
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
    with st.container(border=True):
        st.subheader(title)
        _render_value(getattr(strategy, field))


def _render_header() -> None:
    st.title("AI 服装营销策略助手")
    st.caption("输入商品资料，AI 自动生成结构化营销策略。")


def _render_product_input() -> tuple[dict[str, str], bool]:
    with st.container(border=True):
        st.header("输入商品资料")
        with st.form("product_form"):
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
            submitted = st.form_submit_button(
                "生成营销策略",
                type="primary",
                use_container_width=True,
            )
    return (
        {
            "brand_name": brand_name,
            "product_name": product_name,
            "product_type": product_type,
            "confirmed_features": confirmed_features,
            "usage_scenarios": usage_scenarios,
            "product_details": product_details,
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
    brand_name: str,
    product_name: str,
    product_type: str,
    confirmed_features: str,
    usage_scenarios: str,
    product_details: str,
) -> MarketingStrategy:
    analysis = product_analyzer.analyze_product(
        brand_name=brand_name,
        product_name=product_name,
        product_type=product_type,
        confirmed_features=confirmed_features,
        usage_scenarios=usage_scenarios,
        product_details=product_details,
    )
    query = build_retrieval_query(analysis)
    documents = retrieve_knowledge(query, top_k=TOP_K)
    return generate_marketing_strategy(analysis, documents)


def main() -> None:
    st.set_page_config(
        page_title="AI 服装营销策略助手",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    _render_header()
    st.divider()
    product, submitted = _render_product_input()

    if submitted:
        st.session_state.pop("strategy_result", None)
        required = {
            "商品名称": product["product_name"],
            "商品类型": product["product_type"],
            "已确认商品卖点": product["confirmed_features"],
        }
        missing = [label for label, value in required.items() if not value.strip()]
        if missing:
            st.error(f"请填写：{'、'.join(missing)}")
        else:
            try:
                with st.spinner("正在生成营销策略..."):
                    strategy = _generate(
                        product["brand_name"].strip(),
                        product["product_name"].strip(),
                        product["product_type"].strip(),
                        product["confirmed_features"].strip(),
                        product["usage_scenarios"].strip(),
                        product["product_details"].strip(),
                    )
                st.session_state["strategy_result"] = strategy
            except Exception:
                st.error("营销策略生成失败，请检查商品资料后重试。")

    strategy = st.session_state.get("strategy_result")
    if strategy is None:
        return

    st.success("营销策略生成完成")
    st.divider()
    _render_marketing_strategy(strategy)


if __name__ == "__main__":
    main()
