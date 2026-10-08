"""Offline protocol checks for LangChain tool dispatch and Python constraints.

``ScriptedModel`` returns prebuilt ``AIMessage.tool_calls``.  These tests verify
the Agent protocol but do not prove that a real DashScope model autonomously
produces tool calls.  Real online verification belongs to
``experiments/agent_v2_demo.py`` and is intentionally not run here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.agent_state import AgentState, MAX_ITERATIONS, TaskType
from app.agent.agent_tools import (
    ContentReview,
    XiaohongshuContent,
    content_review_for_state,
)
from app.agent.ecommerce_agent import EcommerceAgent
from app.product_analyzer import ProductAnalysis


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _analysis() -> ProductAnalysis:
    return ProductAnalysis(
        brand_name="AAA",
        product_name="户外三合一冲锋衣",
        product_type="户外服装",
        confirmed_features=["三合一设计", "防风", "防水"],
        core_selling_points=["三合一设计", "防风", "防水"],
        target_audience=["户外运动及登山人群"],
        usage_scenarios=["登山", "徒步", "户外运动"],
        user_needs=["户外活动中的穿着需求"],
        consumer_value=["满足已确认场景下的穿着需求"],
    )


def _mock_analyze_product() -> Mock:
    tool = Mock()
    tool.name = "analyze_product"
    tool.invoke.return_value = _analysis().model_dump()
    return tool


class ScriptedModel:
    """Offline protocol double returning actual LangChain AIMessage objects."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.messages: list[list[object]] = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.messages.append(list(messages))
        if not self.responses:
            raise AssertionError("测试脚本没有提供下一条 LLM 消息")
        return self.responses.pop(0)


def _call(name: str, number: int, **args: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": args,
            "id": f"call_{number}",
            "type": "tool_call",
        }],
    )


def _product_call(number: int, decision_reason: str) -> AIMessage:
    return _call(
        "analyze_product",
        number,
        product_name="户外三合一冲锋衣",
        product_selling_points="三合一设计、防风、防水",
        decision_reason=decision_reason,
    )


def test_task_classification_is_not_routing() -> None:
    cases = (
        ("这款冲锋衣有哪些卖点？", TaskType.PRODUCT_QA),
        ("帮我制定一套小红书推广方案", TaskType.PROMOTION_PLAN),
        ("帮我检查这篇文案有没有事实错误", TaskType.CONTENT_REVIEW),
    )
    for task, expected in cases:
        state = AgentState(task=task)
        EcommerceAgent().understand_task(state)
        _assert(state.task_type == expected, f"任务分类错误：{task}")


def test_offline_protocol_tool_call_message_round_trip() -> None:
    model = ScriptedModel([
        _product_call(1, "回答商品卖点问题需要分析确认事实"),
        AIMessage(content="已确认卖点为三合一设计、防风和防水。"),
    ])
    state = AgentState(
        task="这款冲锋衣有哪些卖点？",
        product_info={
            "brand_name": "AAA",
            "product_name": "户外三合一冲锋衣",
            "product_type": "户外服装",
            "confirmed_features": "三合一设计、防风、防水",
            "usage_scenarios": "登山、徒步、户外运动",
        },
    )
    with patch("app.agent.ecommerce_agent.analyze_product", new=_mock_analyze_product()):
        result = EcommerceAgent(model=model).run(state)
    _assert(result.tool_trace == ["product_analysis"], "LLM 选择的工具没有执行或 trace 不符")
    _assert(result.final_result["answer"].startswith("已确认卖点"), "最终答复没有来自 LLM")
    second_turn = model.messages[1]
    _assert(any(isinstance(message, ToolMessage) for message in second_turn), "ToolMessage 未回传给 LLM")
    _assert(result.tool_call_count == 1, "State 未记录真实工具调用次数")
    _assert(
        [tool.name for tool in model.bound_tools] == [
            "analyze_product",
            "knowledge_retrieval_tool",
            "marketing_strategy_tool",
            "content_generation_tool",
            "content_review_tool",
        ],
        "五个当前 Tool 未按真实注册名称绑定",
    )


def test_product_qa_selects_only_analysis_and_retrieval_offline() -> None:
    model = ScriptedModel([
        _product_call(1, "整理商品已确认卖点"),
        _call("knowledge_retrieval_tool", 2, decision_reason="查找辅助理解该商品的通用营销知识"),
        AIMessage(content="该冲锋衣确认的卖点是三合一设计、防风和防水。"),
    ])
    state = AgentState(
        task="这款冲锋衣有哪些卖点？",
        product_info={
            "product_name": "户外三合一冲锋衣",
            "product_type": "户外服装",
            "confirmed_features": "三合一设计、防风、防水",
            "usage_scenarios": "登山、徒步、户外运动",
        },
    )
    with (
        patch("app.agent.ecommerce_agent.analyze_product", new=_mock_analyze_product()),
        patch("app.agent.ecommerce_agent.knowledge_retrieval_tool", return_value={"retrieved_knowledge": []}),
    ):
        result = EcommerceAgent(model=model).run(state)
    _assert(result.tool_trace == ["product_analysis", "knowledge_retrieval"], "卖点问答调用了错误工具")
    _assert(not {"marketing_strategy", "content_generation", "content_review"}.intersection(result.tool_trace), "卖点问答不应生成推广内容")
    _assert(len(model.messages) == 3, "预期模型经过两次工具结果反馈后再完成任务")
    _assert(
        all(sum(isinstance(message, HumanMessage) for message in turn) == 1 for turn in model.messages),
        "每轮模型上下文都应保留原始用户任务",
    )


def test_promotion_task_can_select_all_five_tools_offline() -> None:
    model = ScriptedModel([
        _product_call(1, "营销计划需先基于确认商品事实"),
        _call("knowledge_retrieval_tool", 2, decision_reason="营销方案需要检索平台和营销规范"),
        _call("marketing_strategy_tool", 3, decision_reason="检索结果可支持方案结构化"),
        _call("content_generation_tool", 4, decision_reason="用户要求制定推广方案，需要配套内容"),
        _call("content_review_tool", 5, decision_reason="生成内容后需检查事实与规范"),
        AIMessage(content="推广方案及文案已生成并通过规则审核。"),
    ])
    state = AgentState(
        task="帮我制定一套小红书推广方案",
        product_info={
            "product_name": "户外三合一冲锋衣",
            "product_type": "户外服装",
            "confirmed_features": "三合一设计、防风、防水",
            "usage_scenarios": "登山、徒步、户外运动",
        },
    )
    passing_review = ContentReview(passed=True, fact_consistency=True, compliance=True, task_match=True)
    with (
        patch("app.agent.ecommerce_agent.analyze_product", new=_mock_analyze_product()),
        patch("app.agent.ecommerce_agent.knowledge_retrieval_tool", return_value={"retrieved_knowledge": []}),
        patch("app.agent.ecommerce_agent.marketing_strategy_tool", return_value={"marketing_strategy": {"direction": "以已确认场景为主"}}),
        patch("app.agent.ecommerce_agent.content_generation_tool", return_value={"generated_content": XiaohongshuContent(title="户外穿搭", body="三合一设计，适合徒步。", tags=["冲锋衣"], call_to_action="查看详情")}),
        patch("app.agent.ecommerce_agent.content_review_tool", return_value={"review_result": passing_review}),
    ):
        result = EcommerceAgent(model=model).run(state)
    _assert(result.tool_trace == [
        "product_analysis", "knowledge_retrieval", "marketing_strategy", "content_generation", "content_review"
    ], "推广任务的模型工具调用轨迹不完整")
    _assert(result.iteration_count == 0, "首次通过时不应发生修正")


def test_review_task_uses_retrieval_and_review_only_offline() -> None:
    model = ScriptedModel([
        _call("knowledge_retrieval_tool", 1, decision_reason="需要适用的营销事实与平台风险规则"),
        _call("content_review_tool", 2, decision_reason="核验用户提供文案中的商品事实"),
        AIMessage(content="文案中的商品卖点和使用场景均有输入资料支持。"),
    ])
    state = AgentState(
        task="帮我检查这篇文案有没有事实错误",
        product_info={
            "product_name": "户外三合一冲锋衣",
            "product_type": "户外服装",
            "confirmed_features": "三合一设计、防风、防水",
            "usage_scenarios": "登山、徒步、户外运动",
            "draft_content": "三合一设计，适合登山、徒步和户外运动。",
        },
    )
    passing_review = ContentReview(passed=True, fact_consistency=True, compliance=True, task_match=True)
    with (
        patch("app.agent.ecommerce_agent.knowledge_retrieval_tool", return_value={"retrieved_knowledge": []}),
        patch("app.agent.ecommerce_agent.content_review_tool", return_value={"review_result": passing_review}),
    ):
        result = EcommerceAgent(model=model).run(state)
    _assert(result.tool_trace == ["knowledge_retrieval", "content_review"], "审核任务应仅检索和审核")
    _assert(not {"product_analysis", "marketing_strategy", "content_generation"}.intersection(result.tool_trace), "审核任务错误地生成了内容")


def test_max_tool_call_count_is_enforced() -> None:
    model = ScriptedModel([
        _product_call(1, "先分析商品事实"),
        _call("knowledge_retrieval_tool", 2, decision_reason="尝试超出配置的调用上限"),
        AIMessage(content="工具调用上限已到，基于已有结果结束。"),
    ])
    state = AgentState(
        task="这款冲锋衣有哪些卖点？",
        product_info={
            "product_name": "户外三合一冲锋衣",
            "product_type": "户外服装",
            "confirmed_features": "三合一设计、防风、防水",
        },
    )
    with patch("app.agent.ecommerce_agent.analyze_product", new=_mock_analyze_product()):
        result = EcommerceAgent(model=model, max_tool_calls=1).run(state)
    _assert(result.tool_call_count == 1, "最大工具调用次数没有生效")
    _assert(result.tool_trace == ["product_analysis"], "超出上限的工具不应被执行")
    _assert(any("最大 Tool Calling 次数" in error for error in result.errors), "上限拒绝未记录")


def test_review_iteration_is_limited_to_two() -> None:
    responses = [
        _product_call(1, "内容生成依赖商品分析"),
        _call("content_generation_tool", 2, decision_reason="生成推广方案文案"),
        _call("content_review_tool", 3, decision_reason="检查刚生成的内容"),
        _call("content_generation_tool", 4, decision_reason="依据审核意见修正文案"),
        _call("content_review_tool", 5, decision_reason="复核第一次修正"),
        _call("content_generation_tool", 6, decision_reason="依据审核意见再次修正"),
        _call("content_review_tool", 7, decision_reason="复核第二次修正"),
        _call("content_generation_tool", 8, decision_reason="不应允许第三次自动修正"),
        AIMessage(content="已达到修正上限，转人工审核。"),
    ]
    model = ScriptedModel(responses)
    state = AgentState(
        task="帮我制定一套小红书推广方案",
        product_info={
            "product_name": "户外三合一冲锋衣",
            "product_type": "户外服装",
            "confirmed_features": "三合一设计、防风、防水",
            "usage_scenarios": "登山、徒步、户外运动",
        },
    )
    failed_review = ContentReview(
        passed=False,
        fact_consistency=False,
        compliance=True,
        task_match=True,
        unsupported_claims=["保暖"],
        suggestions=["删除未确认的保暖描述"],
    )
    generation = {"generated_content": XiaohongshuContent(
        title="户外穿搭",
        body="三合一设计，适合登山和徒步。",
        tags=["冲锋衣"],
        call_to_action="查看详情",
    )}
    with (
        patch("app.agent.ecommerce_agent.analyze_product", new=_mock_analyze_product()),
        patch("app.agent.ecommerce_agent.content_generation_tool", return_value=generation),
        patch("app.agent.ecommerce_agent.content_review_tool", return_value={"review_result": failed_review}),
    ):
        result = EcommerceAgent(model=model).run(state)
    _assert(result.iteration_count == MAX_ITERATIONS == 2, "审核修正超过或未达到上限")
    _assert(result.tool_trace.count("content_generation") == 3, "应为首次生成加两次修正")
    _assert(any("最大自动修正次数" in error for error in result.errors), "第三次修正拒绝未记录")
    _assert(result.final_result["status"] == "needs_human_review", "审核失败后应转人工处理")


def test_fact_boundary_review_still_blocks_unconfirmed_claims() -> None:
    state = AgentState(
        task="帮我检查这篇文案有没有事实错误",
        product_info={
            "confirmed_features": "三合一设计、防风、防水",
            "usage_scenarios": "登山、徒步、户外运动",
        },
        task_type=TaskType.CONTENT_REVIEW,
    )
    state.generated_content = {
        "title": "冲锋衣体验",
        "body": "三合一设计，适合登山；这件衣服保暖且透气。",
        "tags": ["冲锋衣"],
        "call_to_action": "查看详情",
    }
    result = content_review_for_state(state)["review_result"]
    _assert(not result.passed, "未确认属性不应通过")
    _assert("保暖" in result.unsupported_claims and "透气" in result.unsupported_claims, "事实边界漏检")


def test_tool_schema_requires_decision_reason() -> None:
    tool_schema = EcommerceAgent(model=ScriptedModel([]))._create_tools(AgentState(task="test"))[0]
    try:
        tool_schema.invoke({
            "product_name": "户外三合一冲锋衣",
            "product_selling_points": "三合一设计、防风、防水",
        })
    except Exception as exc:
        _assert("decision_reason" in str(exc), "Schema 拒绝原因不是缺少 decision_reason")
        return
    raise AssertionError("缺少必填 decision_reason 时工具没有拒绝执行")


def main() -> int:
    test_task_classification_is_not_routing()
    test_offline_protocol_tool_call_message_round_trip()
    test_product_qa_selects_only_analysis_and_retrieval_offline()
    test_promotion_task_can_select_all_five_tools_offline()
    test_review_task_uses_retrieval_and_review_only_offline()
    test_max_tool_call_count_is_enforced()
    test_review_iteration_is_limited_to_two()
    test_fact_boundary_review_still_blocks_unconfirmed_claims()
    test_tool_schema_requires_decision_reason()
    print("Agent V2 offline checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
