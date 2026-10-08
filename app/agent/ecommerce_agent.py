"""LLM tool-calling orchestration for the AI e-commerce operations agent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.agent_state import AgentState, MAX_ITERATIONS, TaskType
from app.agent.agent_tools import (
    ContentReview,
    XiaohongshuContent,
    analyze_product,
    content_generation_for_state,
    content_generation_tool as content_generation_langchain_tool,
    content_review_for_state,
    content_review_tool as content_review_langchain_tool,
    knowledge_retrieval_for_state,
    knowledge_retrieval_tool as knowledge_retrieval_langchain_tool,
    marketing_strategy_for_state,
    marketing_strategy_tool as marketing_strategy_langchain_tool,
    product_analysis_tool,
)
from app.llm_factory import create_chat_model
from app.marketing_strategy_generator import MarketingStrategy
from app.product_analyzer import ProductAnalysis


# Backward-compatible patch points for existing state-based Agent tests.
knowledge_retrieval_tool = knowledge_retrieval_for_state
marketing_strategy_tool = marketing_strategy_for_state
content_generation_tool = content_generation_for_state
content_review_tool = content_review_for_state


MAX_TOOL_CALLS = 12
TOOL_DESCRIPTIONS = {
    "analyze_product": "Analyze the provided product's confirmed facts, audience, benefits, and use scenarios.",
    "knowledge_retrieval_tool": "Retrieve relevant marketing and platform guidance from the Chroma knowledge base.",
    "marketing_strategy_tool": "Create a structured marketing strategy using product analysis and retrieved guidance.",
    "content_generation_tool": "Generate or correct Xiaohongshu content using only confirmed product facts and the current state.",
    "content_review_tool": "Review the current draft for unsupported claims, compliance risks, and task fit.",
}
TOOL_TRACE_NAMES = {
    "analyze_product": "product_analysis",
    "product_analysis_tool": "product_analysis",
    "knowledge_retrieval_tool": "knowledge_retrieval",
    "marketing_strategy_tool": "marketing_strategy",
    "content_generation_tool": "content_generation",
    "content_review_tool": "content_review",
}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "page_content"):
        return {
            "page_content": str(value.page_content),
            "metadata": _jsonable(getattr(value, "metadata", {}) or {}),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, default=str)


def _decision_text(message: Any) -> str:
    calls = getattr(message, "tool_calls", []) or []
    if calls:
        return "模型请求工具：" + _json_text([
            {"name": call.get("name"), "arguments": call.get("args", {})}
            for call in calls
        ])
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip() or "模型返回了无文本的最终消息"
    return _json_text(content)


class EcommerceAgent:
    """Use model-returned tool calls for planning and Python for execution limits."""

    def __init__(
        self,
        *,
        max_iterations: int = MAX_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        model: Any | None = None,
    ) -> None:
        if max_iterations < 0 or max_iterations > MAX_ITERATIONS:
            raise ValueError(f"max_iterations 必须在 0 到 {MAX_ITERATIONS} 之间")
        if max_tool_calls < 1 or max_tool_calls > MAX_TOOL_CALLS:
            raise ValueError(f"max_tool_calls 必须在 1 到 {MAX_TOOL_CALLS} 之间")
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.model = model

    def understand_task(self, state: AgentState) -> AgentState:
        """Classify for logs/evaluation only; this value does not route tools."""
        task = state.task
        if any(word in task for word in ("检查", "审核", "事实错误", "合规")):
            state.task_type = TaskType.CONTENT_REVIEW
        elif any(word in task for word in ("方案", "推广", "小红书", "文案", "内容")):
            state.task_type = TaskType.PROMOTION_PLAN
        elif any(word in task for word in ("卖点", "介绍", "有哪些", "是什么")):
            state.task_type = TaskType.PRODUCT_QA
        else:
            state.task_type = TaskType.UNKNOWN
        state.record_event("task_understanding", task_type=state.task_type.value)
        return state

    @staticmethod
    def _state_context(state: AgentState) -> dict[str, Any]:
        summary = _jsonable(state.to_dict())
        product_info = summary.get("product_info", {})
        if isinstance(product_info, dict):
            product_info.pop("draft_content", None)
        return summary

    @staticmethod
    def _merge(state: AgentState, tool_name: str, result: dict[str, Any]) -> AgentState:
        state.record_tool(TOOL_TRACE_NAMES[tool_name])
        for key, value in result.items():
            if key == "product_analysis":
                state.product_analysis = value
            elif key == "target_user":
                state.target_user = list(value or [])
            elif key == "platform":
                state.platform = str(value or "")
            elif key == "marketing_goal":
                state.marketing_goal = str(value or "")
            elif key == "retrieved_knowledge":
                state.retrieved_knowledge = list(value or [])
            elif key == "marketing_strategy":
                state.marketing_strategy = value
            elif key == "generated_content":
                state.generated_content = value
            elif key == "review_result":
                state.review_result = value
            elif key == "query":
                state.record_event("retrieval_query", query=value)
            else:
                state.record_event("tool_output", tool=tool_name, field=key, value=value)
        return state

    def _validate_prerequisites(self, state: AgentState, tool_name: str) -> None:
        if tool_name in {"analyze_product", "product_analysis_tool"}:
            product = state.product_info
            required = {
                "商品名称": product.get("product_name", product.get("商品名称")),
                "商品类型": product.get("product_type", product.get("商品类型")),
                "已确认商品事实": product.get(
                    "confirmed_features", product.get("已确认商品事实", product.get("已确认商品卖点"))
                ),
            }
            missing = [name for name, value in required.items() if not str(value or "").strip()]
            if missing:
                raise ValueError(f"商品分析缺少必填资料：{'、'.join(missing)}")
        elif tool_name == "marketing_strategy_tool":
            if not isinstance(state.product_analysis, ProductAnalysis):
                raise ValueError("营销策略前必须先调用并完成 product_analysis_tool")
            if "knowledge_retrieval" not in state.tool_trace:
                raise ValueError("营销策略前必须先调用 knowledge_retrieval_tool")
        elif tool_name == "content_generation_tool":
            if not isinstance(state.product_analysis, ProductAnalysis):
                raise ValueError("内容生成前必须先调用并完成 product_analysis_tool")
            if state.review_result is not None and not state.review_result.passed:
                if state.iteration_count >= self.max_iterations:
                    raise RuntimeError("已达到最大自动修正次数，禁止继续生成")
                state.correction_notes = list(state.review_result.suggestions)
                state.increment_iteration()
                state.record_event("correction", iteration_count=state.iteration_count)
        elif tool_name == "content_review_tool" and state.generated_content in (None, "", {}):
            raise ValueError("内容审核前必须提供待审核文案或调用 content_generation_tool")

    def _create_tools(self, state: AgentState) -> list[BaseTool]:
        return [
            analyze_product,
            knowledge_retrieval_langchain_tool,
            marketing_strategy_langchain_tool,
            content_generation_langchain_tool,
            content_review_langchain_tool,
        ]

    @staticmethod
    def _system_prompt(state: AgentState) -> str:
        product = state.product_info
        confirmed = product.get(
            "confirmed_features",
            product.get("已确认商品事实", product.get("已确认商品卖点", [])),
        )
        scenarios = product.get("usage_scenarios", product.get("使用场景", []))
        analysis = state.product_analysis
        audience = state.target_user or getattr(analysis, "target_audience", [])
        return f"""You are an AI e-commerce operations agent. You choose tools from the user's task and current AgentState; Python executes and validates each call.

Do not follow a fixed tool sequence. Select only capabilities needed for this task, and after every tool result reassess whether another tool is useful or the task is complete. Product questions normally need product analysis and relevant knowledge retrieval; they do not need strategy or content. Review-only tasks should inspect the supplied draft and retrieve rules, and should not create a replacement unless the user asks. Promotion plans may need product analysis, knowledge retrieval, strategy, content generation, and review, but you decide the order and whether each is needed.

For a product-selling-points question, normally call analyze_product and knowledge_retrieval_tool only. For a promotion plan, use the five capabilities as needed, ensuring generation is reviewed. For a request to check an existing draft, normally call knowledge_retrieval_tool and content_review_tool only. These are task-fit guidelines, not a Python-enforced route.

Current product facts explicitly supplied by the caller: {_json_text(confirmed)}
Current usage scenarios explicitly supplied by the caller: {_json_text(scenarios)}
Known target users, if already supplied or analyzed: {_json_text(audience)}
Only the current product's explicit facts may be stated as product attributes. RAG documents are guidance, not evidence of this product's attributes. Do not invent capabilities, materials, measurements, performance levels, test data, standards, or certifications. Do not expand the supplied scenarios into more demanding environments or guarantees. Follow retrieved platform and marketing rules; never make unsupported or absolute claims.

After a failed content review, read the review result and decide whether correction, another retrieval, or a final handoff is appropriate. Python limits automatic content corrections to two. When calling a tool, provide a concise decision_reason. Give a concise final answer based on the task and completed state; do not claim tools ran unless their results are in the conversation."""

    def _initial_messages(self, state: AgentState) -> list[Any]:
        return [
            SystemMessage(content=self._system_prompt(state)),
            HumanMessage(content=(
                f"用户任务：{state.task}\n"
                f"TaskType（仅分类信息，不是路由指令）：{state.task_type.value}\n"
                f"当前 AgentState：{_json_text(self._state_context(state))}"
            )),
        ]

    @staticmethod
    def _event(state: AgentState, event_type: str, **payload: Any) -> None:
        state.record_event(event_type, **_jsonable(payload))

    def _invoke_tool(
        self,
        state: AgentState,
        tool_map: dict[str, BaseTool],
        call: dict[str, Any],
        *,
        remaining_calls: int,
    ) -> ToolMessage:
        name = str(call.get("name", ""))
        call_id = str(call.get("id", ""))
        args = call.get("args", {})
        print(f"[TOOL CALL]\ntool: {name}\narguments: {_json_text(args)}")
        self._event(state, "tool_call", tool=name, arguments=args, tool_call_id=call_id)

        try:
            if not isinstance(args, dict):
                raise ValueError("Tool arguments 必须是 JSON 对象")
            is_legacy_product_call = name == "product_analysis_tool"
            if name not in tool_map and not is_legacy_product_call:
                raise ValueError(f"未知工具：{name}")
            if remaining_calls <= 0:
                raise RuntimeError("已达到最大 Tool Calling 次数")
            if is_legacy_product_call:
                self._validate_prerequisites(state, name)
                result = product_analysis_tool(state)
                self._merge(state, name, result)
            elif name == "knowledge_retrieval_tool" and "query" not in args:
                result = knowledge_retrieval_tool(state)
                self._merge(state, name, result)
            elif name == "marketing_strategy_tool" and "product_analysis" not in args:
                self._validate_prerequisites(state, name)
                result = marketing_strategy_tool(state)
                self._merge(state, name, result)
            elif name == "content_generation_tool" and "product_analysis" not in args:
                self._validate_prerequisites(state, name)
                result = content_generation_tool(state)
                self._merge(state, name, result)
            elif name == "content_review_tool" and "content" not in args:
                self._validate_prerequisites(state, name)
                result = content_review_tool(state)
                self._merge(state, name, result)
            else:
                if name in {
                    "marketing_strategy_tool",
                    "content_generation_tool",
                    "content_review_tool",
                }:
                    self._validate_prerequisites(state, name)
                if name == "marketing_strategy_tool":
                    args = {
                        **args,
                        "product_analysis": _jsonable(state.product_analysis),
                        "retrieved_knowledge": _jsonable(state.retrieved_knowledge),
                        "marketing_goal": state.marketing_goal or "推广",
                    }
                elif name == "content_generation_tool":
                    args = {
                        **args,
                        "product_analysis": _jsonable(state.product_analysis),
                        "marketing_strategy": _jsonable(state.marketing_strategy) or {},
                        "retrieved_knowledge": _jsonable(state.retrieved_knowledge),
                        "platform": state.platform or "小红书",
                        "marketing_goal": state.marketing_goal or "推广",
                        "correction_notes": list(state.correction_notes),
                    }
                elif name == "content_review_tool":
                    args = {
                        **args,
                        "content": _jsonable(state.generated_content),
                        "product_analysis": _jsonable(state.product_analysis) or {},
                        "confirmed_features": str(state.product_info.get("confirmed_features", "")),
                        "usage_scenarios": str(state.product_info.get("usage_scenarios", "")),
                        "task_type": state.task_type.value,
                        "retrieved_knowledge": _jsonable(state.retrieved_knowledge),
                    }
                result = tool_map[name].invoke(args)
            if name == "analyze_product":
                if not isinstance(result, dict):
                    raise TypeError("analyze_product 必须返回结构化字典")
                analysis = ProductAnalysis.model_validate({
                    field: result[field]
                    for field in ProductAnalysis.model_fields
                })
                self._merge(state, name, {
                    "product_analysis": analysis,
                    "target_user": list(analysis.target_audience),
                    "platform": state.product_info.get(
                        "platform",
                        state.product_info.get("平台", state.platform or "小红书"),
                    ),
                    "marketing_goal": state.product_info.get(
                        "marketing_goal",
                        state.product_info.get("运营目标", state.marketing_goal or "推广"),
                    ),
                })
            elif name == "knowledge_retrieval_tool" and "documents" in result:
                self._merge(state, name, {
                    "query": result["query"],
                    "retrieved_knowledge": result["documents"],
                })
            elif name == "marketing_strategy_tool" and "product_positioning" in result:
                strategy = MarketingStrategy.model_validate({
                    field: result[field]
                    for field in MarketingStrategy.model_fields
                })
                self._merge(state, name, {"marketing_strategy": strategy})
            elif name == "content_generation_tool" and "title" in result:
                content = XiaohongshuContent.model_validate({
                    field: result[field]
                    for field in XiaohongshuContent.model_fields
                })
                self._merge(state, name, {"generated_content": content})
            elif name == "content_review_tool" and "passed" in result:
                self._merge(state, name, {
                    "review_result": ContentReview.model_validate(result),
                })
            result_text = str(result)
            print(f"[TOOL RESULT]\n{result_text[:6000]}")
            self._event(state, "tool_result", tool=name, result=result_text[:12000], status="success")
            print(f"[AGENT STATE]\n{_json_text(self._state_context(state))[:7000]}")
            return ToolMessage(content=result_text, tool_call_id=call_id, name=name)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            state.errors.append(error)
            self._event(state, "tool_error", tool=name, arguments=args, error=error)
            result_text = _json_text({"error": error, "agent_state": self._state_context(state)})
            print(f"[TOOL RESULT]\n{result_text[:6000]}")
            return ToolMessage(
                content=result_text,
                tool_call_id=call_id,
                name=name or "unknown_tool",
                status="error",
            )

    def _finish(self, state: AgentState, message: Any) -> AgentState:
        answer = getattr(message, "content", "")
        if not isinstance(answer, str):
            answer = _json_text(answer)
        answer = answer.strip()
        state.final_result = {
            "answer": answer,
            "task_type": state.task_type.value,
            "iteration_count": state.iteration_count,
            "status": (
                "needs_human_review"
                if state.review_result is not None and not state.review_result.passed
                else "completed"
            ),
        }
        self._event(state, "final", answer=answer, status=state.final_result["status"])
        print(f"[FINAL]\n{answer}")
        return state

    def run(self, state: AgentState) -> AgentState:
        """Run a genuine AIMessage.tool_calls -> ToolMessage -> LLM loop."""
        self.understand_task(state)
        if state.task_type == TaskType.CONTENT_REVIEW and state.product_info.get("draft_content"):
            state.generated_content = state.product_info["draft_content"]

        print(f"[AGENT]\n用户任务：{state.task}")
        print(f"[AGENT STATE]\n{_json_text(self._state_context(state))[:7000]}")
        tool_map = {item.name: item for item in self._create_tools(state)}
        model = self.model or create_chat_model()
        bound_model = model.bind_tools(list(tool_map.values()))
        messages = self._initial_messages(state)
        tool_calls_used = 0
        max_decisions = self.max_tool_calls + 3
        decision_count = 0

        try:
            for _ in range(max_decisions):
                active_model = bound_model
                if tool_calls_used >= self.max_tool_calls:
                    messages.append(SystemMessage(content=(
                        "The Python tool-call budget is exhausted. Do not request tools; "
                        "provide the best grounded final answer and state any incomplete work."
                    )))
                    active_model = model
                decision = active_model.invoke(messages)
                if not isinstance(decision, AIMessage):
                    raise TypeError("模型没有返回 LangChain AIMessage，无法确认 Tool Calling 协议")
                decision_label = "LLM DECISION" if decision_count == 0 else "LLM NEXT DECISION"
                decision_detail = _decision_text(decision)
                print(f"[{decision_label}]\n{decision_detail}")
                self._event(
                    state,
                    "llm_decision",
                    tool_calls=getattr(decision, "tool_calls", []),
                    content=getattr(decision, "content", ""),
                )
                if decision_count > 0:
                    self._event(
                        state,
                        "llm_next_decision",
                        decision=decision_detail,
                        tool_calls_used=tool_calls_used,
                        iteration_count=state.iteration_count,
                    )
                decision_count += 1
                messages.append(decision)
                calls = getattr(decision, "tool_calls", []) or []
                if not calls:
                    return self._finish(state, decision)

                tool_messages: list[ToolMessage] = []
                for call in calls:
                    remaining = self.max_tool_calls - tool_calls_used
                    if remaining > 0:
                        tool_calls_used += 1
                        state.tool_call_count = tool_calls_used
                    tool_messages.append(self._invoke_tool(
                        state, tool_map, call, remaining_calls=remaining
                    ))
                messages.extend(tool_messages)

            raise RuntimeError("超过最大 LLM 决策轮数，Agent 已停止")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            state.errors.append(error)
            self._event(state, "agent_error", error=error)
            raise


def run_agent(
    task: str,
    product_info: dict[str, Any],
    *,
    target_user: list[str] | None = None,
    platform: str = "",
    marketing_goal: str = "",
    max_iterations: int = MAX_ITERATIONS,
    max_tool_calls: int = MAX_TOOL_CALLS,
) -> AgentState:
    """Convenience API used by Streamlit, evaluation, and demo scripts."""
    state = AgentState(
        task=task,
        product_info=dict(product_info),
        target_user=list(target_user or []),
        platform=platform,
        marketing_goal=marketing_goal,
    )
    return EcommerceAgent(
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
    ).run(state)
