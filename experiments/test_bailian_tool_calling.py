"""Minimal live diagnostic for Bailian ChatOpenAI tool calling.

This standalone script makes up to three live LLM requests: a basic request, a
tool-call decision, and a final response after the ToolMessage round trip.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.llm_factory import create_chat_model


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}：晴天，25℃"


def _heading(title: str) -> None:
    print(f"\n========== {title} ==========")


def _print_exception(exc: BaseException) -> None:
    print(f"Exception type: {type(exc).__name__}")
    print(f"Exception message: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)


def _print_summary(
    basic_status: str,
    tool_calling_status: str,
    round_trip_status: str,
) -> None:
    _heading("Summary")
    print(f"百炼连接测试：{basic_status}")
    print(f"REAL TOOL CALLING：{tool_calling_status}")
    print(f"Tool Round Trip：{round_trip_status}")


def _tool_call_details(tool_calls: list[dict[str, Any]]) -> str:
    return json.dumps(tool_calls, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    basic_status = "NOT TESTED"
    tool_calling_status = "NOT TESTED"
    round_trip_status = "NOT TESTED"

    _heading("Test 1: Basic LLM")
    try:
        model = create_chat_model()
        basic_response = model.invoke([
            HumanMessage(content="请只回复：百炼连接测试成功"),
        ])
        if not isinstance(basic_response, AIMessage):
            raise TypeError(
                "基础请求未返回 LangChain AIMessage："
                f"{type(basic_response).__name__}"
            )
        print(f"Response: {basic_response.content}")
        basic_status = "PASS"
    except Exception as exc:
        basic_status = "FAIL"
        _print_exception(exc)
        _heading("Test 2: Real Tool Calling")
        print("Skipped because Test 1 failed.")
        print("REAL TOOL CALLING: NOT TESTED")
        _print_summary(basic_status, tool_calling_status, round_trip_status)
        return 1

    _heading("Test 2: Real Tool Calling")
    weather_request = HumanMessage(
        content=(
            "请查询北京今天的天气。"
            "必须调用 get_weather 工具，不要直接回答天气。"
        )
    )
    try:
        bound_model = model.bind_tools([get_weather])
        tool_call_response = bound_model.invoke([weather_request])
        if not isinstance(tool_call_response, AIMessage):
            raise TypeError(
                "Tool Calling 请求未返回 LangChain AIMessage："
                f"{type(tool_call_response).__name__}"
            )
    except Exception as exc:
        tool_calling_status = "FAIL"
        _print_exception(exc)
        print("REAL TOOL CALLING: FAIL")
        _print_summary(basic_status, tool_calling_status, round_trip_status)
        return 1

    tool_calls = list(tool_call_response.tool_calls or [])
    print("Tool calls:")
    print(_tool_call_details(tool_calls))
    if not tool_calls:
        tool_calling_status = "FAIL"
        print("REAL TOOL CALLING: FAIL")
        _print_summary(basic_status, tool_calling_status, round_trip_status)
        return 1

    tool_calling_status = "PASS"
    print("REAL TOOL CALLING: PASS")

    _heading("Test 3: Tool Round Trip")
    try:
        available_tools = {get_weather.name: get_weather}
        tool_messages: list[ToolMessage] = []
        for call in tool_calls:
            tool_name = str(call.get("name", ""))
            tool_args = call.get("args", {})
            tool_call_id = str(call.get("id", ""))
            if tool_name not in available_tools:
                raise ValueError(f"模型请求了未绑定的 Tool：{tool_name!r}")
            if not isinstance(tool_args, dict):
                raise TypeError(f"Tool args 必须是字典，实际为：{type(tool_args).__name__}")
            if not tool_call_id:
                raise ValueError("Tool call 缺少 id，无法创建对应 ToolMessage")

            result = available_tools[tool_name].invoke(tool_args)
            print(f"Tool name: {tool_name}")
            print(f"Tool args: {json.dumps(tool_args, ensure_ascii=False, default=str)}")
            print(f"Tool call id: {tool_call_id}")
            print(f"Tool result: {result}")
            tool_messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tool_name,
            ))

        message_history = [weather_request, tool_call_response, *tool_messages]
        final_response = bound_model.invoke(message_history)
        if not isinstance(final_response, AIMessage):
            raise TypeError(
                "Tool Round Trip 未返回 LangChain AIMessage："
                f"{type(final_response).__name__}"
            )
        if final_response.tool_calls:
            raise RuntimeError(
                "ToolMessage 回传后模型仍请求 Tool，未得到最终文本响应："
                f"{_tool_call_details(list(final_response.tool_calls))}"
            )
        print(f"Final response: {final_response.content}")
        round_trip_status = "PASS"
    except Exception as exc:
        round_trip_status = "FAIL"
        _print_exception(exc)
        _print_summary(basic_status, tool_calling_status, round_trip_status)
        return 1

    _print_summary(basic_status, tool_calling_status, round_trip_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
