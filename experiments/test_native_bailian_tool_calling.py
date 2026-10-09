"""Minimal native OpenAI SDK diagnostic for Bailian tool calling."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from openai import OpenAI


BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-max"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def get_weather(city: str) -> str:
    return f"{city}：晴天，25℃"


def _json_text(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif isinstance(value, list):
        value = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in value
        ]
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _print_exception(exc: BaseException) -> None:
    print(f"Exception type: {type(exc).__name__}")
    print(f"Exception message: {exc}")
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)


def main() -> int:
    print("========== Native Tool Calling Test ==========\n")

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "请查询北京今天的天气。"
                "必须调用 get_weather 工具。"
                "不要直接回答天气。"
            ),
        }
    ]

    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("环境变量 DASHSCOPE_API_KEY 未设置或为空")

        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        if not response.choices:
            raise RuntimeError("百炼响应中没有 choices")
        message = response.choices[0].message
    except Exception as exc:
        print("Basic API: FAIL")
        _print_exception(exc)
        print("NATIVE TOOL CALLING: NOT TESTED")
        print("TOOL ROUND TRIP: NOT TESTED")
        return 1

    print("Basic API: PASS")
    print("\nFull response:")
    print(_json_text(response))
    print(f"\nmessage.content: {message.content!r}")
    print("message.tool_calls:")
    print(_json_text(message.tool_calls or []))

    tool_calls = list(message.tool_calls or [])
    if not tool_calls:
        print("\nNATIVE TOOL CALLING: FAIL")
        print("TOOL ROUND TRIP: NOT TESTED")
        return 1

    print("\nTool calls:")
    for tool_call in tool_calls:
        print(f"tool call name: {tool_call.function.name}")
        print(f"tool call arguments: {tool_call.function.arguments}")
        print(f"tool call id: {tool_call.id}")
    print("\nNATIVE TOOL CALLING: PASS")

    print("\n========== Tool Round Trip ==========\n")
    try:
        assistant_message = {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ],
        }
        messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name != "get_weather":
                raise ValueError(f"模型请求了未知 Tool：{function_name!r}")

            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise TypeError(
                    "Tool arguments 必须解析为 JSON 对象，实际为："
                    f"{type(arguments).__name__}"
                )
            city = arguments.get("city")
            if not isinstance(city, str) or not city.strip():
                raise ValueError("get_weather 的 city 参数缺失或无效")

            tool_result = get_weather(city.strip())
            print(f"Tool result: {tool_result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        if not final_response.choices:
            raise RuntimeError("Tool Round Trip 响应中没有 choices")
        final_message = final_response.choices[0].message

        print("\nFinal response:")
        print(_json_text(final_response))
        print(f"\nFinal answer: {final_message.content!r}")
        if final_message.tool_calls:
            raise RuntimeError(
                "Tool 结果回传后模型仍请求 Tool，未生成最终回答："
                f"{_json_text(final_message.tool_calls)}"
            )
    except Exception as exc:
        _print_exception(exc)
        print("TOOL ROUND TRIP: FAIL")
        return 1

    print("\nTOOL ROUND TRIP: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
