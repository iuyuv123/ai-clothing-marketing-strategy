"""Executable end-to-end demo and evaluation for AI E-commerce Operations Agent V2.

This script intentionally runs only the required single case.  It does not
resume or modify the paused 30x2 experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.ecommerce_agent import run_agent
from app.agent.agent_state import AgentState
from app.evaluation_metrics import (
    evaluate_compliance,
    evaluate_content_fact_consistency,
    evaluate_fact_boundary,
    evaluate_iteration_efficiency,
    evaluate_retrieval_quality,
    evaluate_task_completion,
    evaluate_tool_selection,
)


PRODUCT = {
    "brand_name": "AAA",
    "product_name": "户外三合一冲锋衣",
    "product_type": "户外服装",
    "confirmed_features": "三合一设计、防风、防水",
    "usage_scenarios": "登山、徒步、户外运动",
    "product_details": "",
    "platform": "小红书",
    "marketing_goal": "推广",
}
TASK = "帮我为这款冲锋衣制定一套小红书推广方案。"


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if hasattr(value, "page_content"):
        return {
            "page_content": value.page_content,
            "metadata": getattr(value, "metadata", {}) or {},
        }
    return value


def _retrieval_fields(document: Any) -> tuple[str, str]:
    value = _dump(document)
    if not isinstance(value, dict):
        return "unknown", str(value or "")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    source = str(value.get("source") or metadata.get("source") or "unknown")
    content = str(value.get("content") or value.get("page_content") or "")
    return source, content


def _latest_retrieval_top_k(state: AgentState) -> int:
    for event in reversed(state.events):
        if event.get("type") != "tool_call" or event.get("tool") != "knowledge_retrieval_tool":
            continue
        arguments = event.get("arguments") or {}
        if isinstance(arguments, dict):
            top_k = arguments.get("top_k")
            if isinstance(top_k, int) and top_k > 0:
                return top_k
    raise RuntimeError("无法从 Agent 事件中确定最后一次知识检索使用的 top_k")


def run_demo() -> tuple[AgentState, dict[str, Any]]:
    print("========== AI 电商运营 Agent V2 完整案例 ==========")
    print(f"任务：{TASK}")
    print(f"商品事实：{PRODUCT['confirmed_features']}")
    print(f"场景：{PRODUCT['usage_scenarios']}")
    state = run_agent(TASK, PRODUCT, platform="小红书", marketing_goal="推广")
    print("\n【Agent任务理解】")
    print(state.task_type.value)
    print("\n【Tool调用顺序】")
    print(" → ".join(state.tool_trace))
    print("\n【RAG检索结果】")
    for index, document in enumerate(state.retrieved_knowledge, 1):
        source, page_content = _retrieval_fields(document)
        print(f"{index}. {source} | {page_content[:180]}")
    print("\n【营销策略】")
    print(json.dumps(_dump(state.marketing_strategy), ensure_ascii=False, indent=2))
    print("\n【内容生成】")
    print(json.dumps(_dump(state.generated_content), ensure_ascii=False, indent=2))
    print("\n【审核结果】")
    print(json.dumps(_dump(state.review_result), ensure_ascii=False, indent=2))
    print(f"iteration_count：{state.iteration_count}")
    print("\n【最终结果】")
    print(json.dumps(_dump(state.final_result), ensure_ascii=False, indent=2))

    strategy = state.marketing_strategy
    content = state.generated_content
    review = state.review_result
    retrieval_top_k = _latest_retrieval_top_k(state)
    evaluation = {
        "tool_selection": evaluate_tool_selection(state.tool_trace, TASK),
        "retrieval_quality": evaluate_retrieval_quality(
            state.retrieved_knowledge,
            top_k=retrieval_top_k,
        ),
        "fact_consistency": {
            "passed": False,
            "strategy": evaluate_fact_boundary(strategy, PRODUCT["confirmed_features"]),
            "content": evaluate_content_fact_consistency(content, PRODUCT["confirmed_features"]),
        },
        "compliance": evaluate_compliance(content),
        "task_completion": evaluate_task_completion(
            strategy,
            content,
            platform=state.platform,
            target_user=state.target_user,
            marketing_goal=state.marketing_goal,
        ),
        "iteration_efficiency": evaluate_iteration_efficiency(
            state.iteration_count, bool(getattr(review, "passed", False))
        ),
    }
    evaluation["fact_consistency"]["passed"] = (
        evaluation["fact_consistency"]["strategy"]["passed"]
        and evaluation["fact_consistency"]["content"]["passed"]
    )
    report = {
        "title": "AI 电商运营 Agent V2 Evaluation Report",
        "test_task": TASK,
        "tool_trace": list(state.tool_trace),
        "retrieval_summary": {
            "count": len(state.retrieved_knowledge),
            "sources": [
                _retrieval_fields(document)[0]
                for document in state.retrieved_knowledge
            ],
        },
        "generation_summary": {
            "has_strategy": strategy is not None,
            "has_content": content is not None,
            "content": _dump(content),
        },
        "review": _dump(review),
        "fact_boundary_violation": not evaluation["fact_consistency"]["passed"],
        "rule_risk": not evaluation["compliance"]["passed"],
        "iteration_count": state.iteration_count,
        "task_completed": evaluation["task_completion"]["passed"],
        "metrics": evaluation,
        "known_issues": [
            "当前 MVP 仅支持小红书内容生成；其他平台需要新增内容适配器。",
            "最终发布仍需业务人员依据真实商品资料和最新平台规则复核。",
        ],
        "next_steps": [
            "扩展多平台内容工具和平台规则版本管理",
            "增加人工反馈与转化数据闭环",
            "加入人机协同审批和多 Agent 角色分工",
        ],
    }
    report_path = PROJECT_ROOT / "experiments" / "agent_v2_evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEvaluation Report：{report_path}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return state, report


if __name__ == "__main__":
    run_demo()
