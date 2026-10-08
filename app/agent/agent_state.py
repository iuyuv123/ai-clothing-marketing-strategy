"""Explicit state contract for the AI e-commerce operations agent.

The state is the only data shared between tools.  Tools return values; the
orchestrator merges those values into this object so that a tool cannot mutate
another tool's private variables.  ``iteration_count`` is owned by the review
loop and is capped by :data:`MAX_ITERATIONS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


MAX_ITERATIONS = 2

# Reviewable design artifact: type, ownership, nullability and lifetime for
# every core field required by the product specification.
STATE_FIELD_CONTRACT: dict[str, dict[str, Any]] = {
    "task": {"type": "str", "writer": "caller", "readers": ["planner", "all task tools"], "nullable": False, "lifecycle": "whole run"},
    "user_request": {"type": "str property", "writer": "caller", "readers": ["planner", "all task tools"], "nullable": False, "lifecycle": "alias of task"},
    "product_info": {"type": "dict[str, Any]", "writer": "caller/product_analysis", "readers": ["product_analysis", "knowledge_retrieval", "content_generation", "content_review"], "nullable": False, "lifecycle": "whole run"},
    "product_analysis": {"type": "Any | None property", "writer": "analyze_product", "readers": ["strategy", "content", "review"], "nullable": True, "lifecycle": "stored once inside product_info for compatibility"},
    "target_user": {"type": "list[str]", "writer": "product_analysis", "readers": ["marketing_strategy", "content_generation", "content_review"], "nullable": True, "lifecycle": "after analysis"},
    "platform": {"type": "str", "writer": "caller/planner", "readers": ["content_generation", "content_review"], "nullable": True, "lifecycle": "whole run"},
    "marketing_goal": {"type": "str", "writer": "caller/planner", "readers": ["marketing_strategy", "content_generation", "content_review"], "nullable": True, "lifecycle": "whole run"},
    "retrieved_knowledge": {"type": "list[Document]", "writer": "knowledge_retrieval", "readers": ["marketing_strategy", "content_generation", "content_review", "evaluation"], "nullable": True, "lifecycle": "after retrieval"},
    "marketing_strategy": {"type": "MarketingStrategy | None", "writer": "marketing_strategy", "readers": ["content_generation", "evaluation"], "nullable": True, "lifecycle": "plan tasks only"},
    "generated_content": {"type": "XiaohongshuContent | dict | str | None", "writer": "content_generation/caller", "readers": ["content_review", "evaluation"], "nullable": True, "lifecycle": "content tasks only"},
    "review_result": {"type": "ContentReview | None", "writer": "content_review", "readers": ["planner", "content_generation", "evaluation"], "nullable": True, "lifecycle": "after review"},
    "iteration_count": {"type": "int", "writer": "review loop", "readers": ["planner", "evaluation"], "nullable": False, "lifecycle": "whole run; 0..2"},
    "tool_call_count": {"type": "int", "writer": "orchestrator", "readers": ["planner", "evaluation", "UI"], "nullable": False, "lifecycle": "whole run; 0..MAX_TOOL_CALLS"},
    "final_result": {"type": "Any | None", "writer": "orchestrator", "readers": ["caller", "evaluation"], "nullable": True, "lifecycle": "terminal"},
}


class TaskType(StrEnum):
    """Supported task branches selected from the user's natural-language task."""

    PRODUCT_QA = "product_qa"
    PROMOTION_PLAN = "promotion_plan"
    CONTENT_REVIEW = "content_review"
    UNKNOWN = "unknown"


@dataclass
class AgentState:
    """Runtime state passed through the agent decision graph.

    Field contract:
      - ``task``: user task, read by the planner and every task-specific tool;
        never empty after construction; lifetime: whole run.
      - ``product_info``: runtime product facts and optional draft content,
        written at construction; read by product/RAG/content/review tools;
        never empty for product tasks; whole run.
      - ``target_user``, ``platform``, ``marketing_goal``: planner defaults
        plus product/task analysis; written by the product/plan steps; read by
        strategy, content and review; platform may be empty for product Q&A.
      - ``retrieved_knowledge``: Chroma documents written by the RAG tool and
        read by strategy/content/review/evaluation; empty is allowed when the
        store has no usable results; lifetime: whole run.
      - ``marketing_strategy``: structured strategy written by the strategy
        tool; optional for product Q&A and content-review tasks.
      - ``generated_content``: structured Xiaohongshu content written by the
        content tool; optional unless a promotion plan is requested.
      - ``review_result``: structured review written by the review tool;
        optional before review and required before a plan task ends.
      - ``iteration_count``: number of correction attempts, written only by the
        bounded review loop; starts at 0 and can never exceed 2.
      - ``final_result``: terminal answer written by the orchestrator only.
      - ``tool_trace``/``events``/``errors``: observability data for the run.
    """

    task: str
    product_info: dict[str, Any] = field(default_factory=dict)
    target_user: list[str] = field(default_factory=list)
    platform: str = ""
    marketing_goal: str = ""
    task_type: TaskType = TaskType.UNKNOWN
    retrieved_knowledge: list[Any] = field(default_factory=list)
    marketing_strategy: Any | None = None
    generated_content: Any | None = None
    review_result: Any | None = None
    iteration_count: int = 0
    tool_call_count: int = 0
    final_result: Any | None = None
    tool_trace: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    correction_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task = str(self.task or "").strip()
        if not self.task:
            raise ValueError("task 不能为空")
        if self.iteration_count < 0 or self.iteration_count > MAX_ITERATIONS:
            raise ValueError(f"iteration_count 必须在 0 到 {MAX_ITERATIONS} 之间")

    def record_tool(self, name: str, result: Any | None = None) -> None:
        """Record a tool call without exposing or sharing tool internals."""
        self.tool_trace.append(name)
        event: dict[str, Any] = {"type": "tool_call", "tool": name}
        if result is not None:
            event["result"] = result
        self.events.append(event)

    @property
    def user_request(self) -> str:
        """Canonical request name without duplicating the existing ``task`` value."""
        return self.task

    @user_request.setter
    def user_request(self, value: str) -> None:
        value = str(value or "").strip()
        if not value:
            raise ValueError("user_request 不能为空")
        self.task = value

    @property
    def product_analysis(self) -> Any | None:
        """Expose the existing nested analysis as a first-class state property."""
        return self.product_info.get("product_analysis")

    @product_analysis.setter
    def product_analysis(self, value: Any | None) -> None:
        if value is None:
            self.product_info.pop("product_analysis", None)
        else:
            self.product_info["product_analysis"] = value

    def record_event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"type": event_type, **payload})

    def increment_iteration(self) -> int:
        """Increment once after a failed review; refuse an unbounded loop."""
        if self.iteration_count >= MAX_ITERATIONS:
            raise RuntimeError("已达到最大自动修正次数，停止继续循环")
        self.iteration_count += 1
        return self.iteration_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly state summary for logs and evaluation."""
        def dump(value: Any) -> Any:
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if isinstance(value, list):
                return [dump(item) for item in value]
            if isinstance(value, dict):
                return {str(key): dump(item) for key, item in value.items()}
            if hasattr(value, "page_content"):
                metadata = getattr(value, "metadata", {}) or {}
                return {"page_content": value.page_content, "metadata": metadata}
            return value

        product_info = dump(self.product_info)
        if isinstance(product_info, dict):
            product_info.pop("product_analysis", None)
        return {
            "task": self.task,
            "user_request": self.user_request,
            "product_info": product_info,
            "product_analysis": dump(self.product_analysis),
            "target_user": dump(self.target_user),
            "platform": self.platform,
            "marketing_goal": self.marketing_goal,
            "task_type": self.task_type.value,
            "retrieved_knowledge": dump(self.retrieved_knowledge),
            "marketing_strategy": dump(self.marketing_strategy),
            "generated_content": dump(self.generated_content),
            "review_result": dump(self.review_result),
            "iteration_count": self.iteration_count,
            "tool_call_count": self.tool_call_count,
            "final_result": dump(self.final_result),
            "tool_trace": list(self.tool_trace),
            "events": dump(self.events),
            "errors": list(self.errors),
            "correction_notes": list(self.correction_notes),
        }
