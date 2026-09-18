"""Generate a structured marketing strategy from runtime analysis and RAG context."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag_context import build_rag_context


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MarketingStrategy(BaseModel):
    product_positioning: str
    core_selling_points: list[str]
    target_audience: list[str]
    usage_scenarios: list[str]
    user_needs: list[str]
    consumer_value: list[str]
    core_marketing_direction: list[str]
    channel_strategy: list[str]
    content_direction: list[str]
    execution_suggestions: list[str]
    risk_control: list[str]


def _build_llm() -> ChatOpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    model = os.getenv("DASHSCOPE_MODEL", "").strip().strip("\"'“”‘’")
    if not all((api_key, base_url, model)):
        raise RuntimeError(
            "DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL、DASHSCOPE_MODEL 未完整配置"
        )
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model)


def _strategy_prompt(rag_context: str) -> str:
    return f"""你正在为“AI 服装营销策略助手”生成结构化营销策略。

输入包含两部分：
1. 【当前商品分析】：用户本次输入商品的实时信息，具有最高优先级，只能作为当前商品事实和分析依据。
2. 【RAG参考知识】：知识库中的可复用方法、策略、经验和框架，只能作为策略制定参考。

必须严格区分二者。历史营销案例只能借鉴分析方法、用户洞察方式、场景分析方式、营销思路、
策略结构、渠道选择思路和内容方向，不能直接复制其中的商品属性、参数、功能、材料、数据、
认证、专利、效果、性能指标或历史商品目标人群，除非这些信息同时出现在当前商品事实中。

target_audience 必须根据当前商品动态分析，不能简单复制历史案例目标人群。若参考知识没有适合
当前商品的人群，也不能强行匹配。不得补充当前商品未提供的材质、防水等级、防风等级、保暖等级、
透气性、重量、测试数据、认证、专利、技术参数、功能指标或效果。

事实边界规则（必须逐条遵守）：
1. product_name、product_type、confirmed_features、core_selling_points、target_audience、
   usage_scenarios、user_needs、consumer_value 只能基于【当前商品分析】中的用户输入事实和
   对这些事实的合理营销分析生成。每一项正向策略内容都必须能够追溯到当前商品明确事实，或
   是不增加新商品属性的合理营销表达。
2. “使用场景”不等于“商品能力证明”。用户提供“登山、徒步、户外运动”等场景，只说明计划
   使用场景，不能反向证明商品具备高海拔、极端天气、专业探险、极地环境、全天候使用、所有
   户外环境或专业登山能力。当前商品资料没有明确提供时，严禁把这些能力写入任何正向策略字段。
3. 严禁场景能力反推：不能因为某种场景通常需要某种性能，就推断当前商品拥有该性能。对场景
   的描述必须保持在用户明确提供的范围内，不得擅自扩大为更严苛或更专业的环境。
4. 属性事实边界同样适用：如果当前商品没有明确提供材质、参数、等级、测试数据、认证、专利、
   温度范围、性能指标或功能效果，不得自行补充、改写成营销卖点，或在正向策略中暗示其存在。
5. 如果 RAG 参考知识出现与当前商品不同的商品属性、材质、参数、等级、认证、性能或特定使用
   环境，必须将其视为参考案例信息，只能借鉴营销方法、策略结构、场景分析思路和内容组织方式，
   绝不能迁移到当前商品的正向营销策略。历史案例中的具体商品信息不能覆盖当前商品事实。
6. 正向营销策略字段不得出现未被当前商品事实支持的能力承诺、适用环境、效果保证或绝对化表达。
   不确定时应使用保守、可追溯的表述；无法追溯时宁可不写。risk_control 字段仍必须保留针对
   事实不足、证据缺失、历史案例迁移和夸大宣传的风险控制内容。

正向策略字段与风险控制字段的职责必须分离：
7. 以下字段是正向营销策略字段：product_positioning、core_selling_points、target_audience、
   usage_scenarios、user_needs、consumer_value、core_marketing_direction、channel_strategy、
   content_direction、execution_suggestions。它们只能描述当前商品的定位、已确认卖点、对应人群、
   已确认使用场景、用户需求、消费者价值、营销方向、渠道、内容和可执行建议。
8. risk_control 是唯一专门承载风险边界的字段。所有“禁止”“不得”“避免”“严禁”“不能宣传”以及
   对证据不足、属性迁移、场景扩展和夸大表达的提醒，优先集中写入 risk_control。
9. 正向营销策略字段中不要为了提醒风险而重复写入具体风险词或否定句。例如不要在 content_direction
   中写“避免专业探险、极端天气”，不要在 execution_suggestions 中写“不刻意营造高海拔、暴风、
   暴雨等极端环境”。这些边界应放入 risk_control；正向字段改用正向、可执行且事实可追溯的表达。
10. 正向表达应围绕当前商品已确认的信息，例如可以写“围绕登山、徒步、户外运动等已确认场景进行
    展示”，或“说明已确认的功能、适用场景和使用边界”；不得为了删除风险词而扩展成高海拔、极端
    天气、专业探险、极地、全天候或其他未经确认的能力。
11. risk_control 中提到某个风险词是正常且必要的；只有当正向营销策略字段把未经确认的能力、属性
    或环境作为当前商品能力进行描述时，才构成事实越界。请同时保留清晰、具体的 risk_control 内容。

只返回合法 JSON，禁止 Markdown、```json、解释文字或前后额外说明。JSON 必须严格包含以下字段：
{{
  "product_positioning": "",
  "core_selling_points": [],
  "target_audience": [],
  "usage_scenarios": [],
  "user_needs": [],
  "consumer_value": [],
  "core_marketing_direction": [],
  "channel_strategy": [],
  "content_direction": [],
  "execution_suggestions": [],
  "risk_control": []
}}

以下是待分析上下文：
{rag_context}
"""


def generate_marketing_strategy(
    product_analysis: Any,
    retrieved_documents: Any,
) -> MarketingStrategy:
    """Generate and validate a marketing strategy from runtime RAG context."""
    rag_context = build_rag_context(product_analysis, retrieved_documents)
    response = _build_llm().invoke(_strategy_prompt(rag_context))
    try:
        data = json.loads(response.content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("营销策略模型返回结果不是合法 JSON") from exc
    return MarketingStrategy.model_validate(data)
