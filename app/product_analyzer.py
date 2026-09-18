"""Runtime product understanding for the AI clothing marketing assistant."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProductAnalysis(BaseModel):
    """Structured facts and analysis produced for the current runtime product."""

    brand_name: str
    product_name: str
    product_type: str
    confirmed_features: list[str]
    core_selling_points: list[str]
    target_audience: list[str]
    usage_scenarios: list[str]
    user_needs: list[str]
    consumer_value: list[str]

    @field_validator("brand_name", "product_name", "product_type")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("品牌名称、商品名称和商品类型不能为空")
        return value.strip()

    @field_validator(
        "confirmed_features",
        "core_selling_points",
        "target_audience",
        "usage_scenarios",
        "user_needs",
        "consumer_value",
    )
    @classmethod
    def required_items(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("商品分析列表字段必须包含实际内容")
        return [item.strip() for item in value]


def parse_response(content: str) -> ProductAnalysis:
    """Parse and validate a model response as ``ProductAnalysis``."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return ProductAnalysis.model_validate(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型返回不是有效 JSON：{exc.msg}") from exc
    except ValidationError as exc:
        raise ValueError(f"模型返回格式校验失败：{exc}") from exc


def _build_prompt(
    brand_name: str,
    product_name: str,
    product_type: str,
    confirmed_features: str,
    usage_scenarios: str,
    product_details: str,
) -> str:
    return f"""你是“AI 服装营销策略助手”的商品理解与营销要素分析模块。

你的任务不是直接写营销文案，也不是直接生成完整营销方案，而是：
1. 理解当前商品；
2. 提取已确认商品事实；
3. 归纳核心卖点；
4. 根据当前商品动态分析目标人群；
5. 分析使用场景；
6. 分析用户需求；
7. 分析消费者价值。

必须严格区分：
- 商品事实不等于推测属性；
- 商品卖点不等于消费者利益；
- 消费者利益不等于营销策略。

当前商品信息（运行时输入）：
品牌名称：{brand_name}
商品名称：{product_name}
商品类型：{product_type}
已确认商品事实：{confirmed_features}
已知使用场景：{usage_scenarios or "未提供，请基于已确认事实做保守分析"}
商品详细资料：{product_details or "未提供"}

数据优先级与用途：
1. 当前商品事实包括品牌名称、商品名称、商品类型、已确认商品事实、已知使用场景和商品详细资料。
2. 商品详细资料中的材质、规格、设计细节、参数等已确认信息，可以作为当前商品事实参与分析。
3. AI 只能基于这些当前商品事实分析目标人群、使用场景、用户需求和消费者价值。
4. RAG 或历史案例只能提供营销方法、场景营销方法、用户需求分析方法、渠道方法、案例方法和
   风险规范，不得覆盖或新增当前商品事实。

分析规则：
1. confirmed_features 只能提取用户在“已确认商品事实”中明确提供的卖点，不得自行补充。
2. core_selling_points 可以归纳已确认商品事实和商品详细资料中已确认信息的营销价值，但不得创造新的卖点。
3. target_audience 必须根据当前商品特点和场景动态分析，不得要求用户预先提供，也不得复制历史案例人群。
4. usage_scenarios 优先使用用户提供的场景；未提供时只能基于已确认事实做合理、保守的场景推断，不得把场景推断写成产品性能。
5. user_needs 必须从商品特点、使用场景和动态目标人群推导。
6. consumer_value 必须对应用户需求，使用保守、可验证的消费者价值表述。
7. 用户没有提供的信息不能自行补充，不得推断未提供的参数、材质、认证、功能、效果或性能。
8. 不得因为某个使用场景而反向推导商品具备未提供的性能，不得把 RAG 知识当作当前商品事实。
9. 本模块不输出最终营销方向、渠道策略、内容策略或执行方案。
10. 只输出有效 JSON，不要 Markdown、代码围栏或额外解释。

JSON 必须严格包含以下字段，所有字段都必须有实际内容：
{{
  "brand_name": "",
  "product_name": "",
  "product_type": "",
  "confirmed_features": [],
  "core_selling_points": [],
  "target_audience": [],
  "usage_scenarios": [],
  "user_needs": [],
  "consumer_value": []
}}
"""


def _configured_llm() -> ChatOpenAI:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    model = os.getenv("DASHSCOPE_MODEL", "").strip().strip("\"'“”‘’")
    if not all((api_key, base_url, model)):
        raise RuntimeError(
            "DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL、DASHSCOPE_MODEL 未完整配置"
        )
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model)


def _validate_input(name: str, value: str, required: bool = True) -> str:
    value = value.strip() if isinstance(value, str) else ""
    if required and not value:
        raise ValueError(f"{name}不能为空")
    return value


def analyze_product(
    brand_name: str,
    product_name: str,
    product_type: str,
    confirmed_features: str,
    usage_scenarios: str = "",
    product_details: str = "",
) -> ProductAnalysis:
    """Analyze one runtime product and return structured marketing elements."""
    brand_name = _validate_input("品牌名称", brand_name)
    product_name = _validate_input("商品名称", product_name)
    product_type = _validate_input("商品类型", product_type)
    confirmed_features = _validate_input("已确认商品事实", confirmed_features)
    usage_scenarios = _validate_input("已知使用场景", usage_scenarios, required=False)
    product_details = _validate_input("商品详细资料", product_details, required=False)

    response = _configured_llm().invoke(
        _build_prompt(
            brand_name,
            product_name,
            product_type,
            confirmed_features,
            usage_scenarios,
            product_details,
        )
    )
    return parse_response(str(response.content))


def _print_items(label: str, value: Any) -> None:
    print(f"\n========== {label} ==========")
    if isinstance(value, list):
        for index, item in enumerate(value, 1):
            print(f"{index}. {item}")
    else:
        print(value)


def main() -> int:
    print("========== AI 服装营销策略助手：商品理解 ==========")
    brand_name = input("品牌名称：").strip()
    product_name = input("商品名称：").strip()
    product_type = input("商品类型：").strip()
    confirmed_features = input("已确认商品事实/卖点：").strip()
    usage_scenarios = input("已知使用场景（可选）：").strip()

    try:
        result = analyze_product(
            brand_name=brand_name,
            product_name=product_name,
            product_type=product_type,
            confirmed_features=confirmed_features,
            usage_scenarios=usage_scenarios,
        )
    except Exception as exc:
        print(f"商品分析失败：{type(exc).__name__}: {exc}")
        return 1

    _print_items("品牌", result.brand_name)
    _print_items("商品名称", result.product_name)
    _print_items("商品类型", result.product_type)
    _print_items("已确认商品事实", result.confirmed_features)
    _print_items("核心卖点", result.core_selling_points)
    _print_items("目标人群", result.target_audience)
    _print_items("使用场景", result.usage_scenarios)
    _print_items("用户需求", result.user_needs)
    _print_items("消费者价值", result.consumer_value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
