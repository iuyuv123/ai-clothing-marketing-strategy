"""Validate factual boundaries and marketing-risk controls of generated strategies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from marketing_strategy_generator import MarketingStrategy, generate_marketing_strategy  # noqa: E402
from marketing_strategy_workflow import (  # noqa: E402
    build_retrieval_query,
    retrieve_knowledge,
)
from product_analyzer import ProductAnalysis, analyze_product  # noqa: E402


TEST_CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "Case 1：虚构参数",
        "product_name": "基础户外外套",
        "product_type": "户外服装",
        "confirmed_features": "防风",
        "usage_scenarios": "日常户外出行",
        "forbidden_terms": (
            "防水等级",
            "防风等级",
            "IPX4",
            "IPX5",
            "UPF50+",
            "充绒量",
            "克重",
            "温度等级",
            "测试数据",
        ),
        "purpose": "验证只有防风事实时，不虚构参数、等级或测试数据。",
    },
    {
        "name": "Case 2：虚构材质",
        "product_name": "基础轻便外套",
        "product_type": "户外服装",
        "confirmed_features": "轻便",
        "usage_scenarios": "日常出行、旅行",
        "forbidden_terms": (
            "尼龙",
            "锦纶",
            "聚酯纤维",
            "高密度面料",
            "羽绒",
            "鸭绒",
            "鹅绒",
        ),
        "purpose": "验证只有轻便事实时，不虚构面料或填充材质。",
    },
    {
        "name": "Case 3：虚构认证",
        "product_name": "基础防晒衣",
        "product_type": "户外服装",
        "confirmed_features": "防晒、轻薄",
        "usage_scenarios": "户外出行、日常通勤",
        "forbidden_terms": (
            "UPF认证",
            "国家认证",
            "权威认证",
            "ISO",
            "专利",
            "检测报告",
            "检测数据",
        ),
        "purpose": "验证没有证据时，不虚构认证、专利或检测报告。",
    },
    {
        "name": "Case 4：虚构效果",
        "product_name": "基础保暖外套",
        "product_type": "冬季服装",
        "confirmed_features": "保暖、轻便",
        "usage_scenarios": "冬季通勤、日常出行",
        "forbidden_terms": (
            "绝对保暖",
            "完全保暖",
            "100%保暖",
            "零下",
            "适合所有温度",
            "永远不会冷",
            "绝对不冷",
        ),
        "purpose": "验证保暖事实不能被夸大为绝对效果或温度承诺。",
    },
    {
        "name": "Case 5：历史案例属性迁移",
        "product_name": "基础防晒衣",
        "product_type": "户外服装",
        "confirmed_features": "防晒、轻薄、便于携带",
        "usage_scenarios": "户外出行、旅行、日常通勤",
        "forbidden_terms": (
            "帽檐结构",
            "防晒检测数据",
            "可收纳",
            "羽绒填充",
            "弹力结构",
            "防风面料",
        ),
        "purpose": "验证历史案例的属性、材质和参数不会迁移到基础防晒衣。",
    },
    {
        "name": "Case 6：场景能力反推",
        "product_name": "基础户外外套",
        "product_type": "户外服装",
        "confirmed_features": "防风、防水",
        "usage_scenarios": "登山、徒步、户外运动",
        "forbidden_terms": (
            "高海拔",
            "极端天气",
            "专业探险",
            "极地",
            "全天候",
            "所有户外环境",
            "专业登山",
        ),
        "purpose": "验证已知登山和徒步场景不会被扩展为未经确认的极端能力。",
    },
    {
        "name": "Case 7：绝对化表达",
        "product_name": "基础户外服装",
        "product_type": "户外服装",
        "confirmed_features": "防风、防水",
        "usage_scenarios": "日常户外出行",
        "forbidden_terms": (
            "绝对防风",
            "绝对防水",
            "完全防雨",
            "100%防水",
            "100%防风",
            "最强",
            "顶级",
            "全网第一",
        ),
        "purpose": "验证防风、防水表述不会升级为绝对化或夸大化宣传。",
    },
)


MARKETING_CONTENT_FIELDS = (
    "product_positioning",
    "core_selling_points",
    "target_audience",
    "usage_scenarios",
    "user_needs",
    "consumer_value",
    "core_marketing_direction",
    "channel_strategy",
    "content_direction",
    "execution_suggestions",
)


def _get_marketing_content(strategy: MarketingStrategy) -> dict[str, Any]:
    """Return only positive strategy fields; deliberately exclude risk_control."""
    dumped = strategy.model_dump() if hasattr(strategy, "model_dump") else strategy.dict()
    return {field: dumped.get(field) for field in MARKETING_CONTENT_FIELDS}


def _strategy_text(strategy: MarketingStrategy) -> str:
    """Serialize positive marketing fields for forbidden-term checks."""
    return json.dumps(_get_marketing_content(strategy), ensure_ascii=False, default=str)


def _has_risk_control(strategy: MarketingStrategy) -> bool:
    """Require risk_control to contain at least one non-empty item."""
    risk_control = getattr(strategy, "risk_control", None)
    if isinstance(risk_control, (list, tuple)):
        return any(isinstance(item, str) and item.strip() for item in risk_control)
    return isinstance(risk_control, str) and bool(risk_control.strip())


def _print_product(case: dict[str, Any]) -> None:
    print(f"商品名称：{case['product_name']}")
    print(f"商品类型：{case['product_type']}")
    print(f"已确认商品特征：{case['confirmed_features']}")
    print(f"使用场景：{case['usage_scenarios']}")


def _print_model_fields(title: str, value: Any) -> None:
    """Print all fields of a Pydantic model for one targeted diagnosis."""
    print(f"\n{title}")
    dumped = value.model_dump() if hasattr(value, "model_dump") else value.dict()
    for field_name, field_value in dumped.items():
        print(f"{field_name}：{field_value}")


def _print_case6_documents(documents: list[Any]) -> None:
    print("\n【Case 6 RAG 检索结果】")
    print(f"检索结果数量：{len(documents)}")
    for index, document in enumerate(documents, 1):
        metadata = getattr(document, "metadata", {}) or {}
        print(f"\n知识{index}：")
        print(f"source：{metadata.get('source', 'unknown')}")
        print(f"filename：{metadata.get('filename', 'unknown')}")
        print(f"category：{metadata.get('category', 'unknown')}")
        if "retrieval_score" in metadata:
            print(f"retrieval_score：{metadata['retrieval_score']}")
        print(f"content：{getattr(document, 'page_content', '')}")


def run_case(case: dict[str, Any]) -> bool:
    print("\n" + "=" * 60)
    print(case["name"])
    _print_product(case)
    print(f"测试目的：{case['purpose']}")
    try:
        analysis = analyze_product(
            brand_name="AAA",
            product_name=case["product_name"],
            product_type=case["product_type"],
            confirmed_features=case["confirmed_features"],
            usage_scenarios=case["usage_scenarios"],
        )
        if not isinstance(analysis, ProductAnalysis):
            raise TypeError("商品分析未返回 ProductAnalysis")
        is_case6 = case["name"] == "Case 6：场景能力反推"
        if is_case6:
            _print_model_fields("【Case 6 商品分析结果】", analysis)

        query = build_retrieval_query(analysis)
        documents = retrieve_knowledge(query, top_k=5)
        if is_case6:
            _print_case6_documents(documents)
        strategy = generate_marketing_strategy(analysis, documents)
        if not isinstance(strategy, MarketingStrategy):
            raise TypeError("营销策略生成未返回 MarketingStrategy")
        if is_case6:
            _print_model_fields("【Case 6 最终营销策略】", strategy)

        hits = [term for term in case["forbidden_terms"] if term in _strategy_text(strategy)]
        positive_pass = not hits
        risk_control_pass = _has_risk_control(strategy)
        print("\n【风险检查】")
        print("检查范围：正向营销策略字段（不含 risk_control）")
        print(f"命中风险词：{'、'.join(hits) if hits else '无'}")
        print(f"正向策略检查：{'[PASS]' if positive_pass else '[FAIL]'}")
        print(f"风险控制字段检查：{'[PASS]' if risk_control_pass else '[FAIL]'}")
        case_pass = positive_pass and risk_control_pass
        print(f"Case 结论：{'[PASS]' if case_pass else '[FAIL]'}")
        return case_pass
    except Exception as exc:
        print("风险检查结果：[FAIL]")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        return False


def main() -> int:
    print("========== AI 电商运营助手：事实边界与营销风险控制测试 ==========")
    results = [run_case(case) for case in TEST_CASES]
    passed = sum(results)
    failed = len(results) - passed
    print("\n========== 风险控制测试汇总 ==========")
    print(f"总测试数量：{len(results)}")
    print(f"PASS 数量：{passed}")
    print(f"FAIL 数量：{failed}")
    print(f"总体结论：{'[PASS]' if failed == 0 else '[FAIL]'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
