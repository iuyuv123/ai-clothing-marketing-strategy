"""Step 10 acceptance test for the complete product-to-strategy workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import product_analyzer  # noqa: E402
from marketing_strategy_workflow import (  # noqa: E402
    MarketingStrategy,
    ProductAnalysis,
    build_rag_context,
    build_retrieval_query,
    generate_marketing_strategy,
    retrieve_knowledge,
)


PRODUCT_INPUT = {
    "品牌名称": "AAA",
    "商品名称": "户外三合一冲锋衣",
    "商品类型": "户外服装",
    "已确认商品特征": "三合一设计、防风、防水",
    "使用场景": "登山、徒步、户外运动",
}

ANALYSIS_FIELDS = (
    "brand_name",
    "product_name",
    "product_type",
    "confirmed_features",
    "core_selling_points",
    "target_audience",
    "usage_scenarios",
    "user_needs",
    "consumer_value",
)

STRATEGY_FIELDS = (
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
    "risk_control",
)

POSITIVE_FIELDS = STRATEGY_FIELDS[:-1]

FORBIDDEN_POSITIVE_TERMS = (
    "透气",
    "保暖",
    "耐磨",
    "速干",
    "轻量化",
    "羽绒",
    "锦纶",
    "聚酯纤维",
    "UPF",
    "专利",
    "认证",
    "防水等级",
    "防风等级",
    "高海拔",
    "极端天气",
    "专业探险",
    "极地",
    "全天候",
    "所有户外环境",
)


def _dump_model(value: Any) -> dict[str, Any]:
    dumped = value.model_dump() if hasattr(value, "model_dump") else value.dict()
    if not isinstance(dumped, dict):
        raise TypeError("结构化对象无法转换为字典")
    return dumped


def _non_empty(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value) and all(str(item).strip() for item in value)
    return bool(str(value).strip())


def _print_status(label: str, passed: bool) -> None:
    print(f"{label}：{'[PASS]' if passed else '[FAIL]'}")


def _print_analysis(analysis: ProductAnalysis) -> None:
    data = _dump_model(analysis)
    print("\n【2. 商品分析】")
    _print_status("ProductAnalysis", isinstance(analysis, ProductAnalysis))
    for field in ANALYSIS_FIELDS:
        print(f"{field}：{data.get(field)}")


def _validate_documents(documents: list[Any]) -> tuple[bool, str]:
    if not documents:
        return False, "RAG 检索结果数量为 0"
    for index, document in enumerate(documents, 1):
        if not hasattr(document, "page_content"):
            return False, f"结果 {index} 缺少 page_content"
        metadata = getattr(document, "metadata", None)
        if not isinstance(metadata, dict):
            return False, f"结果 {index} metadata 不是字典"
        missing = [field for field in ("source", "filename", "category") if not metadata.get(field)]
        if missing:
            return False, f"结果 {index} 缺少 metadata：{', '.join(missing)}"
    return True, "文档对象和 metadata 完整"


def _print_documents(documents: list[Any]) -> None:
    print("\n【4. RAG 检索】")
    print(f"检索数量：{len(documents)}")
    for index, document in enumerate(documents, 1):
        metadata = document.metadata
        print(f"\n知识{index}：")
        print(f"source：{metadata.get('source')}")
        print(f"filename：{metadata.get('filename')}")
        print(f"category：{metadata.get('category')}")
        print(f"content：{document.page_content[:300]}")


def _print_strategy(strategy: MarketingStrategy) -> None:
    print("\n【9. 最终营销策略】")
    for field, value in _dump_model(strategy).items():
        print(f"{field}：{value}")


def _check_strategy_fields(strategy: MarketingStrategy) -> tuple[bool, dict[str, bool]]:
    data = _dump_model(strategy)
    checks = {field: field in data and _non_empty(data[field]) for field in STRATEGY_FIELDS}
    return all(checks.values()), checks


def _check_fact_boundary(strategy: MarketingStrategy) -> tuple[bool, list[str]]:
    data = _dump_model(strategy)
    positive_text = json.dumps(
        {field: data.get(field) for field in POSITIVE_FIELDS},
        ensure_ascii=False,
        default=str,
    )
    hits = [term for term in FORBIDDEN_POSITIVE_TERMS if term in positive_text]
    return not hits, hits


def main() -> int:
    print("========== Step 10：产品完整链路验收 ==========")
    print("\n【1. 商品输入】")
    for label, value in PRODUCT_INPUT.items():
        print(f"{label}：{value}")

    checks = {
        "商品分析": False,
        "RAG 查询": False,
        "RAG 检索": False,
        "RAG Context": False,
        "策略生成": False,
        "结构化字段": False,
        "事实边界": False,
        "风险控制": False,
    }

    try:
        analysis = product_analyzer.analyze_product(
            brand_name=PRODUCT_INPUT["品牌名称"],
            product_name=PRODUCT_INPUT["商品名称"],
            product_type=PRODUCT_INPUT["商品类型"],
            confirmed_features=PRODUCT_INPUT["已确认商品特征"],
            usage_scenarios=PRODUCT_INPUT["使用场景"],
        )
        analysis_ok = isinstance(analysis, ProductAnalysis) and all(
            field in _dump_model(analysis) for field in ANALYSIS_FIELDS
        )
        checks["商品分析"] = analysis_ok
        _print_analysis(analysis)

        query = build_retrieval_query(analysis)
        query_ok = isinstance(query, str) and bool(query.strip())
        checks["RAG 查询"] = query_ok
        print("\n【3. 动态 RAG 查询】")
        _print_status("RAG Query", query_ok)
        print(f"query：{query}")

        documents = retrieve_knowledge(query, top_k=5)
        retrieval_ok, retrieval_reason = _validate_documents(documents)
        checks["RAG 检索"] = retrieval_ok
        _print_documents(documents)
        print(f"RAG Retrieval：{'[PASS]' if retrieval_ok else '[FAIL]'}（{retrieval_reason}）")

        context = build_rag_context(analysis, documents)
        context_ok = (
            isinstance(context, str)
            and bool(context.strip())
            and "【当前商品分析】" in context
            and "【RAG参考知识】" in context
        )
        checks["RAG Context"] = context_ok
        print("\n【5. RAG Context】")
        _print_status("RAG Context", context_ok)
        print(f"Context 长度：{len(context) if isinstance(context, str) else 0}")

        strategy = generate_marketing_strategy(analysis, documents)
        strategy_ok = isinstance(strategy, MarketingStrategy)
        checks["策略生成"] = strategy_ok
        print("\n【6. 营销策略生成】")
        _print_status("MarketingStrategy", strategy_ok)

        if not strategy_ok:
            raise TypeError("营销策略生成未返回 MarketingStrategy")
        fields_ok, field_checks = _check_strategy_fields(strategy)
        checks["结构化字段"] = fields_ok
        print("\n【7. 结构化字段】")
        for field in STRATEGY_FIELDS:
            _print_status(field, field_checks[field])

        boundary_ok, hits = _check_fact_boundary(strategy)
        checks["事实边界"] = boundary_ok
        risk_control_ok = _non_empty(_dump_model(strategy).get("risk_control"))
        checks["风险控制"] = risk_control_ok
        print("\n【8. 事实边界】")
        print("检查正向营销字段（不含 risk_control）")
        print(f"命中风险词：{'、'.join(hits) if hits else '无'}")
        _print_status("事实边界", boundary_ok)
        _print_status("风险控制", risk_control_ok)
        _print_strategy(strategy)
    except Exception as exc:
        print("\n[ERROR] 完整链路执行失败")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")

    print("\n========== Step 10 验收汇总 ==========")
    for label, passed in checks.items():
        _print_status(label, passed)
    overall = all(checks.values())
    print(f"总体结论：{'[PASS]' if overall else '[FAIL]'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
