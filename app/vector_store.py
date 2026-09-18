import os
from collections import Counter
from pathlib import Path
from typing import Any

import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv
from langchain_chroma import Chroma

from knowledge_loader import load_and_split


load_dotenv()


def get_embedding_model() -> str:
    load_dotenv()
    value = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
    if not value:
        raise RuntimeError("DASHSCOPE_EMBEDDING_MODEL 未设置")
    return value


MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
COLLECTION = "ai_ecommerce_knowledge"
ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = ROOT / "vector_store" / "marketing_strategy_kb"


class DashScopeEmbeddings:
    def __init__(self, model: str | None = None):
        self.model = model or get_embedding_model()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = TextEmbedding.call(model=self.model, input=texts)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.message}")
        items = response.output.get("embeddings", [])
        vectors = [item["embedding"] for item in sorted(items, key=lambda item: item.get("text_index", 0))]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding 返回数量与输入数量不一致")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def _document_category(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    category = str(metadata.get("category") or "").strip()
    if category:
        return category
    source = str(metadata.get("source") or "").replace("\\", "/")
    for candidate in ("brand", "marketing_rules", "marketing_cases", "platform_rules"):
        if f"/{candidate}/" in f"/{source}/":
            return candidate
    return "other"


def select_diverse_documents(
    scored_documents: list[tuple[Any, float]],
    top_k: int = 5,
    score_tolerance: float = 0.15,
) -> list[tuple[Any, float]]:
    """Keep relevance first, then limit category concentration among close results."""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if score_tolerance < 0:
        raise ValueError("score_tolerance 不能小于 0")

    ranked = sorted(scored_documents, key=lambda item: item[1])
    if not ranked:
        return []

    # Chroma 返回 distance，数值越小越相关。只在原始 Top K 截止分数附近
    # 应用类别多样性，避免为了凑类别引入明显低相关的候选。
    cutoff_index = min(top_k, len(ranked)) - 1
    relevance_limit = ranked[cutoff_index][1] + score_tolerance
    selected: list[tuple[Any, float]] = []
    category_counts: Counter[str] = Counter()

    # 第一阶段：相关性合格的候选中，同一 category 优先最多保留两条。
    for document, score in ranked:
        category = _document_category(document)
        if score <= relevance_limit and category_counts[category] < 2:
            selected.append((document, score))
            category_counts[category] += 1
            if len(selected) == top_k:
                break

    # 第二阶段：类别不足或候选不足时，严格按原始相关性回填。
    # 因此类别多样性是软约束，不会牺牲结果数量或强塞低相关类别。
    if len(selected) < top_k:
        selected_documents = {id(document) for document, _ in selected}
        for document, score in ranked:
            if id(document) in selected_documents:
                continue
            selected.append((document, score))
            selected_documents.add(id(document))
            if len(selected) == top_k:
                break

    return sorted(selected, key=lambda item: item[1])


def retrieve_marketing_knowledge(
    vector_store: Chroma,
    query: str,
    top_k: int = 5,
    candidate_k: int | None = None,
) -> list[Any]:
    """Retrieve relevant candidates, then apply a soft category-diversity rule."""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    candidate_k = max(candidate_k or top_k * 2, top_k, 10)
    candidates = vector_store.similarity_search_with_score(query, k=candidate_k)
    selected = select_diverse_documents(candidates, top_k=top_k)

    documents: list[Any] = []
    for document, score in selected:
        document.metadata = {
            **(getattr(document, "metadata", {}) or {}),
            "retrieval_score": float(score),
            "retrieval_candidate_count": len(candidates),
        }
        documents.append(document)
    return documents


def product_retriever(vector_store: Chroma, product_name: str, top_k: int = 3):
    """构造先按 product_name 过滤、再做 similarity 的商品级 Retriever。"""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
            "filter": {"$and": [{"category": "product"}, {"product_name": product_name}]},
        },
    )


def general_retriever(vector_store: Chroma, top_k: int = 3):
    """检索非商品事实类知识，避免把其他商品 product Chunk 带入上下文。"""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k, "filter": {"category": {"$in": ["brand", "marketing_rules", "platform_rules", "excellent_copy"]}}},
    )


def main() -> int:
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("========== Chroma 创建失败 ==========")
        print("错误类型：配置错误")
        print("错误信息：DASHSCOPE_API_KEY 未设置")
        return 1

    try:
        dashscope.api_key = api_key
        _, _, chunks = load_and_split()
        if not chunks:
            raise RuntimeError("知识库没有可写入的 Chunk")

        embedding_model = get_embedding_model()
        embedding_function = DashScopeEmbeddings(embedding_model)
        existing = Chroma(
            collection_name=COLLECTION,
            persist_directory=str(PERSIST_DIR),
            embedding_function=embedding_function,
        )
        stored_count = existing._collection.count()
        metadata = existing._collection.get(limit=3, include=["metadatas"])["metadatas"]
        print("========== Chroma 向量数据库只读检查 ==========")
        print(f"知识库 Chunk 数量：{len(chunks)}")
        print(f"现有向量数量：{stored_count}")
        print(f"Collection：{COLLECTION}")
        print(f"Embedding 模型：{embedding_model}")
        print("向量维度：1024")
        print("持久化目录：vector_store/marketing_strategy_kb/")
        print("数据库写入：未执行（需在后续迁移阶段处理）")
        print("\n========== Metadata 验证 ==========")
        for item in metadata:
            print(f"category：{item.get('category', '')}")
            print(f"product_name：{item.get('product_name', '')}")
            print(f"source：{item.get('source', '')}")
        return 0
    except Exception as exc:
        print("========== Chroma 创建失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"错误信息：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

