from collections import Counter
from typing import Any

from langchain_chroma import Chroma

from app.embedding_factory import DashScopeEmbeddings, create_embedding_model
from app.runtime_config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR, get_runtime_config
try:  # package import from the agent/application entry points
    from app.knowledge_loader import load_and_split
except ModuleNotFoundError:  # legacy scripts import modules from app/ directly
    from knowledge_loader import load_and_split


def get_embedding_model() -> str:
    return get_runtime_config().dashscope_embedding_model


COLLECTION = CHROMA_COLLECTION
PERSIST_DIR = CHROMA_PERSIST_DIR


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
    try:
        config = get_runtime_config()
        _, _, chunks = load_and_split()
        if not chunks:
            raise RuntimeError("知识库没有可写入的 Chunk")

        embedding_model = config.dashscope_embedding_model
        embedding_function = create_embedding_model()
        existing = Chroma(
            collection_name=config.chroma_collection,
            persist_directory=str(config.chroma_persist_dir),
            embedding_function=embedding_function,
        )
        stored_count = existing._collection.count()
        metadata = existing._collection.get(limit=3, include=["metadatas"])["metadatas"]
        print("========== Chroma 向量数据库只读检查 ==========")
        print(f"知识库 Chunk 数量：{len(chunks)}")
        print(f"现有向量数量：{stored_count}")
        print(f"Collection：{config.chroma_collection}")
        print(f"Embedding 模型：{embedding_model}")
        print("向量维度：1024")
        print(f"持久化目录：{config.chroma_persist_dir}")
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

