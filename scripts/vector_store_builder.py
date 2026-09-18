import os
from pathlib import Path

import dashscope
import chromadb
from dashscope import TextEmbedding
from dotenv import load_dotenv
from langchain_chroma import Chroma

from app.knowledge_loader import load_and_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "vector_store" / "marketing_strategy_kb"
COLLECTION_NAME = "marketing_strategy_kb"
ALLOWED_CATEGORIES = {"brand", "marketing_rules", "marketing_cases", "platform_rules"}
EMBED_BATCH_SIZE = 10
DOCUMENT_BATCH_SIZE = 5


class DashScopeEmbeddings:
    """LangChain-compatible DashScope embedding wrapper."""

    def __init__(self, model: str):
        self.model = model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = TextEmbedding.call(model=self.model, input=texts)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.message}")

        items = response.output.get("embeddings", [])
        vectors = [
            item["embedding"]
            for item in sorted(items, key=lambda item: item.get("text_index", 0))
        ]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding 返回数量与输入数量不一致")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + EMBED_BATCH_SIZE]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def _required_environment() -> tuple[str, str]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    if not model:
        raise RuntimeError("DASHSCOPE_EMBEDDING_MODEL 未设置")
    return api_key, model


def _validate_metadata(chunks) -> None:
    required = {"source", "category", "filename"}
    for index, chunk in enumerate(chunks, 1):
        if not required.issubset(chunk.metadata):
            raise RuntimeError(f"Chunk {index} 缺少 metadata：{required - set(chunk.metadata)}")
        if chunk.metadata["category"] not in ALLOWED_CATEGORIES:
            raise RuntimeError(f"发现不允许的知识库分类：{chunk.metadata['category']}")


def _inspect_existing_store(expected_count: int) -> None:
    """Protect complete/partial stores while allowing an empty store to rebuild."""
    if not PERSIST_DIR.exists():
        return

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME not in collection_names:
        print("检测到现有 Chroma 目录，但目标 collection 不存在，允许重新构建。")
        return

    count = client.get_collection(COLLECTION_NAME).count()
    if count == 0:
        print("检测到空向量库：0 个向量，删除空 collection 后允许重新构建。")
        client.delete_collection(COLLECTION_NAME)
        return
    if count < expected_count:
        raise RuntimeError(
            f"检测到半成品向量库：{count}/{expected_count}，"
            "建议先删除失败现场后重新构建。"
        )
    if count == expected_count:
        raise RuntimeError(f"检测到完整向量库：{count}/{expected_count}，已拒绝覆盖。")
    raise RuntimeError(
        f"检测到异常向量库：{count} 个向量，超过当前文本块数量 {expected_count}，已拒绝覆盖。"
    )


def build_vector_store() -> int:
    api_key, embedding_model = _required_environment()
    dashscope.api_key = api_key

    _, document_count, chunks = load_and_split()
    if not chunks:
        raise RuntimeError("知识库没有可写入的文本块")
    _validate_metadata(chunks)
    _inspect_existing_store(len(chunks))

    embedding_function = DashScopeEmbeddings(embedding_model)
    print("========== Step 7.3 向量库构建 ==========")
    print(f"原始文档数量：{document_count}")
    print(f"文本块数量：{len(chunks)}")
    print("Embedding 初始化成功")
    print("开始写入向量库")

    vector_store = None
    written_count = 0
    try:
        for start in range(0, len(chunks), DOCUMENT_BATCH_SIZE):
            batch = chunks[start : start + DOCUMENT_BATCH_SIZE]
            if vector_store is None:
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_function,
                    collection_name=COLLECTION_NAME,
                    persist_directory=str(PERSIST_DIR),
                )
            else:
                vector_store.add_documents(batch)
            written_count += len(batch)
            print(f"已写入 {written_count}/{len(chunks)}")
    except Exception:
        print(f"写入异常，当前已经成功写入：{written_count}/{len(chunks)}")
        raise

    if vector_store is None:
        raise RuntimeError("没有创建向量库实例")

    stored_count = vector_store._collection.count()
    search_results = vector_store.similarity_search("防风", k=3)
    print("========== 构建完成 ==========")
    print(f"向量数量：{stored_count}")
    print(f"检索到的文档数量：{len(search_results)}")
    for index, document in enumerate(search_results, 1):
        print(f"检索结果 {index}：{document.metadata.get('source', '')}")
        print(f"内容：{document.page_content[:120]}")

    records = vector_store.get(include=["documents", "metadatas"], limit=3)
    all_metadata = vector_store.get(include=["metadatas"])["metadatas"]
    required = {"source", "category", "filename"}
    metadata_complete = all(required.issubset(item or {}) for item in all_metadata)

    print(f"Python：{os.sys.executable}")
    print(f"Embedding模型：{embedding_model}")
    print("向量数据库：Chroma")
    print(f"向量库路径：{PERSIST_DIR}")
    print(f"原始文档数量：{document_count}")
    print(f"文本块数量：{len(chunks)}")
    print(f"成功写入向量数量：{stored_count}")

    for index, (document, metadata) in enumerate(
        zip(records.get("documents", []), records.get("metadatas", [])), 1
    ):
        print(f"\n---------- Vector {index} ----------")
        print(f"来源：{metadata.get('source', '')}")
        print(f"类别：{metadata.get('category', '')}")
        print(f"文件：{metadata.get('filename', '')}")
        print(f"内容：\n{document}")

    if stored_count != len(chunks):
        raise RuntimeError(f"向量数量不一致：文本块 {len(chunks)}，向量 {stored_count}")
    if not metadata_complete:
        raise RuntimeError("存在 metadata 不完整的向量")
    if len(records.get("documents", [])) != 3:
        raise RuntimeError("新向量库读取验证未返回 3 条 Document")

    print("\n向量数量校验：通过")
    print("metadata 完整性校验：通过")
    print("新向量库读取校验：通过")
    print("向量数据库构建成功")
    print("========== Step 7.3 验收 ==========")
    print(f"原始文档：{document_count}")
    print(f"文本块：{len(chunks)}")
    print(f"向量数量：{stored_count}")
    print("检索测试：成功")
    print("Step 7.3：PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(build_vector_store())
    except Exception as exc:
        print("========== Step 7.3 向量库构建失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"完整错误信息：{exc}")
        raise

