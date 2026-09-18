import os
from pathlib import Path

import dashscope
from dashscope import TextEmbedding
from dotenv import load_dotenv
from langchain_chroma import Chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = PROJECT_ROOT / "vector_store" / "marketing_strategy_kb"
COLLECTION_NAME = "marketing_strategy_kb"
QUESTIONS = [
    "冲锋衣适合哪些户外使用场景？",
    "如何根据商品特点分析用户需求？",
    "制定服装营销方案时应该经过哪些步骤？",
]


class DashScopeEmbeddings:
    """Minimal LangChain-compatible wrapper for query embeddings."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
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

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _embedding_model() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "").strip().strip("\"'“”‘’")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    if not model:
        raise RuntimeError("DASHSCOPE_EMBEDDING_MODEL 未设置")
    dashscope.api_key = api_key
    return model


def run_retriever_test() -> int:
    if not PERSIST_DIR.is_dir():
        raise FileNotFoundError(f"正式向量库目录不存在：{PERSIST_DIR}")

    model = _embedding_model()
    embeddings = DashScopeEmbeddings(model=model)
    db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )
    vector_count = db._collection.count()
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    print("========== Step 7.4 Retriever 检索测试 ==========")
    print(f"向量库：{COLLECTION_NAME}")
    print(f"向量数量：{vector_count}")
    print(f"Embedding模型：{model}")

    for question_index, question in enumerate(QUESTIONS, 1):
        results = retriever.invoke(question)
        print(f"\n【测试问题 {question_index}】")
        print(question)
        for result_index, document in enumerate(results, 1):
            source = document.metadata.get("source", "")
            content = document.page_content.strip()
            print(f"\n【检索结果 {result_index}】")
            print(f"source：{source}")
            print(f"content：{content[:200]}")

        if len(results) != 3:
            raise RuntimeError(
                f"问题 {question_index} 返回数量异常：{len(results)}，预期 3 条"
            )
        for result_index, document in enumerate(results, 1):
            if not document.metadata.get("source"):
                raise RuntimeError(f"问题 {question_index} 结果 {result_index} 缺少 source metadata")
            if not document.page_content.strip():
                raise RuntimeError(f"问题 {question_index} 结果 {result_index} content 为空")

    if vector_count != 37:
        raise RuntimeError(f"向量数量异常：{vector_count}，预期 37")

    print("\n========== Step 7.4 验收 ==========")
    print(f"向量库：{COLLECTION_NAME}")
    print(f"向量数量：{vector_count}")
    print("Retriever：成功")
    print(f"测试问题：{len(QUESTIONS)}")
    print("每题返回：3条")
    print("Source metadata：正常")
    print("Step 7.4：PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_retriever_test())
    except Exception as exc:
        print("========== Step 7.4 Retriever 检索失败 ==========")
        print(f"错误类型：{type(exc).__name__}")
        print(f"完整错误信息：{exc}")
        raise
