"""Shared DashScope embedding construction for application runtime code."""

from __future__ import annotations

import dashscope
from dashscope import TextEmbedding

from app.runtime_config import get_runtime_config


class DashScopeEmbeddings:
    """Minimal LangChain-compatible DashScope embedding adapter."""

    def __init__(self, model: str | None = None):
        config = get_runtime_config()
        dashscope.api_key = config.dashscope_api_key
        self.model = model or config.dashscope_embedding_model

    def _embed(self, texts: list[str]) -> list[list[float]]:
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
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def create_embedding_model() -> DashScopeEmbeddings:
    """Create the configured embedding model without making an API request."""
    return DashScopeEmbeddings()
