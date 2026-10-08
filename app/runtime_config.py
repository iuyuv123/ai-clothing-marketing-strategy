"""Centralized runtime configuration for the application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PERSIST_DIR = PROJECT_ROOT / "vector_store" / "marketing_strategy_kb"
CHROMA_COLLECTION = "marketing_strategy_kb"


def _clean(value: str | None) -> str:
    return (value or "").strip().strip("\"'“”‘’")


@dataclass(frozen=True)
class RuntimeConfig:
    dashscope_api_key: str
    dashscope_base_url: str
    dashscope_model: str
    dashscope_embedding_model: str
    chroma_persist_dir: Path
    chroma_collection: str


def get_runtime_config() -> RuntimeConfig:
    """Load and validate the settings needed by the current runtime."""
    load_dotenv(PROJECT_ROOT / ".env")
    values = {
        "DASHSCOPE_API_KEY": _clean(os.getenv("DASHSCOPE_API_KEY")),
        "DASHSCOPE_BASE_URL": _clean(os.getenv("DASHSCOPE_BASE_URL")),
        "DASHSCOPE_MODEL": _clean(os.getenv("DASHSCOPE_MODEL")),
        "DASHSCOPE_EMBEDDING_MODEL": _clean(os.getenv("DASHSCOPE_EMBEDDING_MODEL")),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"缺少必填配置：{', '.join(missing)}")
    return RuntimeConfig(
        dashscope_api_key=values["DASHSCOPE_API_KEY"],
        dashscope_base_url=values["DASHSCOPE_BASE_URL"],
        dashscope_model=values["DASHSCOPE_MODEL"],
        dashscope_embedding_model=values["DASHSCOPE_EMBEDDING_MODEL"],
        chroma_persist_dir=CHROMA_PERSIST_DIR,
        chroma_collection=CHROMA_COLLECTION,
    )
