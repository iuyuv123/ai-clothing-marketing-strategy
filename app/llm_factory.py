"""Shared ChatOpenAI construction for application LLM calls."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_chat_model() -> ChatOpenAI:
    """Create the configured chat model used by the application."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    model = os.getenv("DASHSCOPE_MODEL", "").strip().strip("\"'“”‘’")
    if not all((api_key, base_url, model)):
        raise RuntimeError(
            "DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL、DASHSCOPE_MODEL 未完整配置"
        )
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model)
