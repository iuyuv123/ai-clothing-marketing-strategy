"""Safely add product_name metadata to historical product records."""

from __future__ import annotations

import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb

from app.vector_store import COLLECTION, PERSIST_DIR


ROOT = Path(__file__).resolve().parent.parent
BACKUP_PREFIX = "vector_store_backup_before_metadata_migration_"
PRODUCT_NAME_RE = re.compile(r"^\s*(?:商品名称|产品名称)\s*[：:]\s*(\S.*?)\s*$")


def infer_product_name(document: str, source: str) -> str | None:
    for line in document.splitlines():
        match = PRODUCT_NAME_RE.match(line)
        if match:
            return match.group(1).strip()
    if not source:
        return None
    stem = Path(source).stem
    for suffix in ("_商品资料", "_商品卖点"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.split("_", 1)[0].strip() or None


def snapshot(collection: Any) -> dict[str, Any]:
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = data.get("ids", [])
    return {
        "ids": list(ids),
        "documents": dict(zip(ids, data.get("documents", []))),
        "metadatas": dict(zip(ids, data.get("metadatas", []))),
        "embeddings": dict(zip(ids, data.get("embeddings", []))) if data.get("embeddings") is not None else None,
    }


def print_summary(title: str, state: dict[str, Any]) -> None:
    print(f"========== {title} ==========")
    print(f"Collection: {COLLECTION}")
    print(f"Vector count: {len(state['ids'])}")
    counts = Counter(metadata.get("category", "") for metadata in state["metadatas"].values())
    print("Category statistics:")
    for category in ("product", "brand", "marketing_rules", "platform_rules", "excellent_copy"):
        print(f"{category}: {counts.get(category, 0)}")
    print("Product records:")
    for record_id in state["ids"]:
        metadata = state["metadatas"][record_id]
        if metadata.get("category") == "product":
            print(f"id={record_id}")
            print(f"product_name={metadata.get('product_name', '')}")
            print(f"source={metadata.get('source', '')}")
            print(f"metadata={metadata}")


def validate(before: dict[str, Any], after: dict[str, Any], expected: dict[str, str]) -> None:
    if len(before["ids"]) != len(after["ids"]):
        raise RuntimeError("Vector count changed")
    if before["ids"] != after["ids"]:
        raise RuntimeError("IDs changed")
    if before["documents"] != after["documents"]:
        raise RuntimeError("Documents changed")
    if before["embeddings"] is not None and after["embeddings"] is not None:
        for record_id in before["ids"]:
            if list(before["embeddings"][record_id]) != list(after["embeddings"][record_id]):
                raise RuntimeError(f"Embedding changed for id={record_id}")
    for record_id in after["ids"]:
        metadata = after["metadatas"][record_id]
        original = before["metadatas"][record_id]
        if metadata.get("category") != "product":
            if "product_name" in metadata:
                raise RuntimeError(f"Non-product metadata changed for id={record_id}")
            if metadata != original:
                raise RuntimeError(f"Non-product metadata changed for id={record_id}")
            continue
        if metadata.get("product_name") != expected.get(record_id):
            raise RuntimeError(f"Incorrect product_name for id={record_id}")
        unchanged = {key: value for key, value in metadata.items() if key != "product_name"}
        if unchanged != original:
            raise RuntimeError(f"Existing metadata changed for id={record_id}")


def main() -> int:
    vector_path = Path(PERSIST_DIR)
    if not vector_path.is_dir():
        print(f"Migration failed: vector_store/marketing_strategy_kb 不存在：{vector_path}")
        return 1

    backup = ROOT / f"{BACKUP_PREFIX}{datetime.now():%Y%m%d_%H%M%S}"
    try:
        shutil.copytree(vector_path, backup)
        print(f"✓ Backup created: {backup}")
    except Exception as exc:
        print(f"Migration failed: backup 创建失败：{type(exc).__name__}: {exc}")
        return 1

    try:
        client = chromadb.PersistentClient(path=str(vector_path))
        collection = client.get_collection(COLLECTION)
        before = snapshot(collection)
        print_summary("Migration Before", before)
        expected: dict[str, str] = {}
        updates: list[tuple[str, dict[str, Any]]] = []
        unresolved: list[tuple[str, str]] = []
        for record_id in before["ids"]:
            metadata = before["metadatas"][record_id]
            if metadata.get("category") != "product":
                continue
            product_name = metadata.get("product_name") or infer_product_name(
                before["documents"][record_id], str(metadata.get("source", ""))
            )
            if not product_name:
                unresolved.append((record_id, str(metadata.get("source", ""))))
                continue
            expected[record_id] = product_name
            if metadata.get("product_name") != product_name:
                updated = dict(metadata)
                updated["product_name"] = product_name
                updates.append((record_id, updated))

        if unresolved:
            print("无法自动识别 product_name 的记录：")
            for record_id, source in unresolved:
                print(f"id={record_id} source={source}")
            raise RuntimeError("存在无法可靠推断 product_name 的记录，未执行自动更新")

        if updates:
            collection.update(ids=[record_id for record_id, _ in updates], metadatas=[metadata for _, metadata in updates])

        after = snapshot(collection)
        print_summary("Migration After", after)
        validate(before, after, expected)
        print("========== Migration Result ==========")
        print("✓ Backup created")
        print("✓ Metadata migration completed")
        print("✓ Vector count unchanged")
        print("✓ IDs unchanged")
        print("✓ Documents unchanged")
        print("✓ Embeddings unchanged: verified by metadata-only update")
        print("✓ Product metadata verified")
        print("Migration successful.")
        return 0
    except Exception as exc:
        print(f"Migration failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


