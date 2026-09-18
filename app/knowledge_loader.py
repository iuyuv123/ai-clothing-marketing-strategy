from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def _source_path(path: Path) -> str:
    """Return a portable project-relative source path for metadata."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_and_split() -> tuple[int, int, list]:
    """Load every knowledge-base TXT file and split its documents."""
    print("========== 知识库加载测试 ==========")

    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{BASE_DIR}")

    files = sorted(BASE_DIR.rglob("*.txt"))
    if not files:
        print("未找到知识库 TXT 文件")
        return 0, 0, []

    documents = []
    skipped_empty = []

    for path in files:
        try:
            if not path.read_text(encoding="utf-8").strip():
                skipped_empty.append(path)
                print(f"跳过空文件：{_source_path(path)}")
                continue

            category = path.relative_to(BASE_DIR).parts[0]
            metadata = {
                "source": _source_path(path),
                "category": category,
                "filename": path.name,
            }
            loaded_documents = TextLoader(str(path), encoding="utf-8").load()
            for document in loaded_documents:
                document.metadata.update(metadata)
            documents.extend(loaded_documents)
        except Exception as exc:
            print(f"加载失败：{_source_path(path)}，原因：{exc}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    print(f"知识库文件数量：{len(files)}")
    print(f"原始文档数量：{len(documents)}")
    print(f"切分后文本块数量：{len(chunks)}")

    if not chunks:
        print("加载成功，但切分后没有文本块")
        return len(files), len(documents), chunks

    for index, chunk in enumerate(chunks[:5], 1):
        print(f"\n---------- Chunk {index} ----------")
        print(f"来源：{chunk.metadata.get('source', '')}")
        print(f"类别：{chunk.metadata.get('category', '')}")
        print(f"文件：{chunk.metadata.get('filename', '')}")
        print(f"内容：\n{chunk.page_content}")

    return len(files), len(documents), chunks


if __name__ == "__main__":
    load_and_split()
