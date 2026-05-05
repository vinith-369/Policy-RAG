import sys
from pathlib import Path

# Anchor all relative paths to the directory this script lives in.
_HERE = Path(__file__).resolve().parent

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.vector_store import upsert_chunks, collection_count
from app.agent import stream_rag

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def ingest_directory(docs_dir: str = "docs") -> int:
    base = (_HERE / docs_dir).resolve()
    if not base.exists():
        print(f"Directory '{base}' not found — nothing ingested.")
        return 0

    total = 0
    for filepath in sorted(list(base.glob("*.txt")) + list(base.glob("*.pdf"))):
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(filepath))
            raw = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            raw = filepath.read_text(encoding="utf-8", errors="replace")

        if not raw.strip():
            continue

        text_chunks = _splitter.split_text(raw)
        chunks = [
            {
                "text":        chunk,
                "source_file": filepath.name,
                "chunk_index": i,
                "category":    "policy",
            }
            for i, chunk in enumerate(text_chunks)
        ]
        n = upsert_chunks(chunks)
        print(f"   📄 {filepath.name} → {n} chunks")
        total += n

    return total


def main():
    print("🚀 Policy RAG CLI starting up...")
    print("📦 Loading embedding models (first run may download weights)...")

    existing = collection_count()
    if existing == 0:
        print("📥 No existing data found — ingesting documents from 'docs/'...")
        n = ingest_directory("docs")
        print(f"✅ Ingested → {n} chunks indexed\n")
    else:
        print(f"✅ Collection already contains {existing} chunks — skipping ingest\n")
        print("   💡 To re-ingest, delete the 'qdrant_data/' folder and restart.\n")

    print("Type 'exit' or 'quit' to stop.")
    print("-" * 50)

    while True:
        try:
            q = input("\n👤 Query: ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print("🤖 Thinking...", flush=True)

            meta = None
            first_token = True
            for event in stream_rag(question=q, top_k=5):
                if event["type"] == "token":
                    if first_token:
                        print("\n📝 Answer: ", end="", flush=True)
                        first_token = False
                    print(event["content"], end="", flush=True)

                elif event["type"] == "retry":
                    # Hallucination detected — wipe current line and restart
                    print(
                        f"\n⚠️  Hallucination detected (attempt {event['attempt']}) "
                        "— regenerating...",
                        flush=True,
                    )
                    first_token = True   # reset so the label prints again

                elif event["type"] == "done":
                    meta = event

            print()  # newline after streamed answer
            if meta:
                print(f"🔢 Chunks used: {meta['chunks_used']}")
                if meta["sources"]:
                    print("📚 Sources:", ", ".join(meta["sources"]))
                else:
                    print("📚 Sources: None")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()