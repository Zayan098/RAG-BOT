"""Build the vector store from the scraped book content.

Reads book_content/*.md + manifest.json, splits each page on markdown
headers then recursively into 800-char chunks, embeds them locally with
sentence-transformers, and upserts everything into a persistent ChromaDB
collection (cosine space, normalized vectors).

Usage:
    python ingest.py             # upsert only new/missing chunks
    python ingest.py --rebuild   # wipe the collection and rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer

CONTENT_DIR = Path(__file__).parent / "book_content"
MANIFEST_PATH = CONTENT_DIR / "manifest.json"
DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "book"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBED_BATCH_SIZE = 64
ADD_BATCH_SIZE = 500

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=HEADERS_TO_SPLIT_ON
)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        print("ERROR: book_content/manifest.json not found. Run scrape_book.py first.")
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def split_page(markdown: str) -> list[Document]:
    header_docs = header_splitter.split_text(markdown)
    chunks: list[Document] = []
    for doc in header_docs:
        content = doc.page_content.strip()
        if not content:
            continue
        h2 = doc.metadata.get("H2", "")
        h3 = doc.metadata.get("H3", "")
        sub_headers = " > ".join(h for h in (h3, h2) if h)
        if sub_headers:
            content = f"{sub_headers}\n\n{content}"
        for split in recursive_splitter.split_text(content):
            split = split.strip()
            if split:
                chunks.append(Document(page_content=split, metadata=doc.metadata))
    return chunks


def build_chunks(manifest: dict[str, dict]) -> list[tuple[str, str, dict]]:
    all_chunks: list[tuple[str, str, dict]] = []
    for slug, meta in sorted(manifest.items()):
        page_path = CONTENT_DIR / f"{slug}.md"
        if not page_path.exists():
            print(f"  ! missing file for slug {slug}, skipping")
            continue
        markdown = page_path.read_text(encoding="utf-8")
        chunks = split_page(markdown)
        section = meta.get("section") or meta.get("title", "")
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "url": meta["url"],
                "title": meta.get("title", ""),
                "section": section,
                "source": slug,
            }
            all_chunks.append((f"{slug}#{i}", chunk.page_content, chunk_meta))
        print(f"  {slug}: {len(chunks)} chunks")
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ChromaDB vector store.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the collection")
    args = parser.parse_args()

    manifest = load_manifest()
    chunks = build_chunks(manifest)
    print(f"\nTotal chunks: {len(chunks)}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    if args.rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Removed existing collection.")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(EMBEDDING_MODEL)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings_buf: list = []
    total_upserted = 0
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        batch_ids = [c[0] for c in batch]
        batch_texts = [c[1] for c in batch]
        batch_metas = [c[2] for c in batch]
        vectors = model.encode(
            batch_texts,
            normalize_embeddings=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True,
        )
        ids.extend(batch_ids)
        documents.extend(batch_texts)
        metadatas.extend(batch_metas)
        embeddings_buf.extend(vectors)
        if len(ids) >= ADD_BATCH_SIZE or i + EMBED_BATCH_SIZE >= len(chunks):
            collection.upsert(
                ids=ids,
                embeddings=embeddings_buf,
                documents=documents,
                metadatas=metadatas,
            )
            total_upserted += len(ids)
            print(f"  upserted {total_upserted}/{len(chunks)} chunks")
            ids, documents, metadatas, embeddings_buf = [], [], [], []

    print(f"\nCollection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()