"""Ingest UTF-8 text and Markdown files as searchable text chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

from pgvector.psycopg import register_vector

from vector_learning.db import connect
from vector_learning.embed import embed_texts
from vector_learning.ingest import upsert_documents


SUPPORTED_SUFFIXES = {".md", ".txt"}


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_words])
        if chunk:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def collect_documents(
    corpus_root: Path,
    namespace: str,
    chunk_words: int,
    overlap_words: int,
) -> tuple[list[dict], list[str]]:
    documents = []
    source_paths = []

    files = sorted(
        path for path in corpus_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in files:
        relative_path = path.relative_to(corpus_root).as_posix()
        source_paths.append(relative_path)
        text = path.read_text(encoding="utf-8")
        for chunk_index, chunk in enumerate(chunk_text(text, chunk_words, overlap_words), start=1):
            documents.append(
                {
                    "document_key": f"{namespace}/{relative_path}#chunk-{chunk_index:04d}",
                    "content": chunk,
                    "metadata": {
                        "source_namespace": namespace,
                        "source_path": relative_path,
                        "chunk_index": chunk_index,
                        "file_type": path.suffix.lower(),
                    },
                }
            )

    return documents, source_paths


def ingest_files(corpus_root: Path, namespace: str, chunk_words: int, overlap_words: int) -> None:
    documents, source_paths = collect_documents(corpus_root, namespace, chunk_words, overlap_words)
    if not documents:
        print("No non-empty Markdown or text files found; the database was not changed.")
        return

    embeddings = embed_texts([document["content"] for document in documents])
    document_keys = [document["document_key"] for document in documents]

    with connect() as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            # Remove obsolete chunks and deleted files only from this ingestion namespace.
            cursor.execute(
                """
                DELETE FROM documents
                WHERE metadata ->> 'source_namespace' = %s
                  AND NOT (document_key = ANY(%s))
                """,
                (namespace, document_keys),
            )
            upsert_documents(cursor, documents, embeddings)

    print(
        f"Ingested {len(documents)} chunks from {len(source_paths)} files "
        f"into namespace '{namespace}'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed UTF-8 .txt and .md files and store them as searchable chunks."
    )
    parser.add_argument("directory", type=Path, help="Folder containing files to ingest")
    parser.add_argument("--namespace", help="Stable source namespace (defaults to the folder name)")
    parser.add_argument("--chunk-words", type=int, default=100, help="Words per chunk (default: 100)")
    parser.add_argument("--overlap-words", type=int, default=20, help="Overlap between chunks (default: 20)")
    args = parser.parse_args()

    if args.chunk_words < 1:
        parser.error("--chunk-words must be at least 1")
    if args.overlap_words < 0 or args.overlap_words >= args.chunk_words:
        parser.error("--overlap-words must be at least 0 and less than --chunk-words")

    corpus_root = args.directory.resolve()
    if not corpus_root.is_dir():
        parser.error(f"directory does not exist: {args.directory}")

    namespace = args.namespace or corpus_root.name
    ingest_files(corpus_root, namespace, args.chunk_words, args.overlap_words)


if __name__ == "__main__":
    main()
