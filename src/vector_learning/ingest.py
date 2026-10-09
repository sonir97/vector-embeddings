"""Embed the example corpus and upsert it into PostgreSQL."""

from __future__ import annotations

import json
from importlib.resources import files

from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from vector_learning.db import connect
from vector_learning.embed import embed_texts


def upsert_documents(cursor, documents: list[dict], embeddings) -> None:
    for document, embedding in zip(documents, embeddings, strict=True):
        cursor.execute(
            """
            INSERT INTO documents (document_key, content, metadata, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (document_key) DO UPDATE
            SET content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
            """,
            (
                document["document_key"],
                document["content"],
                Jsonb(document.get("metadata", {})),
                embedding,
            ),
        )


def load_documents() -> list[dict]:
    corpus = files("vector_learning.resources").joinpath("documents.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in corpus.splitlines() if line.strip()]


def ingest_documents() -> None:
    documents = load_documents()
    embeddings = embed_texts([document["content"] for document in documents])

    with connect() as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            upsert_documents(cursor, documents, embeddings)

    print(f"Stored {len(documents)} documents with {embeddings.shape[1]}-dimensional embeddings.")


def main() -> None:
    ingest_documents()


if __name__ == "__main__":
    main()
