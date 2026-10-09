"""Search stored documents using exact cosine distance."""

from __future__ import annotations

import argparse

from pgvector.psycopg import register_vector

from vector_learning.db import connect
from vector_learning.embed import embed_texts


def search(query: str, limit: int = 3) -> list[tuple]:
    query_embedding = embed_texts([query])[0]

    with connect() as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, document_key, content, metadata,
                       embedding <=> %s AS cosine_distance
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            return cursor.fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the closest stored text embeddings.")
    parser.add_argument("query", help="Question or text to search for")
    parser.add_argument("--limit", type=int, default=3, help="Number of matches to show (default: 3)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    matches = search(args.query, args.limit)
    if not matches:
        print("No documents found. Run vector-ingest first.")
        return

    print(f"Query: {args.query}\n")
    for rank, (row_id, key, content, metadata, distance) in enumerate(matches, start=1):
        print(f"{rank}. {key} (id={row_id}, cosine distance={distance:.4f}, similarity={1 - distance:.4f})")
        print(f"   {content}")
        print(f"   metadata: {metadata}")


if __name__ == "__main__":
    main()
