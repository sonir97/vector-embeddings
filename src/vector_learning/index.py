"""Create the approximate cosine HNSW index used in Phase 5."""

from __future__ import annotations

from vector_learning.db import connect


INDEX_SQL = """
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_cosine_idx
ON documents USING hnsw (embedding vector_cosine_ops)
"""


def create_index() -> None:
    with connect() as connection:
        connection.execute(INDEX_SQL)
    print("HNSW cosine index is ready: documents_embedding_hnsw_cosine_idx")


def main() -> None:
    create_index()


if __name__ == "__main__":
    main()
