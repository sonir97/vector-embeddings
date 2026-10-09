"""Database connection and schema setup for the learning project."""

from __future__ import annotations

import os
from importlib.resources import files

import psycopg

DEFAULT_DATABASE_URL = "postgresql://vectors:vectors@localhost:5433/vectors"


def connect() -> psycopg.Connection:
    """Connect to PostgreSQL using DATABASE_URL or the local Compose default."""
    return psycopg.connect(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))


def initialize_schema() -> None:
    """Enable pgvector and create the documents table."""
    schema = files("vector_learning.resources").joinpath("002-create-documents.sql").read_text(encoding="utf-8")
    with connect() as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(schema)
    print("pgvector is enabled and the documents table is ready.")


def main() -> None:
    initialize_schema()


if __name__ == "__main__":
    main()
