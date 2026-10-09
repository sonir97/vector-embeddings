# pgvector basics

pgvector is a PostgreSQL extension. It adds a vector data type and operators for comparing embeddings inside PostgreSQL. Enable it in each database with `CREATE EXTENSION vector`.

An embedding model converts a piece of text into a list of numbers. The all-MiniLM-L6-v2 model used in this project produces 384 numbers. Store the original text beside the vector so a search result can show the passage that matched.

Cosine similarity compares the direction of two vectors. pgvector's `<=>` operator returns cosine distance, which is one minus cosine similarity. Order by the smallest distance to get the nearest passages.

When a file is split into chunks, each chunk gets a stable key made from its relative path and chunk number. For example, `pgvector-basics.md#chunk-0001` identifies the first chunk in that file. If the file changes, ingestion updates chunks with matching keys and removes obsolete chunk keys in the same source namespace.
