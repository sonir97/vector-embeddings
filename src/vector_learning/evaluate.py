"""Compare exact cosine search with HNSW search on labeled sample queries."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from importlib.resources import files

from pgvector.psycopg import register_vector

from vector_learning.db import connect
from vector_learning.embed import embed_texts

SEARCH_SQL = """
SELECT document_key
FROM documents
ORDER BY embedding <=> %s
LIMIT %s
"""


def load_cases() -> list[dict]:
    lines = files("vector_learning.resources").joinpath("evaluation.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in lines.splitlines() if line.strip()]


def set_search_mode(connection, mode: str) -> None:
    if mode == "exact":
        connection.execute("SET LOCAL enable_indexscan = off")
        connection.execute("SET LOCAL enable_bitmapscan = off")
        connection.execute("SET LOCAL enable_seqscan = on")
    else:
        # This deliberately makes PostgreSQL use the HNSW index for a fair demo
        # even though the learning corpus is too small for the planner to prefer it.
        connection.execute("SET LOCAL enable_seqscan = off")
        connection.execute("SET LOCAL enable_indexscan = on")
        connection.execute("SET LOCAL enable_bitmapscan = off")


def explain(connection, mode: str, query_embedding, limit: int) -> list[str]:
    with connection.transaction():
        set_search_mode(connection, mode)
        rows = connection.execute(
            "EXPLAIN (COSTS OFF) " + SEARCH_SQL,
            (query_embedding, limit),
        ).fetchall()
        return [row[0] for row in rows]


def timed_search(connection, mode: str, query_embedding, limit: int) -> tuple[list[str], float]:
    with connection.transaction():
        set_search_mode(connection, mode)
        started = time.perf_counter()
        rows = connection.execute(SEARCH_SQL, (query_embedding, limit)).fetchall()
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [row[0] for row in rows], elapsed_ms


def evaluate(limit: int, iterations: int) -> None:
    cases = load_cases()
    query_vectors = embed_texts([case["query"] for case in cases])

    with connect() as connection:
        register_vector(connection)
        index_name = connection.execute(
            "SELECT to_regclass('documents_embedding_hnsw_cosine_idx')"
        ).fetchone()[0]
        if index_name is None:
            raise RuntimeError("HNSW index not found; run vector-index-create first.")
        # register_vector may query PostgreSQL's type catalog; end that setup
        # transaction so each SET LOCAL is scoped to its individual search.
        connection.commit()

        for case_number, (case, query_vector) in enumerate(zip(cases, query_vectors, strict=True)):
            exact_keys, exact_ms = timed_search(connection, "exact", query_vector, limit)
            approximate_keys, hnsw_ms = timed_search(connection, "hnsw", query_vector, limit)

            exact_set = set(exact_keys)
            ann_recall = len(exact_set.intersection(approximate_keys)) / max(len(exact_set), 1)
            expected = set(case["relevant_keys"])
            relevant_recall = len(expected.intersection(approximate_keys)) / max(len(expected), 1)

            exact_times = [exact_ms]
            hnsw_times = [hnsw_ms]
            for _ in range(iterations - 1):
                _, elapsed = timed_search(connection, "exact", query_vector, limit)
                exact_times.append(elapsed)
                _, elapsed = timed_search(connection, "hnsw", query_vector, limit)
                hnsw_times.append(elapsed)

            print(f"\nQuery: {case['query']}")
            print(f"  Exact top {limit}: {', '.join(exact_keys) or '(no rows)'}")
            print(f"  HNSW top {limit}:  {', '.join(approximate_keys) or '(no rows)'}")
            print(f"  ANN recall@{limit} vs exact: {ann_recall:.1%}")
            print(f"  Labeled relevant@{limit}:  {relevant_recall:.1%}")
            print(f"  Mean DB query time ({iterations} runs): exact {statistics.mean(exact_times):.3f} ms, HNSW {statistics.mean(hnsw_times):.3f} ms")

            if case_number == 0:
                for mode, label in (("exact", "Exact plan"), ("hnsw", "HNSW plan")):
                    plan = explain(connection, mode, query_vector, limit)
                    if mode == "hnsw" and not any("documents_embedding_hnsw_cosine_idx" in line for line in plan):
                        raise RuntimeError("PostgreSQL did not use the HNSW index for the forced HNSW plan.")
                    print(f"  {label}:")
                    for plan_line in plan:
                        print(f"    {plan_line}")

    print("\nThe corpus is small, so timings are illustrative only. HNSW benefits and recall tradeoffs need a much larger representative dataset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact vector search with HNSW on labeled queries.")
    parser.add_argument("--limit", type=int, default=5, help="Results per query (default: 5)")
    parser.add_argument("--iterations", type=int, default=10, help="Timed runs per method (default: 10)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    evaluate(args.limit, args.iterations)


if __name__ == "__main__":
    main()
