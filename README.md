<<<<<<< HEAD
# Learning Vector Embeddings with Python

A hands-on project for learning how text embeddings work and how to store and search them with Python, PostgreSQL, and [pgvector](https://github.com/pgvector/pgvector).

The project is intentionally isolated from the payment-platform database. Its Compose file publishes PostgreSQL on `localhost:5433` and uses a dedicated Docker volume.

## Start PostgreSQL with pgvector

From this directory:

```powershell
docker compose up -d
docker compose ps
docker compose exec postgres psql -U vectors -d vectors -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
docker compose exec postgres psql -U vectors -d vectors -c "SELECT '[1,2,3]'::vector;"
```

On first initialization, Compose mounts `sql/001-enable-vector.sql` into PostgreSQL's initialization directory. PostgreSQL runs it once when it creates a fresh data directory. If you already started this project before adding the initialization script, enable the extension in the existing database with:

```powershell
docker compose exec postgres psql -U vectors -d vectors -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

The first query should show one `vector` extension row; the second should print `[1,2,3]`. The vector cast confirms PostgreSQL can parse pgvector values. The initialization script does not rerun for an existing data volume, which is why the manual command is included.

Connect with:

```text
postgresql://vectors:vectors@localhost:5433/vectors
```

These credentials are for local learning only. Keep the database bound to localhost and do not reuse them for a public or production deployment.

## Learning design

The project uses a small text retrieval corpus so each step has an observable result. Given a question, Python turns it into an embedding; PostgreSQL ranks stored text by vector distance and returns the closest matches. The initial model will be `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional sentence embeddings. We will inspect the model choice and distance metric when implementing the embedding phase.

```text
Text corpus --> Python embedding model --> vectors table in PostgreSQL
                                                 ^
Question --> same embedding model --> nearest-neighbor query
                                                 |
                                         ranked text results
```

### Phases

1. **Database foundation** — Complete. Start the isolated PostgreSQL container, enable pgvector, and confirm the extension and vector type are available.
2. **Embedding fundamentals** — Implemented; run the steps below. Set up a Python environment, embed a handful of short texts, inspect dimensions and similarities, and learn why the same model must encode both stored text and queries.
3. **Store and retrieve** — Complete. Create a `documents` table with text, metadata, and a `vector(384)` column. Insert embeddings and write parameterized cosine-distance queries using the `<=>` operator.
4. **Build a small retrieval CLI** — Implemented; ingest UTF-8 text and Markdown files into overlapping chunks with stable path-based keys, then search them with a natural-language question.
5. **Index and evaluate** — Implemented; create the HNSW index and run the comparison below. Inspect exact and indexed query plans, then compare nearest-neighbor recall, labeled relevance, and query timings.

Each phase leaves the project runnable and includes a short explanation of the concepts and SQL/Python involved. The five planned learning phases are now implemented.

## Planned Python layout

The first implementation phase can establish this minimal structure without adding an API or orchestration framework:

```text
vector-embeddings/
  compose.yaml
  Dockerfile
  README.md
  pyproject.toml
  sql/
    001-enable-vector.sql
  src/vector_learning/
    db.py
    embed.py
    ingest.py
    ingest_files.py
    search.py
    index.py
    evaluate.py
    resources/
      002-create-documents.sql
      documents.jsonl
      evaluation.jsonl
  data/raw/
    pgvector-basics.md
```

The Python client will use `psycopg` with pgvector's Python adapter. Embedding and database access stay in separate modules so it is clear which work happens in the model and which happens in PostgreSQL.

## Phase 2: inspect text embeddings

### Option A: use a local virtual environment

From this directory, create and activate a project-local virtual environment, then install the package:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
learn-embeddings
```

The `-3.11` option selects Python 3.11. The `-m venv .venv` part creates an isolated Python environment in the `.venv` folder, so the project's packages do not affect other Python projects. You only need to create it once; activate it again in a new PowerShell session before running the demo.

The first run downloads the `all-MiniLM-L6-v2` model. The script embeds four example sentences, shows that each embedding has 384 values, prints a few values and each vector's length, then prints a cosine similarity matrix. The two cat/couch sentences should rank as more similar to each other than unrelated examples. Exact scores can vary with model versions.

To remove the virtual environment later, deactivate it if active and delete its folder:

```powershell
deactivate
Remove-Item -LiteralPath .venv -Recurse -Force
```

If `deactivate` is not recognized, the environment is not active; run only the `Remove-Item` command.

### Option B: use Docker

Docker does not require a local virtual environment. This is the recommended option on this Windows machine because the local PyTorch DLL was blocked by Windows Application Control (`WinError 4551`). Keep the policy enabled and run the demo in Docker's Linux container:

```powershell
docker compose run --build --rm embedding-demo
```

The first run builds the Python image with CPU-only PyTorch and downloads the model; the Compose volume `embedding-model-cache` keeps the model files for later runs. Docker Desktop must be running with its Linux engine. This path uses the same code and dependencies as the virtual-environment path.

The model converts text into numeric vectors; it does not store the original text or decide what is relevant for your application. `normalize_embeddings=True` makes every vector unit length. For these normalized vectors, the dot product is cosine similarity. pgvector's cosine distance operator (`<=>`) is `1 - cosine_similarity`, so a smaller distance means a closer match. The ingestion and search commands reuse `embed_texts` so the same model encodes both stored documents and queries.

## Phase 3: store and retrieve embeddings

The `documents` table stores the original text beside its embedding and JSON metadata. `document_key` is a stable unique key, so rerunning ingestion updates existing rows instead of creating duplicates. The `vector-db-init` command enables the extension, creates the table, and adds the key to an older table if needed without deleting rows; `vector-ingest` embeds the sample JSONL corpus and upserts it; `vector-search` embeds a query and asks PostgreSQL to rank all rows by exact cosine distance.

Run these commands from this directory using Docker (no local Python environment is needed):

```powershell
docker compose run --build --rm embedding-demo vector-db-init
docker compose run --rm embedding-demo vector-ingest
docker compose run --rm embedding-demo vector-search "How does cosine distance work?"
```

The first command builds the Python image with CPU-only PyTorch and starts PostgreSQL if needed; later commands reuse that image. The first embedding operation downloads the model into the persistent `embedding-model-cache` volume. Search uses a parameterized SQL query with `ORDER BY embedding <=> query_embedding LIMIT k`. The result prints both cosine distance (lower is closer) and cosine similarity (`1 - distance`). This is exact search and serves as the baseline for Phase 5.

For a locally installed package instead, start PostgreSQL with `docker compose up -d postgres`, activate the virtual environment, and run `vector-db-init`, `vector-ingest`, and `vector-search "How does cosine distance work?"`. The local default connection string is `postgresql://vectors:vectors@localhost:5433/vectors`.

## Phase 4: ingest and search raw text files

Put UTF-8 `.txt` or `.md` files under `data/raw/`. The file ingester walks that folder recursively, splits each file into 100-word chunks with 20 words of overlap, embeds each chunk, and stores it with its original path in metadata. Other formats such as PDF need a text-extraction step first.

Run:

```powershell
docker compose run --build --rm embedding-demo vector-ingest-files data/raw
docker compose run --rm embedding-demo vector-search "How does pgvector rank similar passages?"
```

The file's relative path and chunk number form a deterministic key, for example `raw/pgvector-basics.md#chunk-0001`. The `raw` namespace comes from the input folder name; use `--namespace handbook` to set a different stable namespace. Re-ingesting updates chunks with the same keys and removes obsolete chunks or deleted files only within that namespace. If there are no non-empty supported files, ingestion leaves the database unchanged. Use a distinct namespace for each separately managed corpus.

## Phase 5: index and evaluate retrieval

Create an HNSW index for cosine distance, then compare it with exact search on the hand-labeled queries in `evaluation.jsonl`:

```powershell
docker compose run --build --rm embedding-demo vector-index-create
docker compose run --rm embedding-demo vector-evaluate
```

The evaluator embeds each query once, then runs the SQL search repeatedly. It prints both query plans, **ANN recall@k** (overlap with exact nearest neighbors), **labeled relevant@k** (how many expected passages appeared), and mean database query time. For this small corpus, the evaluator deliberately disables sequential scans only for the HNSW comparison so PostgreSQL demonstrates the index plan; normal PostgreSQL planning generally prefers an exact scan on a tiny table. The timing and recall results are illustrative, not a production benchmark. Add a larger representative corpus before drawing performance conclusions.

## Initial schema direction

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          bigserial PRIMARY KEY,
    document_key text NOT NULL UNIQUE,
    content     text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding   vector(384) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
```

The first search will use exact cosine distance (`ORDER BY embedding <=> query_embedding LIMIT k`). We will add an HNSW index only after exact search is understood, then compare results and query plans. Model changes that alter embedding dimensions require a compatible column and re-embedding the stored corpus.
=======
# vector-embeddings
>>>>>>> fe14c0bad0629e316056e0db3e6f8651123d5254
