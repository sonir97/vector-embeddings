FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 \
    && python -m pip install \
        "numpy>=1.26,<3" \
        "pgvector>=0.3,<1" \
        "psycopg[binary]>=3.2,<4" \
        "sentence-transformers>=3.0,<6"

COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation .

CMD ["learn-embeddings"]
