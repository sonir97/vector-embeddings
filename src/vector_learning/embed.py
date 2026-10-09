"""Create and inspect sentence embeddings with a small local example."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EXAMPLES = [
    "A cat is sleeping on the sofa.",
    "A kitten is resting on a couch.",
    "A spacecraft entered orbit around Mars.",
    "PostgreSQL stores structured data in tables.",
]


def embed_texts(texts: list[str], model: SentenceTransformer | None = None) -> np.ndarray:
    """Return one normalized 384-dimensional embedding for each input text."""
    encoder = model or SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def main() -> None:
    embeddings = embed_texts(EXAMPLES)

    print(f"Model: {MODEL_NAME}")
    print(f"Texts: {len(EXAMPLES)}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print("Each row is one text; each row has 384 learned numeric features.\n")

    for index, (text, vector) in enumerate(zip(EXAMPLES, embeddings, strict=True), start=1):
        print(f"{index}. {text}")
        print(f"   First 8 values: {np.array2string(vector[:8], precision=4)}")
        print(f"   L2 norm: {np.linalg.norm(vector):.4f}")

    # Normalized vectors have cosine similarity equal to their dot product.
    similarities = embeddings @ embeddings.T
    print("\nCosine similarity matrix (higher means more similar):")
    print("      " + " ".join(f"{i:>6}" for i in range(1, len(EXAMPLES) + 1)))
    for index, row in enumerate(similarities, start=1):
        print(f"{index:>3}   " + " ".join(f"{score:6.3f}" for score in row))

    print("\nText key:")
    for index, text in enumerate(EXAMPLES, start=1):
        print(f"  {index}: {text}")


if __name__ == "__main__":
    main()
