"""Batched Gemini embedding client (chat models: ``llm/chat.py``).

Every Gemini object is built from explicitly injected credentials. Nothing in
this module reads application settings or creates clients at import time.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.shared.logging import safe_log


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIMENSION = 3072
EMBED_BATCH_SIZE = 64
MAX_EMBED_RETRIES = 3
RETRY_BASE_WAIT = 1.5
EMBED_PARALLEL_WORKERS = 3  # parallel batch embedding workers


def _require_text(value: str, name: str) -> str:
    text = (value or "").strip()

    if not text:
        raise ValueError(f"{name} must not be empty")

    return text


class GeminiEmbedder:
    """Batched, retrying Gemini embeddings for documents and queries."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        key = _require_text(api_key, "api_key")
        model_name = _require_text(model, "model")

        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension

        self._doc_model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=key,
        )
        self._query_model = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type="RETRIEVAL_QUERY",
            google_api_key=key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in original order."""
        return self._embed_batched(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query."""
        vectors = self._embed_batched([text], is_query=True)
        return vectors[0] if vectors else []

    # ── Internals ───────────────────────────────────────────────────

    def _embed_single_batch(
        self,
        batch: list[str],
        is_query: bool,
    ) -> list[list[float]]:
        """Embed one batch with exponential-backoff retry."""
        model = self._query_model if is_query else self._doc_model

        for attempt in range(1, MAX_EMBED_RETRIES + 1):
            try:
                if is_query and len(batch) == 1:
                    return [model.embed_query(batch[0])]
                return model.embed_documents(batch)
            except Exception:
                if attempt == MAX_EMBED_RETRIES:
                    safe_log(logger, logging.ERROR, "llm.embedding_failed")
                    raise
                wait = min(RETRY_BASE_WAIT * (2 ** (attempt - 1)), 10)
                safe_log(logger, logging.WARNING, "llm.embedding_retry")
                time.sleep(wait)
        return []  # unreachable — satisfies type checker

    def _embed_batched(
        self,
        texts: list[str],
        *,
        is_query: bool,
    ) -> list[list[float]]:
        """
        Embed texts in batches.

        • Single batch  → inline call (no thread-pool overhead).
        • Multiple batches → parallel via ThreadPoolExecutor
          (capped at EMBED_PARALLEL_WORKERS to respect API rate limits).
        """
        if not texts:
            return []

        batches = [
            texts[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]

        if len(batches) == 1:
            return self._embed_single_batch(batches[0], is_query)

        results: list[list[list[float]] | None] = [None] * len(batches)

        with ThreadPoolExecutor(
            max_workers=min(EMBED_PARALLEL_WORKERS, len(batches))
        ) as ex:
            future_to_idx = {
                ex.submit(self._embed_single_batch, batch, is_query): idx
                for idx, batch in enumerate(batches)
            }

            for fut in as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()  # propagates exceptions

        out: list[list[float]] = []
        for result in results:
            if result:
                out.extend(result)
        return out


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "GeminiEmbedder",
]
