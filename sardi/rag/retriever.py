# Retrievers for SARDI.

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import bm25s
import filelock
import numpy as np
import pandas as pd

# Silence bm25s progress bar
_original_tokenize = bm25s.tokenize

def _patched_tokenize(*args, **kwargs):
    kwargs.setdefault("show_progress", False)
    return _original_tokenize(*args, **kwargs)

bm25s.tokenize = _patched_tokenize

# Column names accepted for passage text, in priority order.
_TEXT_COLUMNS = ("contents", "text")


@dataclass
class RetrievedDocument:
    doc_id: Any
    text: str
    score: float
    rank: int


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return the top_k passage texts for a query, most relevant first."""

    @abstractmethod
    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """Return the top_k passages wrapped with id, score and rank."""

    @abstractmethod
    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        """Return the top_k passage texts for each query."""


class SparseBM25SRetriever(Retriever):
    """BM25 retriever over a chunked JSONL corpus, backed by `bm25s`.

    This is the retriever used for every number in Table 1 (`--retriever_type bm25s`).

    The index is loaded with `load_corpus=False`, so passage text comes *solely*
    from `corpus_path` and lookup is positional: row i of the corpus must be the document the index
    knows as i.

    If `index_path` does not exist the index is built from the corpus and saved,
    guarded by a file lock so concurrent jobs cannot race.
    """

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        text_column: Optional[str] = None,
        id_column: str = "id",
        stopwords: str | List[str] = "en",
        stemmer: Any = None,
    ):
        print("[BM25S] Initializing retriever...")
        print(f"  Corpus: {corpus_path}")
        print(f"  Index: {index_path}")

        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus not found: {corpus_path}\n"
                "See the README section 'Assets' for the expected data layout."
            )

        starttime = time.time()

        corpus_start = time.time()
        corpus_df = pd.read_json(corpus_path, lines=True)
        print(f"[BM25S] Corpus loaded in {round(time.time() - corpus_start, 2)}s.")

        if text_column is None:
            for candidate in _TEXT_COLUMNS:
                if candidate in corpus_df.columns:
                    text_column = candidate
                    break
            else:
                raise ValueError(
                    f"None of {_TEXT_COLUMNS} found in corpus columns: "
                    f"{list(corpus_df.columns)}"
                )
            if text_column != _TEXT_COLUMNS[0]:
                print(f"[BM25S] Using '{text_column}' as the passage text column.")
        elif text_column not in corpus_df.columns:
            raise ValueError(
                f"`{text_column}` not found in corpus columns: {list(corpus_df.columns)}"
            )

        if id_column not in corpus_df.columns:
            corpus_df[id_column] = np.arange(len(corpus_df))

        self.text_column = text_column
        self.id_column = id_column
        self.stopwords = stopwords
        self.stemmer = stemmer

        self.doc_texts: List[str] = corpus_df[text_column].astype(str).tolist()
        self.doc_ids: List[Any] = corpus_df[id_column].tolist()
        # Release DataFrame to free memory
        del corpus_df

        lock_path = f"{index_path}.lock"
        lock = filelock.FileLock(lock_path, timeout=600)

        with lock:
            if os.path.exists(index_path):
                self.bm25 = bm25s.BM25.load(index_path, load_corpus=False)
                self.bm25.backend = "numba"
                print(f"[BM25S] Index loaded from: {index_path}")
                self._assert_index_matches_corpus(index_path, corpus_path)
            else:
                print("[BM25S] Building index from corpus (one-off, then cached)...")
                corpus_tokens = bm25s.tokenize(
                    self.doc_texts,
                    stopwords=self.stopwords,
                    stemmer=self.stemmer,
                )
                self.bm25 = bm25s.BM25(backend="numba")
                self.bm25.index(corpus_tokens)
                print(f"[BM25S] Index built with {len(self.doc_texts)} documents")
                self._save_index_unlocked(index_path)

        print("[BM25S] Compiling retriever...")
        # Warm-up retrieval to compile the numba kernels.
        self.retrieve("What is the capital of France?", top_k=1)

        print(f"[BM25S] Retriever initialized (took {round(time.time() - starttime, 2)}s).")

    def _assert_index_matches_corpus(self, index_path: str, corpus_path: str) -> None:
        """Fail loudly if the index was built over a different corpus.

        Retrieval returns positional indices into `self.doc_texts`, so a corpus
        of the wrong length silently returns the wrong passage for every query.
        """
        params_path = os.path.join(index_path, "params.index.json")
        if not os.path.exists(params_path):
            return
        try:
            with open(params_path) as f:
                num_docs = json.load(f).get("num_docs")
        except (OSError, json.JSONDecodeError):
            return
        if num_docs is not None and num_docs != len(self.doc_texts):
            raise ValueError(
                f"Corpus/index mismatch: index '{index_path}' was built over "
                f"{num_docs} documents but corpus '{corpus_path}' has "
                f"{len(self.doc_texts)}. Retrieval would return wrong passages. "
                "Delete the index directory to rebuild it."
            )

    def _save_index_unlocked(self, index_path: str):
        """Save the index without taking the lock (caller already holds it)."""
        temp_path = f"{index_path}.tmp"
        self.bm25.save(temp_path, corpus=self.doc_texts)

        if os.path.exists(index_path):
            import shutil

            shutil.rmtree(index_path, ignore_errors=True)
        os.rename(temp_path, index_path)
        print(f"[BM25S] Index saved to: {index_path}")

    def save_index(self, index_path: str):
        """Save the index to disk, with locking for concurrent jobs."""
        lock = filelock.FileLock(f"{index_path}.lock", timeout=600)
        with lock:
            self._save_index_unlocked(index_path)

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        query_tokens = bm25s.tokenize(
            query, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False
        )

        # A degenerate query (e.g. ".\n\n") tokenizes to [[]] and makes bm25s raise.
        # That happens naturally at early denoising steps, so treat it as "no evidence".
        try:
            doc_id_indices, _scores = self.bm25.retrieve(
                query_tokens, k=top_k, sorted=True, show_progress=False
            )
            doc_id_indices = doc_id_indices[0]
        except Exception as e:
            print(f"[BM25S] Retrieval failed for query {query[:80]!r}: {e}")
            return []

        return [self.doc_texts[int(idx)] for idx in doc_id_indices]

    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        query_tokens = bm25s.tokenize(
            query, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False
        )
        doc_id_indices, scores = self.bm25.retrieve(
            query_tokens, k=top_k, sorted=True, show_progress=False
        )
        doc_id_indices, scores = doc_id_indices[0], scores[0]

        return [
            RetrievedDocument(
                doc_id=self.doc_ids[int(idx)],
                text=self.doc_texts[int(idx)],
                score=float(score),
                rank=rank,
            )
            for rank, (idx, score) in enumerate(zip(doc_id_indices, scores), start=1)
        ]

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        if not queries:
            return []

        query_tokens = bm25s.tokenize(
            queries, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False
        )
        doc_id_indices, _scores = self.bm25.retrieve(query_tokens, k=top_k, show_progress=False)

        return [
            [self.doc_texts[int(i)] for i in doc_id_indices[q_idx]]
            for q_idx in range(doc_id_indices.shape[0])
        ]


class FlashRAGRetriever(Retriever):
    """Dense retriever (E5) via FlashRAG + FAISS.

    Used by the dense-retriever ablation (paper Table 4) (not part of this release).
    Dependencies can be installed via `pip install -e .`:

        pip install flashrag-dev==0.1.2 sentence-transformers==5.3.0 \
                    faiss-gpu-cu12==1.14.1.post1   # or faiss-cpu

    It also needs a FAISS index, which is not part of the released data; build
    one with FlashRAG's index_builder over the chunked corpus.
    """

    def __init__(
        self,
        corpus_path: str,
        index_path: str,
        retrieval_method: str = "e5",
        model_path: Optional[str] = None,
        pooling_method: Optional[str] = None,
        instruction: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 256,
        use_sentence_transformer: bool = False,
        faiss_gpu: bool = False,
    ):
        try:
            from flashrag.config import Config
            from flashrag.utils import get_retriever
        except ImportError as e:
            raise ImportError(
                "The dense retriever needs FlashRAG and faiss, which this release "
                "does not install:\n"
                "  pip install flashrag-dev==0.1.2 sentence-transformers==5.3.0 "
                "faiss-gpu-cu12==1.14.1.post1\n"
                "Table 1 uses BM25 only (--retriever_type bm25s), which needs neither."
            ) from e

        config_dict: Dict[str, Any] = {
            "retrieval_method": retrieval_method,
            "corpus_path": corpus_path,
            "index_path": index_path,
            "retrieval_use_fp16": use_fp16,
            "retrieval_query_max_length": max_length,
            "retrieval_batch_size": batch_size,
            "use_sentence_transformer": use_sentence_transformer,
            "faiss_gpu": faiss_gpu,
            "gpu_id": None,  # don't let FlashRAG override CUDA_VISIBLE_DEVICES
        }
        if model_path is not None:
            config_dict["retrieval_model_path"] = model_path
        if pooling_method is not None:
            config_dict["retrieval_pooling_method"] = pooling_method
        if instruction is not None:
            config_dict["instruction"] = instruction

        print(f"[FlashRAG] Initializing {retrieval_method} retriever...")
        print(f"  Corpus: {corpus_path}")
        print(f"  Index: {index_path}")

        starttime = time.time()
        self.config = Config(config_dict=config_dict)
        self.flashrag_retriever = get_retriever(self.config)
        print("[FlashRAG] Warming up retriever...")
        self.flashrag_retriever.search("What is the capital of France?", num=1)
        self.flashrag_retriever.search(
            "Who directed the film Inception and what year was it released?", num=3
        )
        print(f"[FlashRAG] Retriever initialized (took {round(time.time() - starttime, 2)}s).")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        return [doc["contents"] for doc in self.flashrag_retriever.search(query, num=top_k)]

    def retrieve_with_metadata(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        docs, scores = self.flashrag_retriever.search(query, num=top_k, return_score=True)
        return [
            RetrievedDocument(
                doc_id=doc["id"], text=doc["contents"], score=float(score), rank=rank
            )
            for rank, (doc, score) in enumerate(zip(docs, scores), start=1)
        ]

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        return [self.retrieve(q, top_k=top_k) for q in queries]
