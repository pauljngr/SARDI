#!/usr/bin/env python3
"""Answer your own questions with SARDI.

    python example.py

To evaluate on a full dataset, use evaluate.py.
"""
import os
import sys

from sardi.inference import inference, load_model
from sardi.rag.retriever import SparseBM25SRetriever

# Any bm25s index directory. Point this at your own corpus to run SARDI over your own documents.
INDEX = "data/2wikimultihopqa/corpus/index_chunked"

QUESTIONS = [
    "Which city is the capital of the country where the composer of The Magic Flute was born?",
    "Who died later, Pieter Corneliszoon Hooft or Rudi Carrell?",
]


def main() -> int:
    if not os.path.isdir(INDEX):
        print(
            f"No index at {INDEX}. See the README section 'Assets' for the\n"
            "download commands, or set INDEX to a bm25s index directory of your own.",
            file=sys.stderr,
        )
        return 1

    model, tokenizer = load_model(compile_model=False) # skip compilation for faster loading
    retriever = SparseBM25SRetriever(
        corpus_path=os.path.join(INDEX, "corpus.jsonl"), index_path=INDEX
    )

    for question in QUESTIONS:
        print(f"\nQuestion: {question}")
        print(f"Answer: {inference(model, tokenizer, question, retriever)}")

    # `raw=True` returns the reasoning trace alongside the answer.
    print("\n--- full generation for the first question ---")
    print(inference(model, tokenizer, QUESTIONS[0], retriever, raw=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
