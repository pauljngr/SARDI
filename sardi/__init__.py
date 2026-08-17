"""SARDI: Self-Augmenting Retrieval for Diffusion Language Models.

Answer a question with the published configuration:

    from sardi.inference import inference, load_model
    from sardi.rag.retriever import SparseBM25SRetriever

    index = "data/2wikimultihopqa/corpus/index_chunked"
    model, tokenizer = load_model()
    retriever = SparseBM25SRetriever(corpus_path=f"{index}/corpus.jsonl", index_path=index)
    inference(model, tokenizer,
              "Which city is the capital of the country where the composer "
              "of The Magic Flute was born?",
              retriever)

See example.py for a runnable version, and evaluate.py to evaluate on a dataset.
"""

__version__ = "1.0.0"
