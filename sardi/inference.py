# Run SARDI on a question.
#
# Load a model, build a retriever, ask a question. Every generation setting is fixed to its published value.
# See example.py for the intended usage.

import time
from typing import List, Optional, Tuple

import torch
from torch.amp.autocast_mode import autocast

from sardi.config import DEFAULT_CHECKPOINT, DEFAULT_TAU_C, METHODS, PAPER_CONFIG
from sardi.eval.metrics import extract_final_answer
from sardi.model.adapter import Dream7BAdapter
from sardi.rag.generate import generate_response_rag
from sardi.rag.prompts import build_prompt
from sardi.rag.retriever import Retriever


def load_model(
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    dtype=torch.bfloat16,
    compile_model: bool = True,
    verbose: bool = True,
) -> Tuple:
    """Load the SARDI checkpoint and its tokenizer.

    Returns:
        (model, tokenizer). The model is a BaseModelAdapter, which is what
        `inference` and evaluate.py take.
    """
    adapter, tokenizer = Dream7BAdapter.from_pretrained(
        checkpoint_path=checkpoint, device=device, dtype=dtype
    )
    adapter.eval()

    if compile_model:
        if verbose:
            print("[SARDI] Compiling with torch.compile()...")
        compile_start = time.time()
        adapter._model = torch.compile(adapter._model, dynamic=True)
        with torch.no_grad():
            warmup = tokenizer("warmup", return_tensors="pt").to(device)
            adapter._model(warmup.input_ids)
        if verbose:
            print(f"[SARDI] Compilation complete (took {time.time() - compile_start:.1f}s).")
    elif verbose:
        print("[SARDI] torch.compile disabled; EM may shift by a few tenths.")

    return adapter, tokenizer


def parse_retrieval_steps(spec: str, steps: int) -> Optional[List[int]]:
    """Parse a retrieval-step spec. 's' means "to the end", i.e. steps + 1.

    Accepts a range ("0,s", the paper setting, or "0,10") or an explicit list
    ("0,10,20").
    """
    if not spec:
        return None
    parts = spec.split(",")
    if len(parts) == 2 and parts[1] == "s":
        parts[1] = str(steps + 1)
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return list(range(int(parts[0]), int(parts[1])))
    return [int(x.strip()) for x in parts if x.strip().isdigit()]


def inference(
    model,
    tokenizer,
    question: str,
    retriever: Retriever,
    *,
    tau_c: float = DEFAULT_TAU_C,
    method: str = "sardi",
    raw: bool = False,
    verbose: bool = True,
) -> str:
    """Answer one question with SARDI, using the published configuration.

        >>> from sardi.inference import inference, load_model
        >>> from sardi.rag.retriever import SparseBM25SRetriever
        >>> model, tok = load_model()
        >>> index = "data/2wikimultihopqa/corpus/index_chunked"
        >>> ret = SparseBM25SRetriever(corpus_path=f"{index}/corpus.jsonl", index_path=index)
        >>> inference(model, tok, "Which city is the capital of the country "
        ...                       "where the composer of The Magic Flute was born?", ret)
        'Vienna'

    Args:
        model: From `load_model`.
        tokenizer: From `load_model`.
        question: The question to answer.
        retriever: A Retriever, e.g. SparseBM25SRetriever(corpus_path=...,
            index_path=...). Build it once and reuse it — loading an index
            is expensive.
        tau_c: Commit threshold. Paper uses 0.9 and 0.95.
        method: "sardi" (retrieval refreshed from the partial generation) or
            "ret_static" (the baseline: retrieve once from the question alone).
        raw: Return the full generation, including the reasoning trace, instead
            of just the extracted answer.
        verbose: Print the query and evidence set at each retrieval step.

    Returns:
        The extracted answer, or the full generation if `raw` is set.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {tuple(METHODS)}, got {method!r}")
    if retriever is None:
        raise ValueError(
            "inference() needs a retriever, e.g. SparseBM25SRetriever(corpus_path=..., index_path=...). "
        )

    steps = PAPER_CONFIG["steps"]
    prompt_str = build_prompt(question, tokenizer)

    use_amp = str(model.device).startswith("cuda") and torch.cuda.is_available()
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()

    with ctx:
        generation = generate_response_rag(
            model=model,
            prompt_template=prompt_str,
            question=question,
            retriever=retriever,
            tokenizer=tokenizer,
            retrieval_steps=parse_retrieval_steps(METHODS[method]["retrieval_steps"], steps),
            rag_query_type=METHODS[method]["rag_query_type"],
            threshold=tau_c,
            max_new_tokens=PAPER_CONFIG["max_new_tokens"],
            steps=steps,
            temperature=PAPER_CONFIG["temperature"],
            alg=PAPER_CONFIG["alg"],
            alg_temp=PAPER_CONFIG["alg_temp"],
            retrieve_top_k=PAPER_CONFIG["retrieve_top_k"],
            rag_query_confidence_threshold=PAPER_CONFIG["rag_query_confidence_threshold"],
            rag_deduplicate_query=True,
            deterministic_passage_order=True,
            verbose=verbose,
        )

    return generation if raw else extract_final_answer(generation)
