# The published configurations, all in one place.

from typing import Literal, get_args

DEFAULT_CHECKPOINT = "checkpoints/sardi-dream-7b"

DEFAULT_TAU_C = 0.9 # tau_c, the commit threshold.

SaveHistory = Literal["none", "queries", "full"]
SAVE_HISTORY_CHOICES: tuple = get_args(SaveHistory)

DATASETS = {
    "2wiki": {
        "label": "2WikiMultiHopQA",
        "data": "data/2wikimultihopqa/test.parquet",
        "index": "data/2wikimultihopqa/corpus/index_chunked",
    },
    "hotpotqa": {
        "label": "HotpotQA",
        "data": "data/hotpotqa/test.parquet",
        "index": "data/hotpotqa/corpus/index_chunked",
    },
    "cofca": {
        "label": "CofCA",
        "data": "data/cofca/test.parquet",
        "index": "data/cofca/corpus/index_chunked",
    },
    "musique": {
        "label": "MuSiQue",
        "data": "data/musique/test.parquet",
        "index": "data/musique/corpus/index_chunked",
    },
    "synthworlds": {
        "label": "SynthWorlds-SM",
        "data": "data/synthworlds/sm/test.parquet",
        "index": "data/synthworlds/sm/corpus/index_chunked",
    },
}

# The two DLM methods in Table 1.
METHODS = {
    # SARDI: query = question + proxy response, refreshed at every denoising step.
    "sardi": {"rag_query_type": "question_reasoning", "retrieval_steps": "0,s"},
    # DLM w/ ret@static: query is the question and never changes.
    "ret_static": {"rag_query_type": "question", "retrieval_steps": "0,s"},
}

# Passed to evaluate.py for every published run.
PAPER_CONFIG = {
    "model_type": "dream7b",
    "alg": "confidence_threshold",
    "steps": 100,
    "max_new_tokens": 100,
    "temperature": 0.0,
    "alg_temp": 0.0,
    "retrieve_top_k": 7,                    # K
    "retriever_type": "bm25s",
    "rag_query_confidence_threshold": 0.0,  # tau_q
    "batch_size": 1,
}

# Every setting that has a published value, and the value(s) it took.
# Anything outside these produces a run that cannot be compared to the paper.
PAPER_SETTINGS = {
    "model_type": ("dream7b",),
    "alg": ("confidence_threshold",),
    "alg_temp": (0.0,),
    "steps": (100,),
    "max_new_tokens": (100,),
    "temperature": (0.0,),
    "batch_size": (1,),
    "threshold": (0.9, 0.95),
    "retriever_type": ("bm25s",),
    "retrieve_top_k": (7,),
    "rag_query_confidence_threshold": (0.0,),
    "rag_query_type": ("question_reasoning", "question"),
    "retrieval_steps": ("0,s",),
    "rag_deduplicate_query": (True,),
}


def find_off_paper_settings(**observed) -> list:
    """Return [(name, got, published), ...] for settings that differ from the paper.
    """
    diffs = []
    for name, allowed in PAPER_SETTINGS.items():
        if name not in observed:
            continue
        got = observed[name]
        # `is` comparison for bools so True does not match a published 1.
        if any(got is a or got == a for a in allowed):
            continue
        diffs.append((name, got, allowed[0] if len(allowed) == 1 else allowed))
    return diffs


def format_off_paper_warning(diffs: list) -> str:
    """Render find_off_paper_settings output as a block suitable for stderr."""
    if not diffs:
        return ""
    width = max(len(name) for name, _, _ in diffs)
    lines = [
        f"WARNING: {len(diffs)} setting(s) differ from the published configuration:"
    ]
    for name, got, paper in diffs:
        lines.append(f"    {name:<{width}}  {got!r}   (paper: {paper!r})")
    lines.append("  This run will not reproduce Table 1. See the README section 'Flags'.")
    return "\n".join(lines)
