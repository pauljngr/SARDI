#!/usr/bin/env python3
"""Score SARDI on the benchmark splits.

Name a dataset and you get the published configuration:

    python evaluate.py --dataset 2wiki --threshold 0.9

Point it at your own parquet split with --data_path to change anything else;
every generation and retrieval setting is then yours to set, and the run
announces which of them differ from the published values.

To answer individual questions rather than score a split, use sardi.inference —
see example.py.
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime

# NCCL settings for multi-GPU eval on some node types.
# if __name__ == "__main__" and "--distributed" in sys.argv:
#     os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
#     os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from sardi.config import (  # noqa: E402
    DATASETS,
    DEFAULT_CHECKPOINT,
    DEFAULT_TAU_C,
    METHODS,
    SAVE_HISTORY_CHOICES,
    find_off_paper_settings,
    format_off_paper_warning,
)
from sardi.eval.metrics import print_accuracy_report  # noqa: E402
from sardi.eval.runner import compute_accuracy  # noqa: E402
from sardi.inference import load_model, parse_retrieval_steps  # noqa: E402
from sardi.rag.retriever import SparseBM25SRetriever  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a diffusion model on multi-hop QA")

    # Table 1 cells. --dataset fills in the split, corpus, index and query type;
    # --data_path is the escape hatch for anything else.
    p.add_argument("--dataset", default=None, choices=sorted(DATASETS),
                   help="A benchmark dataset. Uses the default paths and, with "
                        "--method, the paper's retrieval configuration.")
    p.add_argument("--method", default="sardi", choices=sorted(METHODS),
                   help="'sardi', or 'ret_static' for the DLM w/ ret@static baseline. "
                        "Only meaningful with --dataset.")

    # Model
    p.add_argument("--model_type", default="dream7b", choices=["dream7b"])
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="SARDI checkpoint directory or Hugging Face repo id")

    # Generation
    p.add_argument("--steps", type=int, default=100, help="Max denoising steps")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=DEFAULT_TAU_C,
                   help="tau_c, the commit threshold")
    p.add_argument("--alg", default="confidence_threshold",
                   choices=["confidence_threshold", "maskgit_plus", "topk_margin", "entropy", "origin"])
    p.add_argument("--alg_temp", type=float, default=0.0)

    # Data / output
    p.add_argument("--data_path", default=None,
                   help="Path to an eval parquet split. Required unless --dataset is given.")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--output_name", default=None)
    p.add_argument("--no_report", action="store_true")
    p.add_argument("--nocompile", action="store_true", help="Disable torch.compile")
    p.add_argument("--save_history", default="queries", choices=SAVE_HISTORY_CHOICES,
                   help="How much retrieval history to write out. 'full' includes "
                        "passage text and produces very large result files (~300 MB/run).")

    # Retrieval
    p.add_argument("--corpus_path", default=None, help="Chunked corpus JSONL")
    p.add_argument("--index_path", default=None, help="BM25 index directory (built if absent)")
    p.add_argument("--retrieval_steps", default="0,s",
                   help="Steps to refresh retrieval: range '0,s' (every step, the "
                        "paper default), range '0,10', or list '0,10,20'")
    p.add_argument("--rag_query_type", default="question_reasoning",
                   choices=["question_reasoning", "question", "reasoning"],
                   help="'question_reasoning' = SARDI; 'question' = ret@static")
    p.add_argument("--rag_query_confidence_threshold", type=float, default=0.0,
                   help="tau_q. 0.0 exposes every masked position to the retriever "
                        "(paper default); 1.0 uses committed tokens only.")
    p.add_argument("--retrieve_top_k", type=int, default=7, help="K passages per retrieval")
    p.add_argument("--rag_deduplicate_query",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Drop repeated words from the query (useful at beginning of generation).")
    p.add_argument("--deterministic_passage_order",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Order passages by BM25 rank, so runs are reproducible. "
                        "--no-deterministic_passage_order restores the Python set "
                        "order the published numbers were produced with; measured "
                        "over full 2Wiki and HotpotQA the two agree within 0.3 EM.")

    # Dense retriever (paper Table 4 ablation; outside this release — see
    # FlashRAGRetriever in sardi/rag/retriever.py)
    p.add_argument("--retriever_type", default="bm25s", choices=["bm25s", "dense"])
    p.add_argument("--dense_model_path", default="intfloat/e5-base-v2")
    p.add_argument("--dense_index_path", default=None)
    p.add_argument("--dense_pooling_method", default="mean", choices=["mean", "cls", "pooler"])
    p.add_argument("--dense_retrieval_method", default="e5")
    p.add_argument("--dense_instruction", default="query: ")
    p.add_argument("--dense_max_length", type=int, default=512)
    p.add_argument("--dense_faiss_gpu", action="store_true")

    p.add_argument("--distributed", action="store_true", help="Multi-GPU eval via torchrun")
    return p


# Flags that --dataset fills in. Passing one alongside --dataset is a conflict.
_DATASET_OWNED = (
    "data_path", "corpus_path", "index_path", "rag_query_type", "retrieval_steps",
)


def _passed(argv, name: str) -> bool:
    return any(a == f"--{name}" or a.startswith(f"--{name}=") for a in argv)


def apply_dataset(args, argv) -> None:
    """Resolve --dataset into the published paths and retrieval configuration.

    Mutates `args` in place. Without --dataset nothing happens, beyond insisting
    that --data_path was given instead.
    """
    if args.dataset is None:
        if not args.data_path:
            raise SystemExit(
                "Pass --dataset for a benchmark split (the published configuration), "
                "or --data_path for your own.\n"
                f"Datasets: {', '.join(sorted(DATASETS))}"
            )
        return

    clashes = [f"--{n}" for n in _DATASET_OWNED if _passed(argv, n)]
    if clashes:
        raise SystemExit(
            f"--dataset {args.dataset} already sets {', '.join(clashes)}.\n"
            "Drop the conflicting flag, or drop --dataset and pass --data_path yourself."
        )

    ds = DATASETS[args.dataset]
    method = METHODS[args.method]
    args.data_path = ds["data"]
    args.corpus_path = os.path.join(ds["index"], "corpus.jsonl")
    args.index_path = ds["index"]
    args.rag_query_type = method["rag_query_type"]
    args.retrieval_steps = method["retrieval_steps"]


def check_assets(args) -> None:
    """Fail before the model loads if the split, corpus or checkpoint is missing."""
    missing = [
        p for p in (args.data_path, args.corpus_path, args.checkpoint)
        if p and not os.path.exists(p) and os.sep in str(p)
    ]
    if missing:
        raise SystemExit(
            "Missing required assets:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nSee the README section 'Assets' for the download commands.\n"
        )
    if args.index_path and not os.path.exists(args.index_path):
        print(f"[Eval] No BM25 index at {args.index_path} — it will be built from "
              "the corpus on first use and cached there.")


def build_retriever_from_args(args, rank: int):
    if not args.corpus_path:
        raise ValueError("--corpus_path is required.")

    if args.retriever_type == "dense":
        from sardi.rag.retriever import FlashRAGRetriever

        if rank == 0:
            print(f"[Eval] Initializing dense retriever ({args.dense_retrieval_method})")
        return FlashRAGRetriever(
            corpus_path=args.corpus_path,
            index_path=args.dense_index_path or args.index_path,
            retrieval_method=args.dense_retrieval_method,
            model_path=args.dense_model_path,
            pooling_method=args.dense_pooling_method,
            instruction=args.dense_instruction,
            max_length=args.dense_max_length,
            faiss_gpu=args.dense_faiss_gpu,
        )

    if not args.index_path:
        raise ValueError("--index_path is required for the BM25 retriever.")
    if rank == 0:
        print(f"[Eval] Initializing BM25S retriever from index: {args.index_path}")
    return SparseBM25SRetriever(corpus_path=args.corpus_path, index_path=args.index_path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_dataset(args, sys.argv[1:] if argv is None else argv)
    check_assets(args)

    is_distributed = args.distributed
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank, world_size, device = 0, 1, args.device

    if rank == 0:
        # Say loudly when this run cannot be compared to Table 1.
        diffs = find_off_paper_settings(**vars(args))
        if diffs:
            print(format_off_paper_warning(diffs) + "\n", file=sys.stderr)

        print(f"[Eval] device={device} model_type={args.model_type}")
        if is_distributed:
            print(f"[Eval] Distributed evaluation across {world_size} GPUs")

    model, tokenizer = load_model(
        checkpoint=args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
        compile_model=not args.nocompile,
        verbose=(rank == 0),
    )

    retriever = build_retriever_from_args(args, rank)
    if retriever is not None and rank == 0:
        print("[Eval] RAG enabled")

    if is_distributed:
        dist.barrier()

    retrieval_steps = parse_retrieval_steps(args.retrieval_steps, args.steps)
    if rank == 0 and retrieval_steps is not None:
        summary = (
            str(retrieval_steps) if len(retrieval_steps) <= 10
            else f"{retrieval_steps[:5]}...{retrieval_steps[-5:]}"
        )
        print(f"[Eval] Retrieval steps: {summary}")

    metrics = compute_accuracy(
        model=model,
        tokenizer=tokenizer,
        data_paths=[args.data_path],
        retriever=retriever,
        retrieval_steps=retrieval_steps,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        temperature=args.temperature,
        threshold=args.threshold,
        alg=args.alg,
        alg_temp=args.alg_temp,
        device=device,
        use_chat_template=not args.no_chat_template,
        rag_query_type=args.rag_query_type,
        rag_query_confidence_threshold=args.rag_query_confidence_threshold,
        retrieve_top_k=args.retrieve_top_k,
        rag_deduplicate_query=args.rag_deduplicate_query,
        deterministic_passage_order=args.deterministic_passage_order,
        save_history=args.save_history,
        distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        show_progress=(rank == 0),
        return_results=True,
    )

    if is_distributed:
        gathered = [None] * world_size
        dist.all_gather_object(
            gathered,
            {
                "correct": metrics["correct"],
                "total": metrics["total"],
                "results": metrics["results"],
            },
        )
        if rank == 0:
            total_correct = sum(g["correct"] for g in gathered)
            total_samples = sum(g["total"] for g in gathered)
            combined = [r for g in gathered for r in g["results"]]
            metrics = {
                "accuracy": total_correct / total_samples if total_samples else 0.0,
                "f1": (
                    sum(r["f1"] for r in combined) / len(combined) if combined else 0.0
                ),
                "correct": total_correct,
                "total": total_samples,
                "results": combined,
            }

    exit_metrics = None
    if rank == 0:
        if not args.no_report:
            print_accuracy_report(metrics, show_errors=True, n_examples=5)

        print(
            f"----------\n[Eval] EM: {metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['total']})"
        )

        retrieval_times = [
            r["total_retrieval_time"] for r in metrics["results"] if "total_retrieval_time" in r
        ]
        gen_times = [r["gen_time"] for r in metrics["results"] if "gen_time" in r]
        avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else None
        avg_gen = sum(gen_times) / len(gen_times) if gen_times else None
        if avg_retrieval is not None:
            print(f"[Eval] Avg retrieval time per question: {avg_retrieval:.4f}s")
        if avg_gen is not None:
            print(f"[Eval] Avg wall-clock per question: {avg_gen:.4f}s")

        metrics_payload = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "correct": metrics["correct"],
            "total": metrics["total"],
        }
        if avg_retrieval is not None:
            metrics_payload["avg_retrieval_time"] = avg_retrieval
        if avg_gen is not None:
            metrics_payload["avg_gen_time"] = avg_gen
        exit_metrics = metrics_payload

        if args.output_dir:
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
                base = (args.output_name or "eval_results").removesuffix(".json")
                stem = f"{base}-{ts}-{uuid.uuid4().hex[:8]}"

                config = vars(args).copy()

                # Small sidecar first, so the headline numbers survive even if the
                # big file is inconvenient to open.
                with open(os.path.join(args.output_dir, f"{stem}.metrics.json"), "w") as f:
                    json.dump({"config": config, "metrics": metrics_payload}, f, indent=2)

                out_path = os.path.join(args.output_dir, f"{stem}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"config": config, "metrics": metrics_payload, "results": metrics["results"]},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"[Eval] Saved results to: {out_path}")
            except OSError as e:
                print(f"[Eval] Failed to save results: {e}")

    if is_distributed:
        dist.destroy_process_group()

    return exit_metrics


if __name__ == "__main__":
    with torch.inference_mode():
        main()
