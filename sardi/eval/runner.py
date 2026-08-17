# Evaluation loop: generate answer for all questions and score Exact Match.

import time
from typing import Dict, List, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sardi.config import DEFAULT_TAU_C, SAVE_HISTORY_CHOICES, SaveHistory
from sardi.eval.dataset import MultiHopQAEvalDataset, eval_collate
from sardi.eval.metrics import exact_match, extract_final_answer, token_f1
from sardi.model.adapter import BaseModelAdapter
from sardi.rag.prompts import build_prompt
from sardi.rag.generate import generate_response_rag
from sardi.rag.retriever import Retriever


def _trim_history(history: List[Dict], mode: SaveHistory) -> List[Dict]:
    """Decide how much of one question's retrieval log to keep in the results file.

    SARDI retrieves at every denoising step, so `history` holds ~100 records per
    question — each with the query it built, the partial response it built it
    from, the passages that came back, and how long the lookup took. `mode` is
    the `--save_history` setting:

      "none"     discard the log
      "queries"  keep each record but drop `passages`
      "full"     keep everything

    "queries" is the default. "full" writes roughly 300 MB for one 2Wiki run.
    """
    if mode == "none":
        return []
    if mode == "full":
        return history
    return [{k: v for k, v in record.items() if k != "passages"} for record in history]


def compute_accuracy(
    model: BaseModelAdapter,
    tokenizer,
    data_paths: List[str],
    retriever: Retriever,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    max_new_tokens: int = 100,
    steps: int = 100,
    temperature: float = 0.0,
    threshold: float = DEFAULT_TAU_C,
    alg: str = "confidence_threshold",
    alg_temp: float = 0.0,
    device: str = "cuda",
    use_chat_template: bool = True,
    retrieval_steps: Optional[List[int]] = None,
    rag_query_type: str = "question_reasoning",
    rag_query_confidence_threshold: float = 0.0,
    retrieve_top_k: int = 7,
    rag_deduplicate_query: bool = True,
    deterministic_passage_order: bool = True,
    save_history: SaveHistory = "queries",
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    show_progress: bool = True,
    return_results: bool = False,
    num_workers: int = 1,
) -> Dict:
    """Evaluate Exact Match over one or more parquet splits.

    Returns a dict with accuracy, f1, correct, total and (optionally) per-sample
    results. See sardi/config.py for the exact argument values behind Table 1.
    """
    if save_history not in SAVE_HISTORY_CHOICES:
        raise ValueError(
            f"save_history must be one of {SAVE_HISTORY_CHOICES}, got {save_history!r}"
        )

    dataset = MultiHopQAEvalDataset(data_paths=data_paths, max_samples=max_samples)

    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=eval_collate,
    )

    model.eval()
    correct = 0
    total = 0
    f1_scores: List[float] = []
    results: List[Dict] = []

    if show_progress:
        n_local = len(dataloader.dataset)
        if distributed and world_size > 1:
            print(f"Evaluating on {n_local} samples (rank {rank}/{world_size})...")
        else:
            print(f"Evaluating on {n_local} samples...")

    assert batch_size >= 1, "Batch size must be at least 1."

    if alg == "confidence_threshold" and retrieval_steps and retrieval_steps[0] != 0:
        print("Warning: with alg=confidence_threshold, retrieval should start at step 0.")

    effective_retrieval_steps = (
        retrieval_steps if retrieval_steps is not None else list(range(steps))
    )
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    with torch.no_grad():
        iterator = tqdm(
            dataloader, total=len(dataloader), disable=not show_progress, position=0, leave=True
        )

        for batch in iterator:
            questions: List[str] = batch["question"]
            gold_answers: List[str] = batch["answer"]
            sample_ids: List = batch["id"]

            gen_texts: List[str] = []
            gen_times: List[float] = []
            gen_histories: List[List[Dict]] = []

            # RAG generation runs one sample at a time: the prompt is rewritten
            # at every denoising step, so samples cannot share a batch.
            for question, gold_docs in zip(questions, batch["supporting_docs"]):
                gold_doc_contents = (
                    [doc["contents"] for doc in gold_docs] if gold_docs is not None else None
                )

                prompt_str = build_prompt(question, tokenizer, use_chat_template)

                retrieval_history: List[Dict] = []

                def make_retrieval_callback(gold_contents, ret_history):
                    """Build the per-step callback that logs each retrieval event into `ret_history`."""

                    def callback(step, query, passages, response, retrieval_time=0.0):
                        record = {
                            "step": step,
                            "query": query,
                            "response": response,
                            "passages": passages,
                            "retrieval_time": retrieval_time,
                        }
                        if gold_contents is not None:
                            record["num_gold_docs_included"] = sum(
                                1 for p in passages if p in gold_contents
                            )
                            record["total_num_gold_docs"] = len(gold_contents)
                        ret_history.append(record)

                    return callback

                ctx = (
                    autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_amp
                    else torch.no_grad()
                )
                gen_start_time = time.time()
                with ctx:
                    gen_text = generate_response_rag(
                        model=model,
                        prompt_template=prompt_str,
                        retriever=retriever,
                        tokenizer=tokenizer,
                        retrieval_steps=effective_retrieval_steps,
                        question=question,
                        rag_query_type=rag_query_type,
                        rag_query_confidence_threshold=rag_query_confidence_threshold,
                        retrieve_top_k=retrieve_top_k,
                        rag_deduplicate_query=rag_deduplicate_query,
                        deterministic_passage_order=deterministic_passage_order,
                        max_new_tokens=max_new_tokens,
                        steps=steps,
                        temperature=temperature,
                        threshold=threshold,
                        alg=alg,
                        alg_temp=alg_temp,
                        output_history=False,
                        verbose=False,
                        generation_logits_hook_func=None,
                        retrieval_callback=make_retrieval_callback(
                            gold_doc_contents, retrieval_history
                        ),
                    )
                gen_times.append(time.time() - gen_start_time)
                gen_texts.append(gen_text)
                gen_histories.append(retrieval_history)

            for sample_id, question, gold_answer, gen_text, gen_time, history in zip(
                sample_ids, questions, gold_answers, gen_texts, gen_times, gen_histories
            ):
                predicted_answer = extract_final_answer(gen_text)
                is_correct = exact_match(predicted_answer, gold_answer)
                f1 = token_f1(predicted_answer, gold_answer)
                if is_correct:
                    correct += 1
                f1_scores.append(f1)
                total += 1

                if return_results:
                    entry = {
                        "id": sample_id,
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": predicted_answer,
                        "full_generation": gen_text.split(tokenizer.eos_token)[0],
                        "correct": is_correct,
                        "f1": f1,
                        "gen_time": gen_time,
                    }
                    entry["retrieval_history"] = _trim_history(history, save_history)
                    entry["total_retrieval_time"] = sum(
                        e.get("retrieval_time", 0.0) for e in history
                    )
                    results.append(entry)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "correct": correct,
        "total": total,
        "results": results,
    }
