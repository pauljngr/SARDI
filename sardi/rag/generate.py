# SARDI: self-augmenting retrieval for diffusion language models.
#
# This file contains the main method.
#
# Written against Dream's diffusion_generate token_hook API:
# https://github.com/HKUNLP/Dream

import re
import time
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from sardi.config import DEFAULT_TAU_C
from sardi.rag.prompts import fill_rag_prompt_passages
from sardi.rag.retriever import Retriever

if TYPE_CHECKING:
    from sardi.model.adapter import BaseModelAdapter

# "question_reasoning" is SARDI; "question" is the ret@static baseline.
QUERY_TYPES = ("question_reasoning", "question", "reasoning")


@torch.no_grad()
def generate_response_rag(
    model: "BaseModelAdapter",
    prompt_template: str,
    question: str,
    retriever: Retriever,
    tokenizer: PreTrainedTokenizerBase,
    retrieval_steps: Optional[List[int]] = None,
    max_new_tokens: int = 100,
    steps: int = 100,
    temperature: float = 0.0,
    threshold: float = DEFAULT_TAU_C,
    rag_query_type: str = "question_reasoning",
    rag_query_confidence_threshold: float = 0.0,
    retrieve_top_k: int = 7,
    rag_deduplicate_query: bool = True,
    deterministic_passage_order: bool = True,
    generation_tokens_hook_func=None,
    generation_logits_hook_func=None,
    retrieval_callback=None,
    verbose: bool = False,
    **kwargs,
) -> str:
    """Generate an answer, refreshing retrieval during denoising.

    The method lives in the `token_hook` closure below, which Dream calls
    once at initialisation (step=None) and then after every denoising step. At
    each retrieval step the hook:

      1. builds a *proxy response* by filling masked positions with the model's
         current argmax guesses (gated by tau_q, `rag_query_confidence_threshold`),
      2. forms a query from the question plus that proxy,
      3. retrieves K passages, and
      4. rewrites the prompt in place with the new evidence set.

    Because the prompt is rebuilt every step, its length changes during
    generation; only the committed response tokens are carried across.

    Args:
        model: BaseModelAdapter wrapping the loaded model.
        prompt_template: Prompt with a `{facts}` placeholder, question already filled.
        question: The question, used to build every retrieval query.
        retriever: Retriever instance.
        tokenizer: Tokenizer for encode/decode.
        retrieval_steps: Steps at which to refresh retrieval. None (the default)
            refreshes at every step, which is what the paper uses.
        max_new_tokens: Response length in tokens.
        steps: Maximum number of denoising steps.
        temperature: Sampling temperature (0.0 = greedy, as in the paper).
        threshold: tau_c, the confidence needed to commit a token.
        rag_query_type: One of:
            - "question_reasoning": question + proxy response. This is SARDI.
            - "question": question only. This is the `ret@static` baseline —
              the query never changes, so the cache below makes retrieval happen
              exactly once and the same passages are re-injected each step.
            - "reasoning": proxy response only.
        rag_query_confidence_threshold: tau_q, the confidence a masked position
            needs before its guess is exposed to the retriever. 0.0 (the paper
            default) exposes every position; 1.0 disables the proxy entirely, so
            only committed tokens contribute to the query.
        retrieve_top_k: K, passages retrieved per step.
        rag_deduplicate_query: Drop repeated words from the query.
        deterministic_passage_order: Keep passages in BM25 rank order. On by
            default: it makes runs reproducible without pinning PYTHONHASHSEED.
            The published numbers used Python set order instead; measured over
            the full 2Wiki and HotpotQA splits the two agree to within 0.3 EM.
        generation_tokens_hook_func: Extra token hook, called after this one.
        generation_logits_hook_func: Logits hook.
        retrieval_callback: Called as (step, query, passages, response, elapsed).
        verbose: Print the query and evidence set at each retrieval step.
        **kwargs: Forwarded to `model.generate_response` (e.g. alg, alg_temp).

    Returns:
        The generated response text, truncated at EOS.
    """
    # Validate parameters
    if rag_query_type not in QUERY_TYPES:
        raise ValueError(f"rag_query_type must be one of {QUERY_TYPES}, got {rag_query_type!r}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold (tau_c) must be in (0, 1], got {threshold!r}")
    if not 0.0 <= rag_query_confidence_threshold <= 1.0:
        raise ValueError(
            f"rag_query_confidence_threshold (tau_q) must be in [0, 1], "
            f"got {rag_query_confidence_threshold!r}"
        )
    if retrieve_top_k < 1:
        raise ValueError(f"retrieve_top_k (K) must be >= 1, got {retrieve_top_k!r}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature!r}")

    dev = model.device
    mask_token_id = model.mask_token_id

    def deduplicate_passages(items) -> list | set:
        """Deduplicate retrieved passages, preserving BM25 rank by default.

        `set` order is what the published numbers used, but it depends on
        PYTHONHASHSEED and so varies run to run; `dict.fromkeys` does not.
        """
        if deterministic_passage_order:
            return list(dict.fromkeys(items))
        return set(items)

    prompt_encoding = tokenizer.encode(prompt_template, return_tensors="pt").to(dev)

    question_query = re.sub(r"\s+", " ", question)

    prev_query = None
    prev_passages: list | set = set()

    def token_hook(step, x, logits):
        nonlocal prompt_encoding, prev_query, prev_passages

        is_step_zero = step is None
        actual_step = (step + 1) if step is not None else 0

        if is_step_zero or retrieval_steps is None or (step + 1) in retrieval_steps:
            if verbose:
                print(f"\n--- Retrieval Step {actual_step} ---")

            response_encoding = x[0][prompt_encoding.shape[1]:]
            response_encoding_query = response_encoding.clone()

            # --- 1. Build the proxy response -----------------------------------
            # Fill masked positions with the model's argmax guess, keeping only
            # those at least tau_q confident. At tau_q = 0 every position is
            # exposed, which is what the paper uses and what performs best.
            if not is_step_zero and rag_query_confidence_threshold < 1.0:
                probs = torch.softmax(logits, dim=-1)
                confidence, x0 = probs.max(dim=-1)
                mask_index = x == mask_token_id
                confident = mask_index & (confidence >= rag_query_confidence_threshold)
                x_unmasked = x.clone()
                x_unmasked[confident] = x0[confident]
                response_encoding_query = x_unmasked[0][prompt_encoding.shape[1]:]

            def decode_and_normalize(encoding):
                decoded = tokenizer.decode(encoding, skip_special_tokens=False)
                for token in tokenizer.all_special_tokens:
                    decoded = decoded.replace(token, " ")
                decoded = re.sub(r"\s+", " ", decoded.strip())
                if rag_deduplicate_query:
                    # dict.fromkeys keeps first occurrence, drops the rest.
                    decoded = " ".join(dict.fromkeys(decoded.split(" ")))
                return decoded

            reasoning_query = decode_and_normalize(response_encoding_query.cpu())
            current_response = decode_and_normalize(response_encoding.cpu())

            def _retrieve(query, top_k=retrieve_top_k):
                """Retrieve, reusing the previous result if the query is unchanged.

                This is what keeps throughput high: as generation converges the
                query stops changing, and for `rag_query_type="question"` it
                never changes at all.
                """
                nonlocal prev_query, prev_passages
                if query == prev_query and prev_passages:
                    return prev_passages
                result = deduplicate_passages(retriever.retrieve(query, top_k=top_k))
                prev_query = query
                prev_passages = result
                return result

            retrieval_start_time = time.time()

            # --- 2. Form the query ---------------------------------------------
            # At step 0 there is no response yet, so every query type starts from
            # the question alone.
            if is_step_zero or rag_query_type == "question":
                query = question_query
            elif rag_query_type == "question_reasoning":
                query = question_query + " " + reasoning_query
            else:  # "reasoning"
                query = reasoning_query

            # --- 3. Retrieve ----------------------------------------------------
            passages = _retrieve(query)
            retrieval_elapsed_time = time.time() - retrieval_start_time

            if verbose:
                print(f"*** Current completion: \n{reasoning_query}")
                print(f"---> Query: {query}")
                print(f"---> Docs in context: {len(passages)}")

            # --- 4. Rewrite the prompt with the new evidence --------------------
            passages_list = list(passages)
            new_prompt = fill_rag_prompt_passages(prompt_template, passages_list)
            new_prompt_encoding = tokenizer.encode(new_prompt, return_tensors="pt").to(dev)

            x = torch.cat([new_prompt_encoding, response_encoding.unsqueeze(0)], dim=1)
            prompt_encoding = new_prompt_encoding

            if retrieval_callback is not None:
                retrieval_callback(
                    actual_step, query, passages_list, current_response, retrieval_elapsed_time
                )

        if generation_tokens_hook_func is not None:
            x = generation_tokens_hook_func(actual_step, x, logits)

        return x

    if generation_logits_hook_func is None:
        generation_logits_hook_func = lambda step, x, logits: logits  # noqa: E731

    output = model.generate_response(
        inputs=prompt_encoding,
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        threshold=threshold,
        generation_tokens_hook_func=token_hook,
        generation_logits_hook_func=generation_logits_hook_func,
        **kwargs,
    )

    # `output` is the full sequence [1, T_full]; slice off the (final) prompt.
    sequences = output if isinstance(output, torch.Tensor) else output.sequences
    generations = [
        tokenizer.decode(g[len(p):].tolist()) for p, g in zip(prompt_encoding, sequences)
    ]
    return generations[0].split(tokenizer.eos_token)[0]
