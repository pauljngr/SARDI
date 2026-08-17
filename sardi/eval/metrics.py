# Answer extraction and Exact Match / F1 scoring.

import re
import string
from typing import Dict


def extract_final_answer(generation: str) -> str:
    """Pull the predicted answer out of a generation.

    Tries, in order: a `\\boxed{...}`, the last "answer is ..." phrasing, then the
    text after the last `###` marker. Returns "" if nothing matches.
    """
    generation = re.sub(r"<\|[^|]+\|>", "", generation).strip()

    match = re.search(r"\\boxed\{(.+?)\}", generation)
    if match:
        return match.group(1).strip()

    matches = list(
        re.finditer(r"(?:[Tt]he\s+)?(?:[Ff]inal\s+)?[Aa]nswer\s+is[:\s]+(.+)", generation)
    )
    if matches:
        ans = matches[-1].group(1).strip().split("\n")[0].strip()
        return re.sub(r"\.\s*$", "", ans)

    if "###" in generation:
        ans = generation.split("###")[-1].strip().split("\n")[0].strip()
        return re.sub(r"\.\s*$", "", ans)

    return ""


def normalize_answer(answer: str) -> str:
    """HotpotQA-style normalization: lowercase, drop articles and punctuation."""
    s = answer.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> bool:
    """Whether prediction and gold agree after normalization.

    An empty prediction `""` never matches.
    """
    normalized = normalize_answer(prediction)
    return bool(normalized) and normalized == normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (HotpotQA-style)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in set(gold_tokens))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def print_accuracy_report(metrics: Dict, show_errors: bool = True, n_examples: int = 5):
    """Print EM/F1 plus a few example generations."""
    print("\n" + "=" * 80)
    print("ACCURACY REPORT")
    print("=" * 80)
    print(
        f"EM: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})  "
        f"F1: {metrics.get('f1', 0):.2%}"
    )
    print("=" * 80)

    def show(title, subset):
        print(f"\nShowing up to {n_examples} {title} predictions:")
        print("-" * 80)
        for i, result in enumerate(subset[:n_examples]):
            print(f"\nExample {i + 1}:")
            print(f"Question: {result['question']}")
            print(f"\nFull Generation:\n{result['full_generation'][:500]}")
            print(f"Predicted Answer: {result['predicted_answer']}")
            print(f"Gold Answer: {result['gold_answer']}")
            print("-" * 80)

    results = metrics.get("results", [])
    if show_errors:
        show("incorrect", [r for r in results if not r["correct"]])
    show("correct", [r for r in results if r["correct"]])
