# Prompt templates for SARDI.
#
# The prompt is filled in two stages, because SARDI rewrites the evidence while
# the model is generating: the question goes in up-front, and `{facts}` is
# refilled at every retrieval step with a new passage set.

from typing import List

# The prompt used for every number in Table 1. Don't change it!
# It has been used during SFT, so model might be sensitive to its exact wording.
RAG_PROMPT_TEMPLATE = """Use ONLY the provided facts to answer the question.
Think step-by-step, then provide the final answer after the "###" marker.

Question:
{question}

Facts:
{facts}"""


def fill_rag_prompt_question(question: str) -> str:
    """Stage one: substitute the question, leaving `{facts}` for later.
    """
    return RAG_PROMPT_TEMPLATE.replace("{question}", question)


def fill_rag_prompt_passages(prompt_template: str, passages: List[str]) -> str:
    """Stage two: fill `{facts}` with numbered passages.

    Raises:
        ValueError: if the template has no `{facts}` placeholder.
    """
    if "{facts}" not in prompt_template:
        raise ValueError(
            "prompt_template has no '{facts}' placeholder."
        )
    passages_str = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return prompt_template.replace("{facts}", passages_str)


def build_prompt(question: str, tokenizer, use_chat_template: bool = True) -> str:
    """Fill in the question, then wrap the result in the model's chat template.
    """
    prompt = fill_rag_prompt_question(question)
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
