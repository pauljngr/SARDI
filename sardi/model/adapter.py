# Model adapter for Dream-7B.

import os
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


class BaseModelAdapter(ABC):
    """Wraps a language model behind a single generation API."""

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        """Token id used for masked (not-yet-generated) positions."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device the underlying model is on."""

    @abstractmethod
    def generate_response(
        self,
        inputs: torch.Tensor,
        max_new_tokens: int,
        steps: int,
        temperature: float,
        threshold: float,
        generation_tokens_hook_func=None,
        generation_logits_hook_func=None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate, calling `generation_tokens_hook_func(step, x, logits)` at
        every denoising step (step=None at initialisation). Returns the full
        sequence including the prompt."""

    def eval(self):
        if hasattr(self, "_model") and hasattr(self._model, "eval"):
            self._model.eval()
        return self


def _require_flash_attn() -> None:
    """Fail early and clearly if flash-attn is missing.

    All published numbers were produced with `attn_implementation="flash_attention_2"`.
    """
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "flash-attn is required to reproduce the paper's numbers, and is not "
            "installed (or failed to import).\n"
            "  pip install flash-attn --no-build-isolation\n"
            "See the README section 'Install'."
        ) from e


class Dream7BAdapter(BaseModelAdapter):
    """Adapter for Dream-7B and SARDI's fine-tuned checkpoint.

    The checkpoint is self-contained: it carries its own tokenizer, chat template
    and the patched `generation_utils.py` that implements `alg="confidence_threshold"`.
    `trust_remote_code=True` loads that code from the checkpoint directory,
    so the decoding algorithm ships with the weights rather than with this repo.
    See the README section 'What we changed in Dream's sampler'.
    """

    def __init__(self, model):
        self._model = model
        gen_config = model._prepare_generation_config(None)
        self._mask_token_id = gen_config.mask_token_id

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ) -> Tuple["Dream7BAdapter", AutoTokenizer]:
        """Load the SARDI checkpoint from a directory or a Hugging Face repo id.

        Stock Dream-7B is deliberately not accepted: its sampler has no
        `alg="confidence_threshold"` and no `threshold` argument, so it cannot
        run SARDI.
        """
        _require_flash_attn()

        model_path = checkpoint_path.rstrip(os.sep)
        print(f"[Model] Loading {model_path}")

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            raise ValueError(
                f"{model_path} looks like a LoRA adapter. This release ships the full-weight checkpoint only."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            dtype=dtype,
            device_map={"": device},
        )
        print("[Model] Loaded.")

        return cls(model), tokenizer

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def device(self) -> torch.device:
        dev = getattr(self._model, "device", None)
        if dev is not None:
            return dev
        return (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def generate_response(
        self,
        inputs,
        max_new_tokens,
        steps,
        temperature,
        threshold,
        generation_tokens_hook_func=None,
        generation_logits_hook_func=None,
        **kwargs,
    ):
        kwargs.pop("return_dict_in_generate", None)

        no_op_token = lambda s, x, l: x  # noqa: E731
        no_op_logits = lambda s, x, l: l  # noqa: E731

        output = self._model.diffusion_generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            threshold=threshold,
            generation_tokens_hook_func=generation_tokens_hook_func or no_op_token,
            generation_logits_hook_func=generation_logits_hook_func or no_op_logits,
            return_dict_in_generate=True,
            **kwargs,
        )
        return output.sequences if hasattr(output, "sequences") else output
