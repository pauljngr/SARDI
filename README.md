# Self-Augmenting Retrieval for Diffusion Language Models (ICML 2026)

Official code for [**Self-Augmenting Retrieval for Diffusion Language Models**](https://arxiv.org/abs/2606.06474),
by Paul Jünger, Justin Lovelace, Linxi Zhao, Dongyoung Go, Kilian Q. Weinberger.

![Retrieval interleaved with denoising: at each diffusion state, a query built
from the partially denoised sequence retrieves fresh context from the corpus,
which conditions the next denoising step.](docs/retrieve_during_denoising.png)

SARDI is the first framework to condition retrieval on intermediate diffusion
states. It interleaves retrieval with denoising: at each iteration it constructs
a query from the partially denoised sequence, retrieves fresh evidence, and
conditions the next step on the updated context. Central to SARDI is a
**separation between retrieval and generation confidence** unique to
non-autoregressive decoders — speculative future tokens can inform retrieval long
before they are stable enough to commit to the output. SARDI is plug-and-play
with any discrete diffusion language model that can produce reasoning traces.

## Install

First, install the python environment as follows (tested with Python 3.10).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install flash-attn==2.8.3 --no-build-isolation
```

For inference, you need a CUDA GPU with **~18 GB of VRAM**. To load HotpotQA's BM25 index, you need 32 GB of CPU RAM.

## Assets

As discussed in the paper, Dream-7B didn't reliably produce zero-shot reasoning traces out-of-the-box. We therefore provide a simple RAG-fine-tuned Dream-7B checkpoint, shipped with code for threshold-based decoding. 

```bash
hf download pauljngr/sardi-dream-7b --local-dir checkpoints/sardi-dream-7b
```

Additionally, we provide our dataset splits and exact retrieval corpora (~3 GB).
These are reformatted versions of public benchmarks (check HF dataset card for licenses):

```bash
hf download pauljngr/sardi-data --repo-type dataset --local-dir data
```

The files land in the layout `evaluate.py` expects:

```
checkpoints/sardi-dream-7b/             the fine-tuned model + threshold-based decoding code
data/<dataset>/test.parquet             the questions
data/<dataset>/corpus/index_chunked/    BM25 index, ships its own corpus.jsonl
```

for `<dataset>` in `2wikimultihopqa`, `hotpotqa`, `musique`, `cofca`, and
`synthworlds/sm`. 

## Usage

Smoke test on 20 questions:

```bash
python evaluate.py --dataset 2wiki --max_samples 20 --nocompile
```

Score a full benchmark split. `--dataset 2wiki` fills in the published configuration for `2wikimultihopqa`:

```bash
python evaluate.py --dataset 2wiki --method sardi --threshold 0.9
```

Answer your own questions, over any BM25 index:

```python
from sardi.inference import inference, load_model
from sardi.rag.retriever import SparseBM25SRetriever

index = "data/2wikimultihopqa/corpus/index_chunked"
model, tokenizer = load_model()
retriever = SparseBM25SRetriever(corpus_path=f"{index}/corpus.jsonl", index_path=index)

inference(model, tokenizer,
          "Which city is the capital of the country where the composer "
          "of The Magic Flute was born?",
          retriever)
```

Also, see `example.py` for an easy runnable version.

## Reproduce

All fifteen Table 1 cells from the paper, ~6–8 GPU-hours on one B200:

```bash
bash scripts/reproduce_table1.sh
```

Exact Match ×100. Expect ±1 EM variance between runs at identical
configuration.

| Dataset | n | SARDI τ_c=0.9 | SARDI τ_c=0.95 | DLM ret@static |
|---|---|---|---|---|
| 2WikiMultiHopQA | 6253 | 57.8 | 59.1 | 43.7 |
| HotpotQA | 3701 | 48.3 | 48.7 | 39.9 |
| CofCA | 900 | 45.3 | 44.9 | 43.4 |
| MuSiQue | 2417 | 20.5 | 20.6 | 11.1 |
| SynthWorlds-SM | 1200 | 21.1 | 21.7 | 14.4 |

## Flags and paper notation

| Flag | Paper | Meaning |
|---|---|---|
| `--threshold` | τ_c | Confidence needed to commit a token |
| `--rag_query_confidence_threshold` | τ_q | Confidence needed before a masked position joins the query (0 = expose everything, the default and best-performing setting) |
| `--retrieve_top_k` | K | Passages retrieved per step (7) |
| `--retrieval_steps 0,s` | — | Refresh retrieval at every denoising step |
| `--rag_query_type question_reasoning` | — | Query = question + proxy response (SARDI) |
| `--rag_query_type question` | — | Query = question only (`ret@static`) |

`torch.compile` is on by default (to reproduce paper timing numbers). Set `--nocompile` for quick testing.

## What we changed in Dream's sampler

Dream loads its decoding code via `trust_remote_code`, so the threshold-based sampling code ships inside the checkpoint: It lives in
`checkpoints/sardi-dream-7b/generation_utils.py`. Particularly, we added **`alg="confidence_threshold"`**, which commits every masked position whose
  confidence clears τ_c, instead of a fixed number of tokens per step.

## Citation

If you use SARDI in your work, please cite:

```bibtex
@inproceedings{juenger2026sardi,
  title     = {Self-Augmenting Retrieval for Diffusion Language Models},
  author    = {Paul J{\"u}nger and Justin Lovelace and Linxi Zhao and Dongyoung Go and Kilian Q Weinberger},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
