# The Kepler Architecture

Research notebooks exploring novel SLM (Small Language Model) components, including activation functions, optimizers, manifold-constrained residuals, and differential latent attention. The repo is notebook‑first and intended to be run in a GPU-enabled environment (A100 recommended for the benchmarks).

## Repository Structure

All work is contained in the following notebooks:

| Notebook | Focus | Highlights |
| --- | --- | --- |
| `Untitled27.ipynb` | Skeleton & Flesh training config + streaming data pipeline | C4 streaming dataset, GPT‑2 tokenizer, packed token pipeline. |
| `Untitled28.ipynb` | Activation research + micro‑LLaMA experiments | ReLU²/Swish/TeLU/CRReLU, A100 latency benchmarks, CRReLU‑GLU MLP. |
| `Untitled29.ipynb` | Optimizer research | FAdam, Adam‑Mini variants, Llama‑style synthetic benchmarks and memory inspection. |
| `Untitled30.ipynb` | Triton Sinkhorn + Manifold‑Constrained Hyper‑Connections (mHC) | Custom Sinkhorn projection, mHC layer, LLaMA‑style block. |
| `Untitled32.ipynb` | Differential Multi‑Head Latent Attention (Diff‑MLA) | TinyShakespeare training throughput and KV‑cache compression analysis. |

## Quickstart (Recommended)

1. **Install dependencies** (recommended to pin versions in a virtual environment):
   ```bash
   pip install torch transformers datasets einops triton matplotlib numpy tqdm
   ```
2. **Open notebooks** in this order for the cleanest narrative:
   1. `Untitled27.ipynb` — data pipeline + config baseline
   2. `Untitled28.ipynb` — activation research + micro‑LLaMA tests
   3. `Untitled29.ipynb` — optimizer benchmarks
   4. `Untitled30.ipynb` — mHC kernel + stability checks
   5. `Untitled32.ipynb` — Diff‑MLA model and KV cache analysis

> **GPU note:** Several notebooks assume CUDA and/or A100 hardware for performance benchmarks. CPU execution may be slow and some timing sections may be skipped or need guarding.

## External Data & Network Usage

These notebooks download data or models at runtime:

- **Hugging Face datasets**:
  - `allenai/c4` (streaming) in `Untitled27.ipynb`.
- **Hugging Face models**:
  - GPT‑2 tokenizer (via `transformers`).
- **TinyShakespeare**:
  - Downloaded via `wget` in `Untitled32.ipynb`.

If you are offline, cache these artifacts in advance or add a local fallback path.

## Reproducibility Tips

- Set deterministic seeds (`torch.manual_seed`, `torch.cuda.manual_seed_all`) before running experiments.
- For benchmarking, prefer consistent GPU hardware and fixed batch/sequence sizes.
- If you modify notebooks, consider saving results and system metadata (GPU type, driver, CUDA version).

## Contributing

If you add a new experiment, follow the notebook-first pattern and update this README with:

- the notebook name,
- its purpose,
- dependencies/datasets,
- and where it fits in the recommended order.
