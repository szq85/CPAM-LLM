# Base model — Qwen2.5-Coder-7B-Instruct

**The base model is not included in this repository.** Its weights are several
gigabytes (≈15 GB in FP16 safetensors), which exceeds the GitHub upload size, so
it must be downloaded separately and placed in this folder. The fine-tuned LoRA
adapters in [`../`](..) are applied *on top of* this base model; they are small
and are distributed with the project, but they cannot run without the base model
below.

## What to download

| | |
|---|---|
| Model | **Qwen2.5-Coder-7B-Instruct** |
| Source | Hugging Face: [`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| Parameters | 7B |
| Context length | up to 128K tokens |
| License | Apache 2.0 |
| Approx. size | ≈15 GB (FP16 safetensors) |

This is the exact backbone used in the paper. Do not substitute another size
(0.5B/1.5B/3B/14B/32B) or a quantized GGUF build for reproduction, as the
fine-tuning and the reported results are tied to the 7B-Instruct weights.

## How to download

**Option A — Hugging Face CLI (recommended):**

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir ./models/base/Qwen2.5-Coder-7B-Instruct
```

**Option B — git + git-lfs:**

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct \
    ./models/base/Qwen2.5-Coder-7B-Instruct
```

If Hugging Face is slow or blocked in your region, the same repository is
mirrored on ModelScope (`Qwen/Qwen2.5-Coder-7B-Instruct`); set
`HF_ENDPOINT=https://hf-mirror.com` before the CLI download to use a mirror.

## Where to put it

After downloading, this directory should contain the model folder:

```
models/
├── base_model_README.md                # this file
├── base/
│   └── Qwen2.5-Coder-7B-Instruct/      # ← downloaded base model
│       ├── config.json
│       ├── model-*.safetensors
│       ├── tokenizer.json
│       └── ...
├── stage1_lora/                        # math-modeling adapter (included in repo)
└── stage2_lora/                        # code-generation adapter (included in repo)
```

## Role in CPAM-LLM

The base model is the frozen backbone for both fine-tuning stages. During
training, the base weights are kept frozen and only the low-rank LoRA matrices
are updated, which is what keeps the released adapters small. At inference, the
base model is loaded once and the two adapters are attached to serve:

- **Stage 2** (mathematical-model generation) — served as model name `stage2` on port 6006;
- **Stage 3** (solver-code generation) — served as model name `stage3` on port 6008.

The exact serving commands (vLLM / LLaMA-Factory), verification steps, and
troubleshooting are in [`../docs/MODEL_SERVING.md`](../docs/MODEL_SERVING.md).
Qwen recommends vLLM for deployment. The adapter folders, the LoRA configuration,
and the folder-to-stage mapping are documented in [`../README.md`](../README.md).
