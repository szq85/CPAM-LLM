# Fine-tuned model weights (LoRA adapters)

This directory holds the two-stage LoRA adapters that CPAM-LLM uses for
mathematical-model generation (Stage 2) and solver-code generation (Stage 3).
The full download-and-serve workflow is in
[`../docs/MODEL_SERVING.md`](../docs/MODEL_SERVING.md); this file documents the
contents and configuration of the weights themselves.

## Base model

The adapters are applied on top of **Qwen2.5-Coder-7B-Instruct**. The base model
is downloaded separately (it is not stored here):

```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir ./models/base/Qwen2.5-Coder-7B-Instruct
```

## Layout

```
models/
├── base/
│   └── Qwen2.5-Coder-7B-Instruct/   # base model (downloaded; not in this repo)
├── stage1_lora/                     # math-modeling adapter   → served as "stage2" on port 6006
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── stage2_lora/                     # code-generation adapter  → served as "stage3" on port 6008
    ├── adapter_config.json
    └── adapter_model.safetensors
```

The folders are named by training order (`stage1_lora`, `stage2_lora`), while the
pipeline phases that use them are *Stage 2* (math) and *Stage 3* (code). The
mapping is:

| Adapter folder | Pipeline stage | Task | Served as | Port |
|----------------|----------------|------|-----------|------|
| `stage1_lora` | Stage 2 | structured description → formal CP model | `stage2` | 6006 |
| `stage2_lora` | Stage 3 | formal CP model → `docplex.cp` code | `stage3` | 6008 |

## LoRA configuration

The full configuration is in [`../configs/lora_config.yaml`](../configs/lora_config.yaml)
and matches **Supplementary Table 13**: LoRA applied to all linear layers,
learning rate `1.0e-4` with a cosine schedule, 3 epochs, per-device batch size 1
with gradient accumulation 16, FP16 precision, and a maximum sequence length of
4096. Both stages share this configuration. The rank, alpha, and dropout values
are listed in `configs/lora_config.yaml`.

## Distribution

The adapter files are included in this repository. Only the base Qwen model is
external because of its size; download it as described in
[`base_model_README.md`](base_model_README.md).

## Serving

`config.py` expects the two adapters to be served as OpenAI-compatible endpoints
(`FT_URL_STAGE2` → `http://localhost:6006/v1`, served model name `stage2`;
`FT_URL_STAGE3` → `http://localhost:6008/v1`, served model name `stage3`). The
exact vLLM and LLaMA-Factory commands, verification steps, and troubleshooting
are in [`../docs/MODEL_SERVING.md`](../docs/MODEL_SERVING.md). If the endpoints
are not running, the pipeline falls back to the general-purpose API automatically.
