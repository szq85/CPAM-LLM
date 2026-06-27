# Serving the fine-tuned model locally

CPAM-LLM produces the mathematical model (Stage 2) and the solver code (Stage 3)
with a **two-stage LoRA fine-tuned** model. The framework does not load the model
in-process; instead it calls two **local OpenAI-compatible HTTP endpoints**. This
document describes the full flow: download the artifacts from the server, place
them locally, serve the two adapters, and connect the framework.

## What the framework expects

The two endpoints and their model names are fixed in `config.py`:

| Stage | Pipeline phase | URL (default) | Served model name | Env override |
|-------|----------------|---------------|-------------------|--------------|
| Stage 2 | mathematical-model generation | `http://localhost:6006/v1` | `stage2` | `FT_URL_STAGE2` |
| Stage 3 | solver-code generation | `http://localhost:6008/v1` | `stage3` | `FT_URL_STAGE3` |

Key points:

- The endpoints must speak the **OpenAI Chat Completions** protocol
  (`POST /v1/chat/completions`). vLLM and LLaMA-Factory both provide this.
- The served model name **must be exactly `stage2` and `stage3`** — the client
  sends `model="stage2"` / `model="stage3"` and the server matches on it.
- The API key is a dummy value (`FT_API_KEY = "0"` in `config.py`); local servers
  do not check it.
- The base model is **Qwen2.5-Coder-7B-Instruct**; Stage 2 and Stage 3 are two
  **LoRA adapters** applied on top of that same base model.
- Stage 2's endpoint is also the readiness gate (`FT_BASE_URL = FT_URL_STAGE2`).
  If it is unreachable, the client prints a warning and **falls back to the cloud
  API** for that call rather than failing.

If you only want to try the pipeline without the fine-tuned model, do nothing:
leave the endpoints down and every stage runs on the cloud API (`config.py` →
`API_BASE_URL` / `API_MODEL`). The steps below are for running the actual
fine-tuned model.

## Step 1 — Download the artifacts from the server

You need the base model and the two LoRA adapters.

**Base model (Qwen2.5-Coder-7B-Instruct).** Download once from Hugging Face:

```bash
pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir ./models/base/Qwen2.5-Coder-7B-Instruct
```

**LoRA adapters (Stage 2 and Stage 3).** These are the fine-tuned weights released
with this work. Download them from the release location given in
[`../models/README.md`](../models/README.md) (a DOI-minting repository, e.g.
Zenodo or Hugging Face, since the adapters exceed the standard GitHub upload
size). Place them so the tree looks like:

```
models/
├── base/
│   └── Qwen2.5-Coder-7B-Instruct/      # downloaded base model
├── stage1_lora/                        # Stage-2 adapter (math modeling)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── stage2_lora/                        # Stage-3 adapter (code generation)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

> Note on naming: the two pipeline stages that use the fine-tuned model are
> *Stage 2* (math) and *Stage 3* (code). The adapter folders may be named by
> training order (`stage1_lora`, `stage2_lora`); what matters is that you serve
> the math adapter as model name `stage2` on port 6006 and the code adapter as
> model name `stage3` on port 6008, as shown below. Confirm which folder is which
> in `models/README.md`.

If you transfer the artifacts from a remote training server rather than a
public repository, copy them down with `scp` / `rsync`, for example:

```bash
rsync -avz user@server:/path/to/CPAM_LLM/models/ ./models/
```

## Step 2 — Serve the two adapters locally

Pick one of the two serving backends below. Both expose the required
OpenAI-compatible endpoints. Each command starts **one** server; run the Stage 2
and Stage 3 commands in **two separate terminals** (or as two background
processes), because the framework expects them on two different ports.

### Option A — vLLM (recommended)

```bash
pip install vllm
```

Terminal 1 — Stage 2 (math modeling) on port 6006, model name `stage2`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model        ./models/base/Qwen2.5-Coder-7B-Instruct \
    --enable-lora \
    --lora-modules stage2=./models/stage1_lora \
    --served-model-name stage2 \
    --port 6006 \
    --dtype float16 \
    --max-model-len 8192
```

Terminal 2 — Stage 3 (code generation) on port 6008, model name `stage3`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model        ./models/base/Qwen2.5-Coder-7B-Instruct \
    --enable-lora \
    --lora-modules stage3=./models/stage2_lora \
    --served-model-name stage3 \
    --port 6008 \
    --dtype float16 \
    --max-model-len 8192
```

The `--lora-modules NAME=PATH` flag tells vLLM to load the adapter at `PATH` and
expose it under model id `NAME`. Setting `--served-model-name` to the same value
makes `model="stage2"` / `model="stage3"` resolve correctly.

> **One GPU, two servers.** Each server holds its own copy of the 7B base model.
> On a single 48 GB GPU you can either give each server a fraction of memory
> (add `--gpu-memory-utilization 0.45` to both commands) or run them on two GPUs
> (prefix each with `CUDA_VISIBLE_DEVICES=0` and `=1`).

### Option B — LLaMA-Factory

If the model was fine-tuned with LLaMA-Factory, serve each adapter with its
OpenAI-compatible API. Stage 2 (port 6006):

```bash
API_PORT=6006 llamafactory-cli api \
    --model_name_or_path ./models/base/Qwen2.5-Coder-7B-Instruct \
    --adapter_name_or_path ./models/stage1_lora \
    --template qwen \
    --finetuning_type lora \
    --infer_dtype float16
```

Stage 3 (port 6008):

```bash
API_PORT=6008 llamafactory-cli api \
    --model_name_or_path ./models/base/Qwen2.5-Coder-7B-Instruct \
    --adapter_name_or_path ./models/stage2_lora \
    --template qwen \
    --finetuning_type lora \
    --infer_dtype float16
```

LLaMA-Factory serves a single model per process; the framework matches on the
URL (port) for each stage, so the exact model-name string is less critical with
this backend than with vLLM. If a stage call still fails to match a model name,
set `FT_MODEL` in `config.py` to the name the server reports at
`GET /v1/models`.

## Step 3 — Point the framework at the endpoints (only if non-default)

The defaults already match the commands above (`:6006` and `:6008` on
localhost), so usually nothing is needed. Override only if you serve elsewhere:

```bash
export FT_URL_STAGE2="http://localhost:6006/v1"
export FT_URL_STAGE3="http://localhost:6008/v1"
```

Routing is controlled by `STAGE_ROUTING` in `config.py`: `math_modeling` and
`code_generation` are set to `ft`, so those two stages use the endpoints above;
all other phases use the cloud API. You do not need to change this.

## Step 4 — Verify the endpoints

Before running the pipeline, confirm both servers respond. Each should list its
model and answer a chat request:

```bash
# Stage 2
curl http://localhost:6006/v1/models
curl http://localhost:6006/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"stage2","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'

# Stage 3
curl http://localhost:6008/v1/models
curl http://localhost:6008/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"stage3","messages":[{"role":"user","content":"ping"}],"max_tokens":8}'
```

A JSON response with a `choices` field means the endpoint is ready.

## Step 5 — Run CPAM-LLM

With both servers up, run any interface as usual:

```bash
python main.py            # command line
python webapp/app.py      # web app  → http://127.0.0.1:5000
```

During generation the console prints which backend each stage used, e.g.:

```
[Backend:math_modeling] FT  adapter=stage2  @ http://localhost:6006/v1
[Backend:code_generation] FT  adapter=stage3  @ http://localhost:6008/v1
```

If you instead see a fall-back warning
(`Fine-tuned model … error … falling back to API`), the corresponding endpoint
is not reachable — re-check Step 2 and Step 4.