# Docker installation

This repository provides a lightweight Docker image for the CPAM-LLM web
runtime. The image contains the Python application, runtime dependencies,
example data, and a prebuilt RAG-FCA knowledge base.

The image intentionally does not include:

- IBM ILOG CPLEX Optimization Studio / CP Optimizer.
- Qwen base-model weights, LoRA adapter weights, vLLM, LLaMA-Factory, CUDA, or
  other GPU inference dependencies.
- Training archives.

Those components are external to the web/runtime container.

## Build

Run from the repository root:

```bash
docker build -t cpam-llm:latest .
```

If Docker Hub is not reachable from your network, use an equivalent mirror for
the Python base image:

```bash
docker build \
  --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim \
  -t cpam-llm:latest .
```

The build runs `python build_kb.py --force --no-review` so the container can
serve the web UI immediately. Rebuild the image after changing files under
`data/` or the RAG-FCA modules.

## Run with the cloud API fallback

```bash
docker run --rm \
  -p 5000:5000 \
  -e CPAM_API_KEY="sk-..." \
  cpam-llm:latest
```

Open <http://localhost:5000>.

The default cloud endpoint is DashScope's OpenAI-compatible endpoint. Override
it if needed:

```bash
docker run --rm \
  -p 5000:5000 \
  -e CPAM_API_KEY="sk-..." \
  -e CPAM_API_BASE_URL="https://your-openai-compatible-host/v1" \
  -e CPAM_API_MODEL="your-model-name" \
  cpam-llm:latest
```

## Connect external fine-tuned model servers

Stage 2 and Stage 3 are expected to be served as OpenAI-compatible HTTP
endpoints outside this image. See `docs/MODEL_SERVING.md` for the vLLM and
LLaMA-Factory commands.

On Docker Desktop for Windows or macOS, use `host.docker.internal` to reach
model servers running on the host:

```bash
docker run --rm \
  -p 5000:5000 \
  -e CPAM_API_KEY="sk-..." \
  -e FT_URL_STAGE2="http://host.docker.internal:6006/v1" \
  -e FT_URL_STAGE3="http://host.docker.internal:6008/v1" \
  cpam-llm:latest
```

On Linux, either use the host gateway name supported by your Docker version or
run with host networking:

```bash
docker run --rm --network host \
  -e CPAM_API_KEY="sk-..." \
  -e FT_URL_STAGE2="http://127.0.0.1:6006/v1" \
  -e FT_URL_STAGE3="http://127.0.0.1:6008/v1" \
  cpam-llm:latest
```

If the fine-tuned endpoints are unavailable, CPAM-LLM falls back to the cloud
API for those calls.

## Solver support

`docplex` is installed in the image because generated code imports
`docplex.cp`. Actually solving generated CP models requires IBM ILOG CPLEX
Optimization Studio / CP Optimizer and a valid license.

Because CP Optimizer is proprietary, this Dockerfile does not install it. The
container can still generate, validate, repair, and display models without CP
Optimizer. To solve inside a container, build a private derived image or mount
your licensed CP Optimizer installation and set the IBM environment variables
required by your installation, for example `PATH`, `LD_LIBRARY_PATH`, and
`ILOG_LICENSE_FILE`.

## Runtime environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `CPAM_API_KEY` | empty | API key for the cloud OpenAI-compatible endpoint. |
| `CPAM_API_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Cloud OpenAI-compatible base URL. |
| `CPAM_API_MODEL` | `qwen-plus` | Cloud model name. |
| `FT_URL_STAGE2` | `http://localhost:6006/v1` | Fine-tuned Stage 2 endpoint. In Docker Desktop, usually use `host.docker.internal`. |
| `FT_URL_STAGE3` | `http://localhost:6008/v1` | Fine-tuned Stage 3 endpoint. |
| `FT_API_KEY` | `0` | Dummy key for local fine-tuned endpoints unless your server requires one. |
| `MAX_FEEDBACK_ROUNDS` | `3` | Validation-repair iterations. |
| `ENABLE_LLM_SEMANTIC_CHECK` | `0` | Enable LLM semantic validation. |
| `DYNAMIC_SOLVE_TIME_LIMIT` | `30` | Solve time limit for dynamic validation attempts. |

## Troubleshooting

- If the web UI is not reachable, confirm the container is exposing `5000` and
  that `docker ps` shows the container as healthy.
- If Stage 2 or Stage 3 falls back to the cloud API, check the corresponding
  `FT_URL_STAGE*` value from inside the container network.
- If solving fails with a `docplex` or `cpoptimizer` error, install or mount IBM
  CP Optimizer. `pip install docplex` alone is not sufficient for solving.
