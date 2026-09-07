# CPAM-LLM

CPAM-LLM is an automated data-to-model pipeline that converts natural-language descriptions of optimization problems into validated, executable Constraint Programming models for IBM CP Optimizer (docplex.cp). It combines RAG-FCA for constraint mapping, chaos-mapping for data augmentation, and two-stage LoRA fine-tuning for CP modeling and code generation.

## Contents

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Data availability](#data-availability)
- [Model weights](#model-weights)
- [Reproducing the results](#reproducing-the-results)
- [Interactive use](#interactive-use)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [License](#license)

## Overview

Given a natural-language problem description, the pipeline proceeds through four stages and a closed validation–repair loop:

```
Natural language
      │
      ▼  Stage 1   natural-language structuring + RAG-FCA retrieval
 structured problem  ───────────  concept-lattice match → formal templates
      │
      ▼  Stage 2   mathematical-model generation        (fine-tuned model)
 formal CP model
      │
      ▼  Stage 3   solver-code generation               (fine-tuned model)
 docplex.cp code  ◀── validate → repair loop ──  static + dynamic checks
      │
      ▼  Solve     run IBM CP Optimizer
 solution
```

Stage 2 (mathematical modeling) and Stage 3 (code generation) are produced by a two-stage LoRA fine-tuned model (base model: Qwen2.5-Coder-7B-Instruct). Natural-language structuring, validation, repair, and dynamic-constraint phrasing use a general-purpose API model.

## System requirements

- **Operating system.** Linux, macOS, or Windows. Tested on Ubuntu 22.04 and Windows 11.
- **Python.** 3.9 or later. The provided Docker image uses Python 3.11.
- **Solver.** A working installation of **IBM ILOG CPLEX Optimization Studio** (CP Optimizer) is required to *solve* generated models. Model generation and validation do not require the solver. CPLEX is proprietary and must be obtained separately from IBM (a free academic edition is available).
- **Hardware.** Inference and the two-stage LoRA fine-tuning of the 7B backbone run on a single 48 GB GPU (e.g. NVIDIA RTX 5880 Ada) under FP16. No multi-GPU or data-center-scale hardware is required.
- **Python dependencies.** Listed in `requirements.txt` (`requests`, `numpy`, `pandas`, `openpyxl`, `docplex`, `Flask`, `flask-cors`, `waitress`).

## Installation

Typical install time on a standard desktop is under five minutes (excluding the CPLEX installation).

```bash
git clone https://github.com/szq85/CPAM-LLM.git
cd CPAM-LLM
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

A containerized environment is provided for a reproducible runtime:

```bash
docker build -t cpam-llm .
docker run --rm -p 5000:5000 -e CPAM_API_KEY="sk-..." cpam-llm
```

If Docker Hub is not reachable, use the mirror-enabled form:

```bash
docker build --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim -t cpam-llm .
```

For model serving and deployment details, see
[`docs/MODEL_SERVING.md`](docs/MODEL_SERVING.md).

Set the API key for the general-purpose model (any OpenAI-compatible endpoint):

```bash
export CPAM_API_KEY="sk-..."          # Windows PowerShell: $env:CPAM_API_KEY="sk-..."
```

The prebuilt knowledge base is already included at `kb_store/knowledge_base.json`,
so no build step is needed for normal use. If you change the source table or the
FCA code, rebuild it with:

```bash
python build_kb.py                    # uses config.DATA_PATH and writes the JSON store
python build_kb.py --xlsx PATH --force # rebuild from an explicit source workbook
```

## Data availability

The datasets used to train and evaluate CPAM-LLM are released with this repository.

## Model weights

The two-stage LoRA adapters are released so that the fine-tuned model can be reconstructed and its outputs verified.

- Adapter weights are included in `models/stage1_lora/` and `models/stage2_lora/`; the Qwen base model itself is not included. See `models/README.md` and `models/base_model_README.md` for the mapping and download instructions.
- The fine-tuned stages are served as OpenAI-compatible endpoints (`FT_URL_STAGE2`, `FT_URL_STAGE3` in `config.py`). If the endpoints are unreachable, the client falls back to the general-purpose API automatically.
- For the full workflow — downloading the base model and adapters, serving them locally on the two endpoints, and connecting the framework — see [`docs/MODEL_SERVING.md`](docs/MODEL_SERVING.md).

## Reproducing the results

```bash
# 1. build the knowledge base (once)
python build_kb.py

# 2. run a single problem end to end and inspect every stage
python main.py

# 3. run the offline end-to-end demonstration (no API key or solver required)
python -X utf8 demo_offline.py
```


## Interactive use

Three interfaces call the same pipeline and the same knowledge base:

```bash
python webapp/app.py     # web app   → http://127.0.0.1:5000  (see webapp/webapp_README.md)
python gui_app.py        # desktop GUI (Tkinter)
python main.py           # command-line, interactive
```

The web interface is documented separately in [`webapp/webapp_README.md`](webapp/webapp_README.md).

## Repository structure

```
.
├── main.py                   # command-line interactive entry point
├── gui_app.py                # Tkinter desktop GUI
├── build_kb.py               # builds kb_store/knowledge_base.json (run once)
├── demo_offline.py           # offline end-to-end demonstration (no API/solver)
├── config.py                 # settings: API keys, stage routing, paths, thresholds
├── requirements.txt
├── Dockerfile
│
├── pipeline/                 # modular generation pipeline
│   ├── main_pipeline.py      #   orchestration and interactive menu
│   ├── nl_structuring.py     #   Stage 1: NL → structured problem (RAG-FCA)
│   ├── math_model.py         #   Stage 2: structured → formal math model
│   ├── cp_code_gen.py        #   Stage 3: math model → docplex.cp code
│   ├── feedback.py           #   iterative validate → repair loop
│   ├── validator.py          #   syntax / static / semantic checks
│   ├── dynamic_constraint.py #   localized model update (add a constraint)
│   └── solver.py             #   sandboxed execution of generated code
│
├── rag_fca/                  # RAG-FCA knowledge base
│   ├── knowledge_base.py     #   formal context, concept lattice, retrieval, update
│   ├── fca.py                #   FCA core: concepts, implication base, rough-set discovery
│   ├── constraints.py        #   named constraint types per scenario (attributes M)
│   └── formalizations.py     #   each constraint → {description, math, code}
│
├── llm/client.py             # OpenAI-compatible chat client (retries, FT → API fallback)
├── augmentation/chaos_augment.py   # chaos-mapping data augmentation
│
├── webapp/                   # Flask web interface (see webapp/webapp_README.md)
├── data/                     # domain inputs, training archive, and test set
├── kb_store/                 # built knowledge base + human-readable exports
├── models/                   # two-stage LoRA adapter weights
├── configs/                  # augmentation seeds and LoRA configuration
├── docs/                     # deployment, serving, and curation documentation
└── output/                   # generated results (JSON / CSV per task)
```

## Configuration

Settings are centralized in `config.py`. The most relevant:

| Setting                              | Meaning                               | Default                                               |
| ------------------------------------ | ------------------------------------- | ----------------------------------------------------- |
| `CPAM_API_KEY` (env)               | API key for the general-purpose model | —                                                    |
| `CPAM_API_BASE_URL` / `API_BASE_URL` | OpenAI-compatible endpoint          | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `CPAM_API_MODEL` / `API_MODEL`     | API model name                        | `qwen-XXX` (set explicitly for your endpoint)      |
| `FT_URL_STAGE2`, `FT_URL_STAGE3` | fine-tuned model endpoints            | `http://localhost:6006/v1`, `:6008/v1`            |
| `STAGE_ROUTING`                    | per-stage choice of fine-tuned vs API | math/code →`ft`; others → `api`                 |
| `KB_UPDATE_POLICY`                 | knowledge-base admission gate         | `strict`                                            |
| `MAX_FEEDBACK_ROUNDS`              | validate → repair iterations         | `3`                                                 |
| `TOP_K_RETRIEVAL`                  | concepts / records retrieved          | `3`                                                 |
| `SOLVER_TIME_LIMIT`                | CP solver time limit (s)              | `60`                                                |

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
