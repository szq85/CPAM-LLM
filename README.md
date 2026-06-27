# CPAM-LLM

CPAM-LLM is an automated data-to-model pipeline that converts an unstructured natural-language description of an optimization problem into a validated, executable Constraint-Programming (CP) model for IBM CP Optimizer (`docplex.cp`). The method integrates three components evaluated in the paper: a Retrieval-Augmented Generation framework grounded in Formal Concept Analysis (**RAG-FCA**) for structurally grounded constraint mapping; a **chaos-mapping** data-augmentation strategy for corpus construction; and a **two-stage LoRA** fine-tuning process that adapts a general-purpose language model to CP modeling and to CP Optimizer code. The framework is validated on five real-world applications — aircraft-skin processing, DNA nanostructure sequence design, reconfigurable photovoltaic energy storage, unmanned-vehicle logistics, and electric-vehicle charging-station location.

This README describes how to install the software, obtain the data and model weights, and reproduce the main results. It also documents the repository layout and the interactive interfaces used in the paper.

## Citation

If you use this code, please cite.

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

Stage 2 (mathematical modeling) and Stage 3 (code generation) are produced by a two-stage LoRA fine-tuned model (base model: Qwen2.5-Coder-7B-Instruct). Natural-language structuring, validation, repair, and dynamic-constraint phrasing use a general-purpose API model. Per-stage routing is configured in `config.py` (`STAGE_ROUTING`).

The RAG-FCA knowledge base is a formal concept lattice `L(K)` over a formal context `K = (G, M, I)`, where `G` is the set of problem instances, `M` is the set of named constraint types, objectives, and solution properties, and `I` is their incidence. Retrieval returns a formalized constraint structure (mathematical and code templates), which is what enables Stages 2 and 3 to produce the formal model and solver code. The released knowledge base contains 140 records, 138 concepts, 54 attributes, and 68 implications.

## System requirements

- **Operating system.** Linux, macOS, or Windows. Tested on Ubuntu 22.04 and Windows 11.
- **Python.** 3.9 or later. The provided Docker image uses Python 3.11.
- **Solver.** A working installation of **IBM ILOG CPLEX Optimization Studio** (CP Optimizer) is required to *solve* generated models. Model generation and validation do not require the solver. CPLEX is proprietary and must be obtained separately from IBM (a free academic edition is available).
- **Hardware.** Inference and the two-stage LoRA fine-tuning of the 7B backbone run on a single 48 GB GPU (e.g. NVIDIA RTX 5880 Ada) under FP16. No multi-GPU or data-center-scale hardware is required.
- **Python dependencies.** Listed in `requirements.txt` (`requests`, `numpy`, `pandas`, `openpyxl`, `docplex`, `Flask`, `flask-cors`, `waitress`).

## Installation

Typical install time on a standard desktop is under five minutes (excluding the CPLEX installation).

```bash
git clone https://github.com/szq85/CPAM_LLM.git
cd CPAM_LLM
pip install -r requirements.txt
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

For the full environment checklist and Docker deployment commands, see
[`docs/INSTALL_DOCKER.md`](docs/INSTALL_DOCKER.md).

Set the API key for the general-purpose model (any OpenAI-compatible endpoint):

```bash
export CPAM_API_KEY="sk-..."          # Windows PowerShell: $env:CPAM_API_KEY="sk-..."
```

Build the knowledge base once; afterwards every entry point loads the resulting JSON file directly:

```bash
python build_kb.py                    # writes kb_store/knowledge_base.json; add --force to rebuild
```

## Data availability

The datasets used to train and evaluate CPAM-LLM are released with this repository.

- **Benchmark and case-study data** read by the generated solver code are in `data/` (one file per application domain).
- **Training and test corpora.** For each sample, the natural-language description, the structured formulation, the formal CP model, and the executable solver code are provided. See `data/train/README.md` for the per-sample schema and the location of the corpus files.
- **Chaos-mapping augmentation seeds** (initial state, control parameter, per-level perturbation rates, iteration count) are in `configs/augmentation_seeds.yaml`, so the augmented corpus can be regenerated deterministically.

Large data files that exceed the GitHub upload size are deposited in a persistent DOI-minting repository; the link is provided in the manuscript's Data-availability statement and in `data/train/README.md`.

## Model weights

The two-stage LoRA adapters are released so that the fine-tuned model can be reconstructed and its outputs verified.

- Adapter weights are placed in `models/stage1_lora/` and `models/stage2_lora/`; see `models/README.md` for the base-model identifier (Qwen2.5-Coder-7B-Instruct), the LoRA configuration, and the serving command.
- The fine-tuned stages are served as OpenAI-compatible endpoints (`FT_URL_STAGE2`, `FT_URL_STAGE3` in `config.py`). If the endpoints are unreachable, the client falls back to the general-purpose API automatically.
- For the full workflow — downloading the base model and adapters, serving them locally on the two endpoints, and connecting the framework — see [`docs/MODEL_SERVING.md`](docs/MODEL_SERVING.md).

Adapter weights that exceed the GitHub upload size are deposited in the same persistent repository as the data and linked from `models/README.md`.

## Reproducing the results

```bash
# 1. build the knowledge base (once)
python build_kb.py

# 2. run a single problem end to end and inspect every stage
python main.py

# 3. run the offline end-to-end demonstration (no API key or solver required)
python demo_offline.py
```

`python main.py --show-kb` prints the knowledge-base statistics reported in the paper (records, concepts, attributes, implications). Generated results are written per task to `output/` as JSON and CSV.

The data-curation and closed-loop sample-admission criteria used to build the corpus and to admit new samples to the knowledge base are documented in `docs/data_curation_criteria.md`.

## Interactive use

Three interfaces call the same pipeline and the same knowledge base:

```bash
python webapp/app.py     # web app   → http://127.0.0.1:5000  (see webapp/README.md)
python gui_app.py        # desktop GUI (Tkinter)
python main.py           # command-line, interactive
```

The web interface is documented separately in [`webapp/README.md`](webapp/README.md).

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
├── webapp/                   # Flask web interface (see webapp/README.md)
├── data/                     # datasets and training/test corpora
├── kb_store/                 # built knowledge base + human-readable exports
├── models/                   # two-stage LoRA adapter weights
├── configs/                  # augmentation seeds and LoRA configuration
├── docs/                     # data-curation criteria
└── output/                   # generated results (JSON / CSV per task)
```

## Configuration

Settings are centralized in `config.py`. The most relevant:

| Setting                              | Meaning                               | Default                                               |
| ------------------------------------ | ------------------------------------- | ----------------------------------------------------- |
| `CPAM_API_KEY` (env)               | API key for the general-purpose model | —                                                    |
| `CPAM_API_BASE_URL` / `API_BASE_URL` | OpenAI-compatible endpoint          | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `CPAM_API_MODEL` / `API_MODEL`     | API model name                        | `qwen-plus`                                         |
| `FT_URL_STAGE2`, `FT_URL_STAGE3` | fine-tuned model endpoints            | `http://localhost:6006/v1`, `:6008/v1`            |
| `STAGE_ROUTING`                    | per-stage choice of fine-tuned vs API | math/code →`ft`; others → `api`                 |
| `KB_UPDATE_POLICY`                 | knowledge-base admission gate         | `strict`                                            |
| `MAX_FEEDBACK_ROUNDS`              | validate → repair iterations         | `3`                                                 |
| `TOP_K_RETRIEVAL`                  | concepts / records retrieved          | `3`                                                 |
| `SOLVER_TIME_LIMIT`                | CP solver time limit (s)              | `60`                                                |

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
