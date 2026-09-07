# CPAM-LLM — web interface

This directory contains the browser interface for CPAM-LLM. It is a thin
front-end over the pipeline described in the top-level
[`README`](../README.md) and adds no modeling logic of its own; the same
pipeline is also available through the command line (`main.py`) and the desktop
GUI (`gui_app.py`). The interface exposes each stage of the pipeline —
structured problem, RAG-FCA concept match, mathematical model, solver code, and
solution — so that the data-to-model process can be followed interactively.

## Requirements

The web interface adds `flask` and `flask-cors` to the dependencies listed in
the top-level `requirements.txt`. It uses the same configuration as the rest of
the project (`config.py`): the same API key (`CPAM_API_KEY`), the same
per-stage routing (`STAGE_ROUTING`), and the same knowledge base. Solving a
generated model requires a working IBM CP Optimizer installation, as described
in the top-level README.

## Running

```bash
python build_kb.py                 # optional: rebuild only after changing the source table or FCA code
pip install flask flask-cors       # one-time
export CPAM_API_KEY="sk-..."        # Windows PowerShell: $env:CPAM_API_KEY="sk-..."
python webapp/app.py
```

The server binds to `127.0.0.1:5000`. Open <http://127.0.0.1:5000> in a
browser. If the fine-tuned endpoints in `config.py` are not running, the
interface falls back to the general-purpose API automatically.

## Interface

The page moves through three views:

1. **Scenario selection.** Choose one of the bundled application scenarios —
   aircraft-skin processing, DNA sequence design, battery-pack design,
   logistics / VRP, or charging-station location — to pre-fill a problem
   description, or begin from a blank description.
2. **Workspace.** A two-panel view. The left panel is a conversation in which
   the base problem is described and, in subsequent messages, additional
   constraints are added; each message updates the model in place. The right
   panel shows the structured constraints, the objective, the RAG-FCA implicit
   (certain) constraints surfaced by the concept lattice, and the resolved
   formal templates.
3. **Code view.** The generated `docplex.cp` code, with actions to re-validate
   the code, load the example dataset, run the CP solver, and admit the result
   to the knowledge base. The solver output appears below the code.

## HTTP routes

The front-end is driven by JSON routes defined in `app.py`:

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serve the single-page interface. |
| `/api/scenes` | GET | List the bundled scenario cards. |
| `/api/kbinfo` | GET | Knowledge-base summary (records, concepts, attributes). |
| `/api/generate` | POST | Stage 1: structure the problem and run RAG-FCA retrieval. |
| `/api/formalize` | POST | Produce the formal view of the structured problem. |
| `/api/build_model` | POST | Stage 2: generate the formal mathematical model. |
| `/api/build_code` | POST | Stage 3: generate the `docplex.cp` solver code. |
| `/api/add_constraint` | POST | Add one constraint to the current model (localized update). |
| `/api/check` | POST | Re-run static and semantic validation on the current code. |
| `/api/fix` | POST | Run one validate → repair iteration. |
| `/api/solve` | POST | Execute the generated code on the matching dataset. |
| `/api/save_kb` | POST | Admit the current result to the knowledge base. |

## Files

```
webapp/
├── app.py                 # Flask backend: the pipeline exposed as JSON routes
├── templates/index.html   # single-page front-end
├── static/logo.svg
└── webapp_README.md       # this file
```

## Notes

- Generation, validation, and constraint editing work without a solver; running
  the solver requires IBM CP Optimizer.
- If the interface appears unchanged, confirm that the web app
  (`python webapp/app.py`) was launched rather than the desktop GUI
  (`python gui_app.py`).
- If port 5000 is already in use, stop the conflicting process or change the
  port in the `app.run(...)` call at the end of `app.py`.
