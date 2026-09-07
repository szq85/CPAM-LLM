# scripts/

This is a placeholder directory for optional helper scripts. **It is not required
to run CPAM-LLM** — every entry point already exists at the project root and
works without anything here.

## Entry points (at the project root)

| Command | What it does |
|---------|--------------|
| `python build_kb.py` | Build the concept-lattice knowledge base once → `kb_store/knowledge_base.json` (`--force` to rebuild). |
| `python main.py` | Command-line interactive pipeline (`--show-kb` prints KB statistics). |
| `python webapp/app.py` | Web interface → `http://127.0.0.1:5000` (see `webapp/webapp_README.md`). |
| `python gui_app.py` | Tkinter desktop GUI. |
| `python -X utf8 demo_offline.py` | End-to-end demonstration with canned outputs (no API key or solver needed). |
| `python main.py --batch-solve` | Batch generation and solving for the five built-in demo scenarios. |

## Serving the fine-tuned model

Stage 2 and Stage 3 run on local model endpoints. The commands to download and
serve the adapters are documented in
[`../docs/MODEL_SERVING.md`](../docs/MODEL_SERVING.md), including ready-to-run
vLLM and LLaMA-Factory invocations for ports 6006 (`stage2`) and 6008 (`stage3`).

## Adding your own scripts

If you want one-command reproduction, a convenience wrapper can be placed here,
for example a shell script that builds the knowledge base and then runs the batch
evaluation:

```bash
#!/usr/bin/env bash
set -e
python build_kb.py
python main.py --batch-solve
```

This is optional; the entry points above are the supported way to run the
framework.
