# Training data

This directory contains the training archive used by CPAM-LLM. The archive is
kept separate from the runtime knowledge base and the test set.

## Overview

The dataset spans **five real-world application domains**:

- **DNA Sequence Design** — synthetic DNA sequence design under biochemical constraints;
- **Battery Pack Design** — reconfigurable battery energy-storage scheduling;
- **Vehicle Routing Problem** — capacitated vehicle routing / logistics;
- **Charging Station Location** — electric-vehicle charging-station facility location;
- **Aircraft Skin Processing** — flexible job-shop scheduling.

`train_data.zip` contains two JSON arrays:

- `train_data/stage2_formal2model.json`: structured problem formulation to formal CP model examples;
- `train_data/stage3_model2code.json`: formal CP model to executable `docplex.cp` code examples.

The two files are JSON arrays (not JSONL files) and are aligned by record order.
They support the two fine-tuning
stages: *description → formal model* and *formal model → code*. The five
domains are DNA sequence design, battery-pack design, vehicle routing,
charging-station location, and aircraft-skin processing.

The current smoke-test set is separate at
[`../test/test_dataset.json`](../test/test_dataset.json) and contains 10
records (base plus first variant for each domain).
