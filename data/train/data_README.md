# Dataset

This directory holds the data used to build, train, and evaluate CPAM-LLM.
The dataset is the foundation of the work: because high-quality, structured
training data are scarce in constraint programming, CPAM-LLM is built on a
purpose-made corpus of aligned *(natural-language description, formal CP model,
solver code)* triples, expanded by a deterministic chaos-mapping augmentation
strategy and organized into two linked corpora for the two-stage fine-tuning.

## Overview

The dataset spans **five real-world application domains**:

- **DNA Sequence Design** — synthetic DNA sequence design under biochemical constraints;
- **Battery Pack Design** — reconfigurable photovoltaic energy-storage scheduling;
- **Vehicle Routing Problem** — capacitated vehicle routing / logistics;
- **Charging Station Location** — electric-vehicle charging-station facility location;
- **Aircraft Skin Processing** — multi-robot flexible job-shop scheduling.

Every instance is an aligned triple:

1. a **natural-language description** of the optimization problem;
2. a **formal CP model** (decision variables, constraints, objective) in CPLEX CP Optimizer notation;
3. **executable `docplex.cp` solver code** that reads the corresponding input file and solves the instance.

The three representations are kept consistent with one another, which is what
allows the two-stage model to learn *description → formal model* and then
*formal model → code*.