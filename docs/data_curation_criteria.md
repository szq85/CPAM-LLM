# Data-curation / manual-review criteria

Promised in the response to Reviewer 1 (R1-7) and documented in
**Supplementary Section F.1.6**. This file makes the closed-loop sample-admission
rule explicit and checkable, exactly as described in the response letter.

A generated sample is **retained** (admitted to the knowledge base and the
fine-tuning pool) only if it satisfies **all** of the following conditions:

1. **Parses cleanly.** The code parses without syntax errors and uses the
   required CPLEX CP Optimizer (`docplex.cp`) modeling constructs.
2. **Matches the intended model.** The structure and content match the intended
   formal model.
3. **Executes successfully.** The code runs in the solver with status `solved`
   (a separate hard gate, not a graded contribution).
4. **Meets the quality threshold.** The composite quality score q ∈ [0, 1]
   (Supplementary Section F.1.6, "Composite quality score for closed-loop sample
   admission") exceeds the preset threshold.
5. **Is not a near-duplicate.** A near-duplicate guard rejects candidates too
   similar to existing records.

Samples that fail any criterion are **rejected or returned for correction**, not
retained.

> Keep the wording here aligned with Supplementary Section F.1.6 so the
> repository and the Supplementary Information do not diverge.

The knowledge-base update policy is controlled in `config.py` via
`KB_UPDATE_POLICY` (`strict` | `relaxed` | `manual`).
