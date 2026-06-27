# Knowledge-base files — what the framework actually uses

This project builds the RAG-FCA knowledge base **once** and stores it as a
single permanent JSON. The pipeline then loads only that one file.

## Code generation follows the KB base template

The code stage does **not** invent toy data. For each scenario the knowledge
base stores a base CP model (`base_template`) that reads the real input file in
the documented format. Stage 3 uses that template as the **starting point to
extend**: it keeps the template's data-loading and structure verbatim and adds
the constraints from the concept-lattice checklist. The fix/feedback pass
receives the current code and repairs it in place (it does not regenerate from
scratch), and a static antipattern check flags invalid constructs such as the
`>>` operator on CP expressions (which must be `mdl.if_then(...)`). All five
scenarios ship with a file-reading base template.

## What retrieval returns (concept-centric, per the paper)

`kb.rag_fca_retrieve(query)` is the paper's RAG-FCA (Eq. 15–22, Fig 1b). It is
**structural**, not example-text retrieval:

1. Extract the query's constraint attributes Q′ and close them under the
   implication base (Eq. 22).
2. Score every concept by Eq. 15 `score(Q,B)=Σ_{m∈Q∩B}w / Σ_{m∈Q∪B}w` against
   its **intent B** (a constraint set).
3. Take the max-overlap concept (A\*,B\*) and form
   `Q_aug = Q′ ∪ B* ∪ Constraints(Q′)` (certain constraints, Eq. 17–22).
4. Resolve every constraint in `Q_aug` to its **formal template** (math + CP).

It returns the formalized constraint structure (`augmented_constraints`,
`formal_templates`, `matched_concepts` with their intents, and `certain/possible`
constraints). It deliberately returns **no natural-language paragraphs and no
code blobs**. The scenario base skeleton is provided separately, only as a
docplex API/style reference for the code stage — not as an example to copy.

The earlier `kb.retrieve()` (record/row-centric) is kept only for backward
compatibility and is no longer used by the pipeline.

## The file the framework loads

    kb_store/knowledge_base.json        ← THE knowledge base (loaded at runtime)

`get_knowledge_base()` (used by `main.py` and every stage in
`pipeline/main_pipeline.py`) calls `KnowledgeBase.load_from_json()` on this file
and on **nothing else**. The source spreadsheet is *not* needed after the build.

This single file already contains the **concept lattice** — there is no separate
"lattice JSON". Its top-level structure is:

| key                          | meaning                                                                                                                                             |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `meta`                     | build info (timestamp, source, counts)                                                                                                              |
| `records`                  | the problem instances G — only the fields the pipeline needs                                                                                       |
| `formal_context`           | K = (G, M, I):`objects` (G), `attributes` (M), `relation` (the I matrix)                                                                      |
| `attribute_weights`        | w(m) used by the similarity score (Eq. 15)                                                                                                          |
| `attribute_formalizations` | **formal representation of every attribute**: name → {desc, math, cp} — the constraint in the paper's math notation + a docplex.cp template |
| `concept_lattice`          | **the concept lattice**: `concepts` (A,B), `implications` (Duquenne-Guigues base), `hasse_edges`                                        |

The lattice intents are sets of attribute *names* (the FCA structure). Each name
is only a label; `attribute_formalizations` is what turns it into a FORMAL
constraint (math + CP code). At retrieval time the matched/augmented constraint
set is resolved to these templates (`kb.get_constraint_templates()`), which is
what lets the second stage (formal model → solver code) emit correct CP
expressions instead of working from bare words.

So: the framework uses the **concept-lattice knowledge base** (`knowledge_base.json`,
whose `concept_lattice` section is the lattice). Retrieval (similarity search,
implicit-constraint discovery, rough-set certain/possible constraints) all run
against this lattice.

## Files that are NOT loaded by the framework

Everything under `kb_store/review/` is produced for **human inspection** of
the FCA construction. The pipeline does not *load* these files, but two of them
mirror structures the runtime KB builds live:

    kb_store/review/kb_formal_context.csv     # G × M incidence table (open in Excel)
    kb_store/review/kb_formal_concepts.json    # every concept (A,B)
    kb_store/review/kb_implication_base.json   # the Duquenne-Guigues implications
    kb_store/review/kb_hasse_edges.json        # lattice cover relations
    kb_store/review/kb_domain_lattices.json    # ★ per-scenario FORMAL knowledge bases
    kb_store/review/kb_summary.txt             # readable summary

### kb_domain_lattices.json — the per-scenario knowledge bases

This is the view that matches the paper's domain knowledge bases (Figs 2b–6b),
one per scenario, but **fuller than the simplified figures**. For every concept
and every constraint it gives the **formal representation**, not just a word:

    "DNA Sequence Design": {
      "objective":  {"name":"feasibility", "desc":"...", "math":"...", "cp":"..."},
      "base_constraints": [
        {"name":"gc_content",
         "desc":"GC content exactly 50% (G+C count == L/2).",
         "math":"count(W_i, G) + count(W_i, C) == L/2",
         "cp":"mdl.add(mdl.count(W[i],2) + mdl.count(W[i],3) == L//2)"}, ...],
      "additional_constraints": [ ... each with desc/math/cp + dataset id ... ],
      "concepts":     [ ... each concept's intent split into objective/base/additional, all formalized ... ],
      "implications": [ ... domain-internal Duquenne-Guigues implications ... ]
    }

The **same structure is available at runtime** without reading the file:

    kb.domain_knowledge_base("DNA Sequence Design")   # → the formalized per-domain KB
    rag = kb.rag_fca_retrieve(query)                  # rag["domain_knowledge_base"] = matched scenario's KB

So the per-domain knowledge base is a first-class runtime object; the JSON under
`review/` is its on-disk, human-readable copy.

## How attributes M are defined

`rag_fca/constraints.py` defines the **named constraint types** (paper:
M = constraint types + objective functions + solution properties) per scenario.
The paper's domain-KB figures are simplified ("etc."); M here is the fuller set.
Incidence I is taken from the dataset's `Constraint` label when present (exact),
and from NL signatures otherwise (so new plain-NL problems can still be added).

## Workflow

    python build_kb.py            # build once → writes kb_store/knowledge_base.json
    python build_kb.py --force    # rebuild (required after changing the code/catalog)
    python main.py ...            # loads kb_store/knowledge_base.json; table not needed

## Desktop GUI

A Tkinter desktop front-end is included (no extra dependencies — Tkinter ships
with Python on Windows). From the project root:

    python gui_app.py

The window provides:

- a problem-description box (with sample problems in a dropdown);
- tabs for the four stages — Structured, RAG-FCA (concepts + formal
  templates), Math model, and CP code;
- **Generate** (run the full pipeline), **Solve** (run the CP solver on the
  matching data file), **Check code** (re-validate), **Add constraint**, and
  **Save to KB** (you decide whether to persist the result);
- a Console tab that streams the pipeline logs.

All LLM/solver work runs on a background thread, so the window stays responsive.
The GUI uses the same API key / settings as the command-line tool.
