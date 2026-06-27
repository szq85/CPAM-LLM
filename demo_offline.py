#!/usr/bin/env python3
"""
CPAM-LLM — offline demonstration of the full generation pipeline.

This script shows the complete CPAM-LLM flow for a chosen application scenario,
end to end, WITHOUT the cloud API, the fine-tuned model endpoints, or the CPLEX
solver. It is driven entirely by the pre-built concept-lattice knowledge base
(kb_store/knowledge_base.json), so it runs anywhere after `python build_kb.py`.

All five scenarios are included in this one file; you choose which to display
when you run it.

Usage:
    python demo_offline.py                  # interactive menu (pick a scenario)
    python demo_offline.py 2                # run scenario #2 directly
    python demo_offline.py "DNA Sequence Design"
    python demo_offline.py all              # run every scenario in turn

For the chosen scenario it shows:
    1. the natural-language problem
    2. RAG-FCA retrieval against the concept lattice
       (matched concept, certain / possible constraints, formal templates)
    3. the structured problem representation
    4. the mathematical model (objective + every constraint, in math + CP)
    5. the generated docplex.cp solver code
    6. dual validation (syntax + required API patterns)
    7. a dynamic constraint update: inject a new constraint and re-validate
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))

from rag_fca.knowledge_base import KnowledgeBase
from pipeline.validator import syntax_check, static_semantic_check
from pipeline.dynamic_constraint import _insert_before_objective

# The five scenarios, in the order used throughout the paper.
SCENARIOS = [
    "Aircraft Skin Processing",
    "DNA Sequence Design",
    "Battery Pack Design",
    "VRP",
    "Charging Station Location",
]

# A worked additional-constraint example per scenario, with the exact docplex.cp
# lines the pipeline injects (each uses only variables present in that scenario's
# reference code, written at top scope). For the Aircraft case the window targets
# the FIRST operation of the job — the job's commencement — which keeps the model
# feasible (constraining every operation would over-constrain it).
DYNAMIC = {
    "Aircraft Skin Processing": {
        "nl": "Job 1 must start its first operation within the time window [20, 100].",
        "lines": [
            "# [C6] Job-1 commencement window: first operation starts within [20, 100]",
            "mdl.add(mdl.start_of(job_operations[0][0]) >= 20)",
            "mdl.add(mdl.start_of(job_operations[0][0]) <= 100)",
        ],
    },
    "DNA Sequence Design": {
        "nl": "Extend the design to a three-strand triangular nanostructure (topological complementarity over 3 strands).",
        "replace": [("n_words = 6", "n_words = 3")],
        "lines": [
            "# [C5] triangular topological complementarity (3 strands, L=8, L/2=4):",
            "#      the complement of strand i's second half binds strand (i+1)%3's first half",
            "#      element([1,0,3,2], W_i[p+L/2]) == W_{(i+1)%3}[p]",
            "for i in range(3):",
            "    for p in range(4):",
            "        model.add(model.element(complement_map, words[i][p + 4]) == words[(i + 1) % 3][p])",
        ],
    },
    "Battery Pack Design": {
        "nl": "Permanently isolate a faulty module: module at position (0,0) is deactivated in every period.",
        "lines": [
            "# [C5] faulty-module isolation: module (0,0) stays off in all periods  (s_{k,1,1} = 0)",
            "for k in range(NUM_PERIODS):",
            "    model.add(s[k, 0, 0] == 0)",
        ],
    },
    "VRP": {
        "nl": "Require full fleet utilization: every vehicle must be dispatched (leave the depot exactly once).",
        "lines": [
            "# [C6] full fleet utilization: each vehicle leaves the depot exactly once",
            "for v in range(num_vehicles):",
            "    model.add(model.sum(arc[v][depot][j] for j in locations) == 1)",
        ],
    },
    "Charging Station Location": {
        "nl": "Expand the demand set with ten new high-demand areas (30 -> 40), re-solved under the original constraints.",
        "replace": [("nbDemandAreas = 30", "nbDemandAreas = 40")],
        "lines": [
            "# [C5] demand expansion: the 10 added areas are served under the original",
            "#      station-opening, allocation, capacity and cost constraints (no new",
            "#      constraint type — the existing model is re-solved on the larger demand set)",
        ],
    },
}


# ── pretty-printing helpers ──────────────────────────────────────────────────
def banner(title, ch="=", w=66):
    print("\n" + ch * w)
    print("  " + title)
    print(ch * w)


def section(title):
    print("\n" + "-" * 66)
    print("  " + title)
    print("-" * 66)


def first_record(kb, problem_type):
    """The base record (constraint == 'base' if available) for a scenario."""
    base = [r for r in kb.records if r["problem_type"] == problem_type and r.get("cp_code")]
    for r in base:
        if str(r.get("constraint", "")).strip().lower() == "base":
            return r
    return base[0] if base else None


# ── the full pipeline flow for one scenario ──────────────────────────────────
def run_scenario(kb, problem_type):
    banner(f"CPAM-LLM Pipeline  —  {problem_type}", ch="=")
    rec = first_record(kb, problem_type)
    if rec is None:
        print("  (no base record found for this scenario)")
        return None

    nl = rec["nl_text"]
    code = rec["cp_code"]
    dkb = kb.domain_knowledge_base(problem_type)

    # ── 1. the problem ───────────────────────────────────────────────────────
    section("1. Natural-language problem")
    print("  " + nl.replace("\n", "\n  "))

    # ── 2. RAG-FCA retrieval against the concept lattice ─────────────────────
    section("2. RAG-FCA retrieval  (concept lattice)")
    rag = kb.rag_fca_retrieve(nl, top_k=3)
    matched = rag.get("matched_concepts", [])
    certain = rag.get("certain_constraints", [])
    possible = rag.get("possible_constraints", [])
    templates = rag.get("formal_templates", [])
    print(f"  matched domain        : {rag.get('matched_domain', problem_type)}")
    if matched:
        intent = matched[0].get("constraints", []) if isinstance(matched[0], dict) else matched[0]
        print(f"  best-concept intent   : {sorted(intent)}")
    print(f"  certain constraints   ({len(certain)}, injected automatically):")
    for c in certain:
        print(f"      • {c}")
    print(f"  possible constraints  ({len(possible)}, offered for confirmation):")
    for c in possible:
        print(f"      ? {c}")
    print(f"  formal templates resolved: {len(templates)}  (each constraint -> math + CP)")

    # ── 3. structured representation ─────────────────────────────────────────
    section("3. Structured problem representation")
    obj = dkb.get("objective", {})
    print(f"  problem type : {problem_type}")
    print(f"  objective    : {obj.get('name', '-')}  —  {obj.get('desc', '')}")
    bcs = dkb.get("base_constraints", [])
    acs = dkb.get("additional_constraints", [])
    print(f"  base constraints       : {len(bcs)}")
    print(f"  additional constraints : {len(acs)}  (available for dynamic updates)")

    # ── 4. mathematical model (objective + every constraint, math + CP) ──────
    section("4. Mathematical model  (objective + constraints, math + CP)")
    print(f"  OBJECTIVE  [{obj.get('name','-')}]")
    print(f"    math : {obj.get('math','')}")
    print(f"    cp   : {obj.get('cp','')}")
    print("\n  BASE CONSTRAINTS")
    for i, c in enumerate(bcs, 1):
        print(f"    C{i}. {c.get('name','')}")
        print(f"        desc: {c.get('desc','')}")
        print(f"        math: {c.get('math','')}")
        print(f"        cp  : {c.get('cp','')}")
    if acs:
        print("\n  ADDITIONAL CONSTRAINTS  (selectable extensions)")
        for c in acs:
            tag = f"#{c.get('id')}" if c.get("id") is not None else ""
            print(f"    [{tag}] {c.get('name','')}: {c.get('math','')}")

    # ── 5. generated solver code ─────────────────────────────────────────────
    section("5. Generated CP code  (docplex.cp)")
    print("    " + code.replace("\n", "\n    "))

    # ── 6. dual validation ───────────────────────────────────────────────────
    section("6. Validation  (dual check: syntax + required API patterns)")
    ok_syntax, syn_msg = syntax_check(code)
    ok_static, issues = static_semantic_check(code)
    print(f"  syntax check        : {'OK' if ok_syntax else 'FAIL — ' + syn_msg}")
    print(f"  static API patterns : {'OK' if ok_static else 'issues: ' + '; '.join(issues[:3])}")
    print(f"  reference code size : {len(code)} chars, {code.count(chr(10))+1} lines")

    # ── 7. dynamic constraint update + re-validation ─────────────────────────
    section("7. Dynamic update  (inject a new constraint, then re-validate)")
    dyn = DYNAMIC.get(problem_type, {})
    print(f"  new constraint : {dyn.get('nl','-')}")
    repls = dyn.get("replace", [])
    base_code = code
    for old, new in repls:
        if old in base_code:
            base_code = base_code.replace(old, new)
            print(f"  parameter edit : {old}  ->  {new}")
    inj = dyn.get("lines", [])
    print("  injected lines :")
    for ln in inj:
        print("    + " + ln)
    updated = _insert_before_objective(base_code, inj)
    ok_syntax2, syn_msg2 = syntax_check(updated)
    ok_static2, issues2 = static_semantic_check(updated)
    print(f"  re-check syntax      : {'OK' if ok_syntax2 else 'FAIL — ' + syn_msg2}")
    print(f"  re-check API patterns: {'OK' if ok_static2 else 'issues: ' + '; '.join(issues2[:3])}")
    print(f"  updated code size    : {len(updated)} chars, {updated.count(chr(10))+1} lines "
          f"(+{updated.count(chr(10)) - code.count(chr(10))} lines)")
    print("\n  Updated CP code  (new constraint injected before the objective):")
    print("    " + updated.replace("\n", "\n    "))

    banner(f"Complete  —  {problem_type}", ch="=")
    return ok_syntax and ok_static and ok_syntax2 and ok_static2


# ── scenario selection ───────────────────────────────────────────────────────
def choose_scenario(argv):
    """Resolve a scenario from the command line, or None for the menu / 'all'."""
    if not argv:
        return "MENU"
    arg = " ".join(argv).strip()
    if arg.lower() == "all":
        return "ALL"
    if arg.isdigit():
        i = int(arg)
        if 1 <= i <= len(SCENARIOS):
            return SCENARIOS[i - 1]
        return None
    for pt in SCENARIOS:                       # exact or case-insensitive name
        if arg.lower() == pt.lower():
            return pt
    for pt in SCENARIOS:                       # partial match
        if arg.lower() in pt.lower():
            return pt
    return None


def menu():
    print("\nChoose a scenario to display its full pipeline flow:")
    for i, pt in enumerate(SCENARIOS, 1):
        print(f"   {i}. {pt}")
    print("   a. all scenarios")
    choice = input("\n  Enter 1-5 or 'a' (Enter for 1): ").strip().lower()
    if choice in ("", "1"):
        return SCENARIOS[0]
    if choice in ("a", "all"):
        return "ALL"
    if choice.isdigit() and 1 <= int(choice) <= len(SCENARIOS):
        return SCENARIOS[int(choice) - 1]
    print("  (unrecognized — defaulting to scenario 1)")
    return SCENARIOS[0]


def main():
    banner("CPAM-LLM — Offline Pipeline Demonstration", ch="#")
    print("  Full generation flow, driven by the pre-built concept-lattice KB.")
    print("  No cloud API, no fine-tuned endpoints, no CPLEX solver required.")

    kb = KnowledgeBase()
    from config import KB_JSON_PATH
    if not os.path.exists(KB_JSON_PATH):
        print("\n  Knowledge base not found. Build it first with:  python build_kb.py")
        sys.exit(1)
    kb.load_from_json(KB_JSON_PATH)

    target = choose_scenario(sys.argv[1:])
    if target is None:
        print(f"\n  Unknown scenario. Choose from 1-{len(SCENARIOS)} or a name:")
        for i, pt in enumerate(SCENARIOS, 1):
            print(f"    {i}. {pt}")
        sys.exit(1)
    if target == "MENU":
        target = menu()

    t0 = time.time()
    if target == "ALL":
        results = []
        for pt in SCENARIOS:
            results.append((pt, run_scenario(kb, pt)))
        banner("Summary", ch="#")
        for pt, ok in results:
            print(f"   {pt:28} {'PASS' if ok else 'FAIL'}")
        n = sum(1 for _, ok in results if ok)
        print(f"\n   {n}/{len(results)} scenarios passed validation (base + after update).")
    else:
        run_scenario(kb, target)

    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()