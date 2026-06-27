#!/usr/bin/env python3
"""
CPAM-LLM Entry Point

Usage:
  python main.py                          # interactive mode
  python main.py --demo scheduling        # run a demo problem
  python main.py --demo dna
  python main.py --demo vrp
  python main.py --input "problem..."    # single problem
  python main.py --file problem.txt       # from file
  python main.py --show-kb               # knowledge base stats
  python main.py --augment 3             # run chaos-map augmentation demo
  python main.py --solve                 # generate + immediately solve
  python main.py --batch-solve           # batch solve all demo problems
  python main.py --data-dir /path/data   # specify data directory
  python main.py --export-sft            # export SFT dataset
  python main.py --export-kb             # export KB as FCA files for review
  python main.py --export-kb OUTPUT_DIR  # specify output directory
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

DEMO_PROBLEMS = {
    "scheduling": (
        "We have a flexible job shop scheduling problem in aircraft skin manufacturing. "
        "There are 3 jobs and 4 machines. Each job has a sequence of operations; each operation "
        "can be assigned to one of several candidate machines with different processing times. "
        "Each machine processes at most one operation at a time. "
        "Consecutive operations within a job have temporal constraints: "
        "'tight' means the next operation must start within 15 time units after the previous ends; "
        "'loose' means the next must start at least 30 time units after the previous ends. "
        "Job 1: Op1(M1:10,M2:15), Op2(M2:8,M3:12)[tight], Op3(M3:6,M4:9)[loose]. "
        "Job 2: Op1(M1:12,M3:10), Op2(M4:7,M2:11)[none]. "
        "Job 3: Op1(M2:9,M4:13), Op2(M1:8,M3:10)[tight], Op3(M4:11,M2:14)[none]. "
        "Minimize makespan. Write a Python program using the docplex.cp module."
    ),
    "dna": (
        "Design 6 unique DNA sequences each of length 8 bases (A=0,T=1,C=2,G=3). "
        "Constraints: (1) GC content: each sequence must have exactly 4 G/C bases. "
        "(2) Sequence diversity: Hamming distance between any two sequences >= 4. "
        "(3) Reverse complementarity: Hamming distance between any sequence and the "
        "reverse complement of any other sequence >= 4. "
        "Watson-Crick complement: A↔T, C↔G. "
        "This is a feasibility problem. Write a Python program using docplex.cp."
    ),
    "vrp": (
        "A fleet of 3 unmanned vehicles serves 10 customers from a depot (location 0). "
        "Vehicle capacity = 50. Customer demands (1-10): 10,15,8,12,9,7,14,11,6,13. "
        "Use a symmetric 11x11 distance matrix with random values (seed=42, range 10-100). "
        "Constraints: each customer visited exactly once, flow conservation, "
        "vehicle capacity not exceeded, sub-tour elimination. "
        "Minimize total distance. Write a Python program using docplex.cp."
    ),
    "battery": (
        "Design an optimal switching strategy for a 4-parallel, 3-series battery energy "
        "storage system over 3 operating periods. Each module has internal resistance 0.1Ω "
        "and nominal voltage 3.7V. Load demands per period: 20A, 15A, 25A at 11.1V. "
        "Minimize total energy loss (I²R loss minus recovery effect for inactive modules). "
        "Each period requires exactly 4 modules active in parallel (series constraint). "
        "Write a Python program using docplex.cp."
    ),
    "charging": (
        "Select optimal locations from 20 candidate sites for electric vehicle charging stations "
        "to serve 30 demand areas. Each demand area must be covered by at least one station. "
        "Station capacity: 100 charging piles each. Total demand per area: 5-25 EVs. "
        "Fixed construction cost: $50,000-$200,000 per site. "
        "Travel cost: proportional to distance (1-50 km). "
        "Minimize total cost (construction + average user travel cost) while ensuring "
        "all demand areas are covered and no station exceeds capacity. "
        "Write a Python program using docplex.cp."
    ),
}

DEMO_ADD_CONSTRAINTS = {
    "scheduling": "Job 1 must start its first operation within the time window [20, 100].",
    "dna":        "Additionally, no two adjacent bases within any single sequence may be the same.",
    "vrp":        "Customer 3 must be visited before customer 7 by the same vehicle.",
    "battery":    "Module B11 (row 0, col 0) has permanently failed and must be excluded.",
    "charging":   "Ten new high-demand areas (areas 31-40) with demand 15 EVs each are added.",
}


def _check_key():
    from config import API_KEY
    key = os.environ.get("CPAM_API_KEY", API_KEY)
    if key.startswith("sk-your"):
        print("⚠  Set CPAM_API_KEY:  export CPAM_API_KEY=sk-xxx\n"); sys.exit(1)
    import config; config.API_KEY = key


def interactive_mode(kb, data_dir: str):
    from pipeline.main_pipeline import run_interactive
    print("\n" + "="*64)
    print("  CPAM-LLM Interactive Mode")
    print("  Type optimization problem description, press Enter twice to submit.")
    print("  After generation: [1] Modify  [2] Add constraint  [3] Solve  [4] End")
    print("  Type 'quit' to exit immediately.")
    print("="*64)
    while True:
        print("\n[Input] Enter optimization problem (double-Enter to submit):")
        lines = []
        while True:
            line = input()
            if line.lower() in ("quit", "exit"):
                print("Bye."); return
            if line.lower() == "kb":
                import json
                print(json.dumps(kb.summary(), indent=2)); break
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        nl = "\n".join(lines).strip()
        if not nl:
            continue
        tag = input("Tag (optional, Enter to skip): ").strip()
        try:
            run_interactive(nl, kb=kb, verbose=True, auto_update_kb=True,
                            tag=tag, data_dir=data_dir)
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()


def run_augment_demo(n: int, kb):
    """Demonstrate chaos-map augmentation on n records."""
    from augmentation.chaos_map import augment_dataset, logistic_sequence, perturbation_params
    print(f"\n[Augmentation Demo] Logistic chaos map on {n} records")
    seq = logistic_sequence(10)
    print(f"  Chaos sequence (first 10): {[round(x,4) for x in seq]}")
    for i, x in enumerate(seq[:5]):
        a, b, g = perturbation_params(x)
        print(f"  x={x:.4f} → alpha={a:.4f} beta={b:.4f} gamma={g:.4f}")
    recs = [r for r in kb.records if r["cp_code"]][:n]
    augmented = augment_dataset(recs, n_iterations=1)
    print(f"\n  Base records: {len(recs)}")
    print(f"  Augmented:    {len(augmented)}")
    if augmented:
        s = augmented[0]
        print(f"  Sample aug record: id={s['id']}, chaos_x={s['chaos_x']}, "
              f"alpha={s['alpha']}, beta={s['beta']}, gamma={s['gamma']}")
        print(f"  NL diff (first 100): {s['nl_text'][:100]}")


def run_batch_solve_demo(kb, data_dir: str, verbose: bool = True):
    """Batch code generation + solving over all demo problems (--batch-solve)."""
    from pipeline.main_pipeline import run
    from pipeline.solver import execute_code, save_solve_result, check_data_files, list_available_data
    from config import SOLVER_TIME_LIMIT

    print("\n" + "="*64)
    print("  CPAM-LLM Batch Solve Demo")
    print(f"  Data dir : {data_dir}")
    print(f"  Available: {list_available_data(data_dir)}")
    print("="*64)

    all_results = []
    for demo_name, nl in DEMO_PROBLEMS.items():
        print(f"\n{'─'*64}")
        print(f"  Demo: {demo_name}")
        print(f"{'─'*64}")
        try:
            result = run(nl, kb=kb, verbose=False, auto_update_kb=False, tag=demo_name)
        except Exception as e:
            print(f"  ⚠  Pipeline failed: {e}")
            continue

        final  = result.get("final_output", {})
        code   = final.get("cp_code", "")
        ptype  = final.get("problem_type", "")

        ok, missing = check_data_files(ptype, data_dir)
        if missing and verbose:
            print(f"  ⚠  Missing data files: {missing}")

        solve_res = execute_code(code, data_dir=data_dir, problem_type=ptype,
                                 time_limit=SOLVER_TIME_LIMIT)
        task_id   = result.get("task_id", demo_name)
        save_solve_result(result, solve_res, task_id)

        icon = {"solved":"✓","no_solution":"○","error":"✗","timeout":"⏱"}.get(
            solve_res["status"], "?")
        obj  = f"  obj={solve_res['objective']}" if solve_res.get("objective") is not None else ""
        print(f"  {icon} [{ptype}] status={solve_res['status']}{obj} "
              f"time={solve_res['solve_time']:.1f}s")
        all_results.append({"demo": demo_name, **solve_res})

    # Summary table
    print("\n" + "="*64)
    print("  Batch Solve Summary")
    print(f"  {'Demo':<15} {'Status':<14} {'Objective':<15} {'Time(s)'}")
    print("  " + "─"*52)
    for r in all_results:
        obj = f"{r['objective']:.2f}" if r.get("objective") is not None else "—"
        print(f"  {r['demo']:<15} {r['status']:<14} {obj:<15} {r['solve_time']:.1f}")
    print("="*64)


def main():
    parser = argparse.ArgumentParser(description="CPAM-LLM")
    parser.add_argument("--input",       "-i", type=str, help="NL problem description")
    parser.add_argument("--file",        "-f", type=str, help="Read NL from file")
    parser.add_argument("--demo",        "-d", choices=list(DEMO_PROBLEMS),
                        help="Demo problem")
    parser.add_argument("--add-constraint", "-a", type=str, help="Add constraint to demo result")
    parser.add_argument("--tag",         "-t", type=str, default="", help="Output tag")
    parser.add_argument("--no-kb-update",action="store_true")
    parser.add_argument("--quiet",       "-q", action="store_true")
    parser.add_argument("--show-kb",     action="store_true")
    parser.add_argument("--augment",     type=int, metavar="N",
                        help="Run chaos-map augmentation demo on N records")
    # ── addsolvingparameter ─────────────────────────────────────────
    parser.add_argument("--data-dir",    type=str, default=None,
                        help="Path to data directory (default: ./data/)")
    parser.add_argument("--solve",       action="store_true",
                        help="After code generation, immediately run CP solver")
    parser.add_argument("--batch-solve", action="store_true",
                        help="Batch generate + solve all demo problems")
    parser.add_argument("--time-limit",  type=int, default=None,
                        help="CP solver TimeLimit in seconds (default from config)")
    # ── data ───────────────────────────────────────────
    parser.add_argument("--export-sft",  type=str, metavar="PATH",
                        help="Export SFT training dataset to JSONL file")
    parser.add_argument("--export-kb",   nargs="?", const="output/kb_export",
                        metavar="DIR",
                        help="Export KB as FCA files for review (default: output/kb_export)")
    
    args = parser.parse_args()

    _check_key()
    print("Initializing knowledge base...")
    from rag_fca.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()

    # data
    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "data")
    data_dir = os.path.abspath(data_dir)

    # cover TimeLimit e.g.
    if args.time_limit:
        import config
        config.SOLVER_TIME_LIMIT = args.time_limit
    from config import SOLVER_TIME_LIMIT, SOLVER_PROC_TIMEOUT

    if args.show_kb:
        print(json.dumps(kb.summary(), indent=2)); return

    if args.augment:
        run_augment_demo(args.augment, kb); return

    if args.export_sft:
        n = kb.export_sft_dataset(args.export_sft, n_iterations=1)
        print(f"SFT dataset: {n} entries → {args.export_sft}"); return

    if args.export_kb:
        out_dir = args.export_kb
        print(f"\n[KB Export] Building FCA concept lattice from {len(kb.records)} KB records...")
        outputs = kb.export_to_file(output_dir=out_dir)
        print(f"\n[KB Export] Done. {len(outputs)} files written to: {out_dir}")
        for key, fpath in outputs.items():
            import os as _os
            size = _os.path.getsize(fpath)
            print(f"  {key:30s} → {_os.path.basename(fpath)}  ({size:,} bytes)")
        print("\nFiles summary:")
        print("  kb_formal_context.csv     — cross-table K=(G,M,I): rows=KB records, cols=attributes, x=(g,m)∈I")
        print("  kb_formal_concepts.json   — all formal concepts C=(A,B), A'=B and B'=A (NextClosure)")
        print("  kb_implication_base.json  — canonical implication base U→V (Duquenne-Guigues)")
        print("  kb_hasse_edges.json       — Hasse diagram cover relations of L(K)")
        print("  kb_summary.txt            — human-readable text summary of above")
        return

    if args.batch_solve:
        run_batch_solve_demo(kb, data_dir, verbose=not args.quiet); return

    verbose = not args.quiet
    auto_kb = not args.no_kb_update
    from pipeline.main_pipeline import run, run_interactive, add_constraint
    from pipeline.solver import execute_code, save_solve_result

    def maybe_solve(result, tag=""):
        """After generation, optionally run the solver."""
        if not args.solve:
            return result
        from pipeline.solver import run_solve_interactive
        return run_solve_interactive(result, data_dir=data_dir,
                                     time_limit=SOLVER_TIME_LIMIT, verbose=verbose)

    if args.demo:
        nl = DEMO_PROBLEMS[args.demo]
        if args.quiet:
            result = run(nl, kb=kb, verbose=False, auto_update_kb=auto_kb,
                         tag=args.tag or args.demo)
            maybe_solve(result, args.demo)
        else:
            result = run_interactive(nl, kb=kb, verbose=True,
                                     auto_update_kb=auto_kb,
                                     tag=args.tag or args.demo,
                                     data_dir=data_dir)

    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            nl = f.read().strip()
        if args.quiet:
            result = run(nl, kb=kb, verbose=False, auto_update_kb=auto_kb, tag=args.tag)
            maybe_solve(result)
        else:
            run_interactive(nl, kb=kb, verbose=True,
                            auto_update_kb=auto_kb, tag=args.tag, data_dir=data_dir)

    elif args.input:
        if args.quiet:
            result = run(args.input, kb=kb, verbose=False,
                         auto_update_kb=auto_kb, tag=args.tag)
            maybe_solve(result)
        else:
            run_interactive(args.input, kb=kb, verbose=True,
                            auto_update_kb=auto_kb, tag=args.tag, data_dir=data_dir)

    else:
        interactive_mode(kb, data_dir)


if __name__ == "__main__":
    main()
