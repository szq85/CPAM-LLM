from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Tuple, List, Optional
from config import (MAX_FEEDBACK_ROUNDS, ENABLE_DYNAMIC_VALIDATION,
                    DYNAMIC_SOLVE_TIME_LIMIT, SOLVER_PROC_TIMEOUT,
                    GATE_ON_MISSING_GLOBAL_CONSTRAINTS)
from pipeline.validator import validate
from pipeline.cp_code_gen import run as gen_code
from pipeline.solver import execute_code, check_data_files
from pipeline.utils import safe_str as _safe_str
from rag_fca.knowledge_base import KnowledgeBase

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


_ENV_MARKERS = (
    "modulenotfounderror", "no module named", "importerror",
    "not installed",
    "executable file should be defined",  # Missing solver executable.
    "cannot execute the solver", "no cp optimizer", "cplex studio",
)


def _is_env_problem(res: Dict) -> bool:
    """True only when the failure is a genuine ENVIRONMENT issue (solver engine /
    package not installed, import error, engine not located/licensed) rather than
    a real code/runtime bug. A real docplex API misuse (e.g. AssertionError from
    no_overlap with wrong args) is NOT an environment problem and MUST be fed back
    to the fix pass. Environment problems must not gate code validity."""
    blob = f"{res.get('error_msg','')}\n{res.get('stderr','')}".lower()
    return any(m in blob for m in _ENV_MARKERS)


def _run_dynamic(code: str, structured: Dict, verbose: bool = True) -> Optional[Dict]:
    """
    Dynamic execution leg of the paper's dual verification (§Validation and
    iterative optimization module): actually run the generated code on the CP
    solver and interpret the outcome.

    Returns a classification dict, or None if dynamic execution is unavailable
    (disabled, docplex/CPLEX missing, or required data files absent) — in which
    case the loop falls back to static-only gating, unchanged.

        {
          "kind":      "solved" | "no_solution" | "runtime_error",
          "dynamic_ok": bool,          # True only when the code actually runs
          "status":    <solver status>,
          "objective": float | None,
          "error":     str,            # runtime traceback / solver error (if any)
        }
    """
    if not ENABLE_DYNAMIC_VALIDATION:
        return None

    ptype = structured.get("problem_type", "") or ""

    # Skip gracefully if the scenario's data files are not present — a missing
    # dataset is an environment issue, not a code defect.
    try:
        ok_data, missing = check_data_files(ptype, _DATA_DIR)
    except Exception:
        ok_data, missing = True, []
    if not ok_data:
        if verbose:
            print(f"  [Dynamic] skipped — data files missing for '{ptype}': {missing}")
        return None

    if verbose:
        print(f"  [Dynamic] executing generated code on CP solver "
              f"(time_limit={DYNAMIC_SOLVE_TIME_LIMIT}s)...")
    try:
        res = execute_code(
            code,
            data_dir=_DATA_DIR,
            problem_type=ptype,
            time_limit=DYNAMIC_SOLVE_TIME_LIMIT,
            process_timeout=SOLVER_PROC_TIMEOUT,
        )
    except Exception as e:
        # Importing/launching the solver subprocess itself failed → treat as env.
        if verbose:
            print(f"  [Dynamic] skipped — solver unavailable: {e}")
        return None

    status = res.get("status", "error")

    if status in ("solved", "no_solution"):
        # Code RAN. 'no_solution' is a legitimate solver verdict (e.g. an
        # over-constrained / infeasible instance), not a code crash, so it does
        # not block the loop — but it is surfaced as an advisory hint.
        if verbose:
            obj = res.get("objective")
            print(f"  [Dynamic] ✓ code runs — status={status}"
                  + (f", objective={obj}" if obj is not None else ""))
        return {"kind": status, "dynamic_ok": True, "status": status,
                "objective": res.get("objective"), "error": ""}

    # status == "error" or "timeout"
    if _is_env_problem(res):
        if verbose:
            print(f"  [Dynamic] skipped — environment problem (CPLEX/docplex not "
                  f"available): {res.get('error_msg','')[:80]}")
        return None

    err = (res.get("error_msg") or res.get("stderr") or "unknown runtime error").strip()
    if verbose:
        print(f"  [Dynamic] ✗ runtime/solve error → will feed back to fix pass:\n"
              f"           {err[:200]}")
    return {"kind": "runtime_error", "dynamic_ok": False, "status": status,
            "objective": None, "error": err[:1200]}


def _feedback_msg(
    report: Dict,
    code: str,
    possible_constraints: Optional[List[str]] = None,
    dynamic_feedback: str = "",
) -> str:
    """
    Build the feedback message for the fix-pass round.
    Includes the dynamic-execution feedback (Paper §dynamic execution → real-time
    error interpretation and correction) — either a runtime/solver error traceback
    or a no-solution / over-constraint hint — plus possible implicit constraints
    (Eq. 21).
    """
    errors      = report.get("all_errors", [])
    cov         = report.get("llm_result", {}).get("coverage_check", {})
    missing     = [_safe_str(x) for x in cov.get("constraints_missing", [])]
    suggestions = [_safe_str(x) for x in report.get("all_suggestions", [])]
    cq          = report.get("code_quality", {})

    parts = ["Previous code (to fix):", f"```python\n{code}\n```"]

    # Dynamic-execution feedback (runtime error OR no-solution hint) takes priority
    # — it is the most actionable signal for the fix pass.
    if dynamic_feedback:
        parts += ["\nDynamic execution feedback (the code was run on the CP solver):",
                  f"```\n{dynamic_feedback}\n```"]

    parts += ["\nStatic errors found:"]
    parts += [f"- {e}" for e in errors] or ["- No static errors detected"]

    if missing:
        parts += ["\nMissing constraints (must implement):"] + [f"- {m}" for m in missing]

    # Include possible implicit constraints from RAG-FCA as fix guidance
    if possible_constraints:
        parts += ["\nPossible implicit constraints to consider (from RAG-FCA analysis):"]
        parts += [f"- {pc}" for pc in possible_constraints[:6]]

    if suggestions:
        parts += ["\nSuggestions:"] + [f"- {s}" for s in suggestions]

    # Advisory notes (coverage / numeric heuristics) are HINTS ONLY and may be
    # false positives (e.g. a percentage written as 0.95 vs the literal 95, a
    # feasibility model with no objective). Tell the fix pass explicitly NOT to
    # rewrite otherwise-correct code on their account.
    advisory = [_safe_str(x) for x in report.get("advisory_notes", [])]
    if advisory:
        parts += ["\nAdvisory checks (HINTS — may be false positives; do NOT change "
                  "code that is already correct just to satisfy these):"]
        parts += [f"- {a}" for a in advisory[:6]]

    if cq:
        parts.append(f"\nCode quality metrics: keyword={cq.get('keyword_match',0):.2f} "
                     f"ast={cq.get('ast_structure',0):.2f} "
                     f"ngram={cq.get('ngram_overlap',0):.2f}")

    return "\n".join(parts)


def _candidate_rank(report: Dict) -> tuple:
    """
    Ranking key for choosing the best candidate across rounds.

    Priority (high → low):
      1. dynamic_ok      - no-crash first: actually runnable on the CP solver
                           (True) or dynamic execution unavailable (None) both
                           count as "runnable"; a runtime crash (False) ranks last.
      2. n_code_add      - constraint completeness (number of <model>.add in code).
                           Key: prevents "a fix round that drops constraints to get
                           a solved" from being treated as better -- a stripped
                           model solves a different, relaxed problem and must not
                           beat a complete model (even one that is only no_solution).
      3. overall_valid   - parses and has the required docplex API shape.
      4. solved          - actually found a solution (has objective value).
      5. static_ok / syntax_ok / score - tie-breakers in order.

    In one line: first "no crash", then "as complete as possible", then
    "solved / score".
    """
    dyn = report.get("dynamic_ok")
    dyn_rank = 2 if dyn is True else (1 if dyn is None else 0)
    n_add = int((report.get("constraint_coverage") or {}).get("n_code_add", 0) or 0)
    solved = 1 if (report.get("dynamic_result") or {}).get("status") == "solved" else 0
    return (
        dyn_rank,
        n_add,
        1 if report.get("overall_valid") else 0,
        solved,
        1 if report.get("static_ok") else 0,
        1 if report.get("syntax_ok") else 0,
        float(report.get("score", 0.0)),
    )


def run_loop(
    structured: Dict,
    math_model: Dict,
    kb: KnowledgeBase,
    verbose: bool = True,
) -> Tuple[str, Dict, List[Dict]]:
    print("\n" + "="*60)
    print(f"[Feedback Loop] Max rounds: {MAX_FEEDBACK_ROUNDS}")
    print("="*60)

    # Retrieve possible constraints for use in feedback messages (Eq. 21)
    possible_constraints: List[str] = kb.get_possible_constraints()

    history: List[Dict] = []
    code = ""
    report: Dict = {}
    feedback = ""
    best_rank: tuple = ()
    best_code  = ""
    best_report: Dict = {}
    base_complete_code = ""
    base_complete_n    = -1

    def _n_add(rep: Dict) -> int:
        return int((rep.get("constraint_coverage") or {}).get("n_code_add", 0) or 0)

    for rnd in range(1, MAX_FEEDBACK_ROUNDS + 1):
        print(f"\n  ── Round {rnd}/{MAX_FEEDBACK_ROUNDS} ──")

        is_fix = rnd > 1   # Round 1: first-pass; Round 2+: fix pass
        if is_fix:
            print("  [Feedback] Calling API to fix the reported issues (must preserve all constraints).")

        fix_base = base_complete_code if (is_fix and base_complete_code) else code

        code = gen_code(
            structured, math_model, kb,
            verbose=verbose,
            feedback=feedback,
            current_code=fix_base,
        )
        report = validate(code, math_model, structured, verbose=verbose)

        # ── Dynamic verification (Paper §dual verification, 3rd leg) ──────────
        runtime_error = ""
        no_solution_hint = ""
        dyn: Optional[Dict] = None
        if report.get("syntax_ok") and report.get("static_ok"):
            dyn = _run_dynamic(code, structured, verbose=verbose)
        report["dynamic_ok"]     = (dyn or {}).get("dynamic_ok") if dyn else None
        report["dynamic_result"] = dyn
        if dyn and dyn.get("kind") == "runtime_error":
            runtime_error = dyn.get("error", "")

        n_add        = _n_add(report)
        ran_no_crash = report.get("dynamic_ok") is not False
        regressed = bool(is_fix and base_complete_n > 0
                         and n_add < base_complete_n * 0.7)
        if regressed and verbose:
            print(f"  [regression-guard] constraints dropped {base_complete_n} -> {n_add} add() "
                  f"-> flagged regression: round rejected, re-fix on the complete version.")

        # Preserve the latest complete, non-regressed code.
        if ran_no_crash and not regressed and n_add >= base_complete_n:
            base_complete_code, base_complete_n = code, n_add

        is_no_sol = bool(dyn and dyn.get("kind") == "no_solution")
        if is_no_sol and rnd == 1 and MAX_FEEDBACK_ROUNDS > 1:
            no_solution_hint = (
                "The code RUNS but the solver reports NO SOLUTION. This is often an "
                "over-constrained model rather than a genuinely infeasible problem. "
                "Check especially:\n"
                "- Exclusion / skip constraints (e.g. 'location k must not be visited' or "
                "'module is faulty'): the excluded element must also be REMOVED from any "
                "'must be covered / visited exactly once' set — otherwise coverage and "
                "exclusion contradict each other.\n"
                "- Added time-window / precedence bounds that are tighter than the data allows.\n"
                "- Wrong constant magnitudes (e.g. capacity, current, load) that make all "
                "assignments infeasible.\n"
                "Relax or reconcile ONLY the single conflicting ADDED constraint; keep the "
                "base model and ALL other constraints intact. Do NOT drop multiple "
                "constraints. If after review the problem is truly infeasible, keep the "
                "model as is.")

        missing_global = (report.get("constraint_coverage", {})
                          .get("missing_global_ops", []) or [])
        cons_gate = ""
        if (GATE_ON_MISSING_GLOBAL_CONSTRAINTS and missing_global
                and rnd < MAX_FEEDBACK_ROUNDS):
            cons_gate = (
                "The generated code is MISSING required global constraints. The math "
                "model uses these docplex global constraints, but the corresponding "
                "API call(s) do NOT appear anywhere in the code: "
                + ", ".join(missing_global) + ". "
                "These are genuine global constraints with no arithmetic substitute — "
                "add the missing `<model>.add(<model>.<constraint>(...))` call(s) so "
                "that EVERY constraint in the math model is implemented. Keep all the "
                "existing, already-correct code (data loading, variables, other "
                "constraints, objective); only ADD what is missing.")
            if verbose:
                print(f"  [coverage-gate] code missing hard global constraint(s) {missing_global} "
                      f"-> triggering one fix round")

        regression_note = ""
        if regressed:
            regression_note = (
                f"CONSTRAINT REGRESSION DETECTED: your previous fix DELETED constraints "
                f"(from {base_complete_n} down to {n_add} `<model>.add(...)` calls). This is "
                f"NOT allowed. Start again from the code shown below — it is the most "
                f"complete version — and KEEP EVERY constraint it contains. Change ONLY the "
                f"minimal thing the error/feedback requires (e.g. fix one buggy API call or "
                f"relax the single conflicting bound). Do NOT remove, merge, simplify, or "
                f"omit any constraint. Re-output the FULL model with all constraints intact.")

        static_valid = report["overall_valid"]
        dyn_pass     = report["dynamic_ok"] is not False
        if is_no_sol and no_solution_hint:
            dyn_pass = False
        if cons_gate:
            dyn_pass = False
        fully_valid  = static_valid and dyn_pass and not regressed

        # Keep the highest-ranked candidate across feedback rounds.
        rank = _candidate_rank(report)
        if not regressed and (not best_code or rank > best_rank):
            best_rank   = rank
            best_code   = code
            best_report = report
        elif not best_code:
            best_code, best_report = code, report

        history.append({
            "round":        rnd,
            "backend":      "feedback_fix(api)" if is_fix else "code_generation",
            "code_lines":   len(code.split('\n')),
            "n_code_add":   n_add,
            "regressed":    regressed,
            "score":        report["score"],
            "syntax_ok":    report["syntax_ok"],
            "static_ok":    report["static_ok"],
            "dynamic_ok":   report["dynamic_ok"],
            "solver_status":(dyn or {}).get("status") if dyn else None,
            "objective":    (dyn or {}).get("objective") if dyn else None,
            "llm_valid":    report["llm_result"].get("valid", False),
            "code_quality": report.get("code_quality", {}),
            "errors":       report["all_errors"],
            "suggestions":  report["all_suggestions"],
        })

        if fully_valid:
            ran = report["dynamic_ok"] is True
            print(f"\n  ✓ Validation passed at round {rnd}"
                  + (" (static+dynamic)" if ran else " (static; dynamic execution unavailable)")
                  + (" (first-pass, no API fix needed)" if rnd == 1 else " (after API fix)"))
            return best_code, best_report, history

        if rnd < MAX_FEEDBACK_ROUNDS:
            extra = "\n\n".join(p for p in (regression_note, runtime_error,
                                            cons_gate, no_solution_hint) if p)
            fb_code = base_complete_code or code
            feedback = _feedback_msg(report, fb_code, possible_constraints, extra)
            why = ("constraint regression" if regressed
                   else "runtime/solver error" if runtime_error
                   else "missing global constraints" if cons_gate
                   else "no-solution (possible over-constraint)" if no_solution_hint
                   else "static issues")
            print(f"  → Preparing round {rnd+1} (API fix pass for {why})...")
        else:
            print(f"\n  ⚠ Max rounds reached — returning best result "
                  f"(score={best_report.get('score', 0.0):.3f}, "
                  f"add={_n_add(best_report)} constraints)")
            return best_code, best_report, history

    return best_code or code, best_report or report, history
