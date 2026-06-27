from __future__ import annotations
import json, os, sys, time, datetime, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, Optional, List
from config import OUTPUT_DIR, SOLVER_TIME_LIMIT, SOLVER_PROC_TIMEOUT
from rag_fca.knowledge_base import KnowledgeBase, get_knowledge_base
from pipeline.nl_structuring import run as stage1
from pipeline.math_model import run as stage2
from pipeline.feedback import run_loop
from pipeline.dynamic_constraint import adapt_constraint, _new_label, _constraint_id
from pipeline.validator import validate


def _get_structured(result: Dict) -> Dict:
    """
    Safely extract the structured problem dict from any result type
    (base run, first update, or Nth chained update).

    Search order:
      1. final_output._structured   — set by both run() and add_constraint()
      2. pipeline_stages.stage1_nl_structuring.output  — set only by run()
      3. Minimal fallback built from final_output fields
    """
    fo = result.get("final_output", {})

    # 1. Always-present after fix: _structured in final_output
    s = fo.get("_structured")
    if isinstance(s, dict) and s:
        return s

    # 2. Direct run() result
    s = (result.get("pipeline_stages", {})
               .get("stage1_nl_structuring", {})
               .get("output"))
    if isinstance(s, dict) and s:
        return s

    # 3. Fallback — reconstruct minimal dict so the pipeline never crashes
    return {
        "problem_type":      fo.get("problem_type", "Unknown"),
        "problem_summary":   fo.get("problem_summary", ""),
        "constraints":       [],
        "decision_variables":[],
        "parameters":        [],
        "implicit_constraints": [],
        "objective":         {},
    }



def _build_formal_language(structured: Dict, math_model: Dict) -> str:
    """Build the CP formal language description string (matches dataset column format)."""
    lines = ["1. Decision Variables"]
    for v in math_model.get("variables_section", []):
        sym   = v.get("symbol", "") or v.get("name", "")
        dtype = v.get("docplex_type", "")
        desc  = v.get("description", "")
        api   = v.get("docplex_call", "")
        lines.append(f"{sym} [{dtype}]: {desc}")
        if api:
            lines.append(f"  API: {api}")

    lines.append("\n2. Parameters and Input Data")
    for p in math_model.get("parameters_section", []):
        sym  = p.get("symbol", "") or p.get("name", "")
        desc = p.get("description", "")
        lines.append(f"{sym}: {desc}")

    lines.append("\n3. Constraints")
    for c in math_model.get("constraints_section", []):
        cid   = c.get("id", "")
        name  = c.get("name", "")
        mexpr = c.get("math_expression", "")
        dexpr = c.get("docplex_expression", "")
        desc  = c.get("description", "")
        lines.append(f"[{cid}] {name}: {mexpr}")
        if dexpr:
            lines.append(f"  docplex: {dexpr}")
        if desc:
            lines.append(f"  Note: {desc}")

    lines.append("\n4. Objective Function")
    obj       = math_model.get("objective_section", {})
    direction = obj.get("direction", "")
    mexpr     = obj.get("math_expression", "")
    dexpr     = obj.get("docplex_expression", "")
    desc      = obj.get("description", "")
    lines.append(f"{direction.upper()} {mexpr}")
    if dexpr:
        lines.append(f"docplex: {dexpr}")
    if desc:
        lines.append(f"Note: {desc}")

    lines.append("\n5. Data Specification")
    for p in structured.get("parameters", []):
        name = p.get("name", "")
        desc = p.get("description", "")
        src  = p.get("value_or_source", "")
        lines.append(f"{name}: {desc} ({src})")

    return "\n".join(lines)


def _constraint_label(structured: Dict) -> str:
    ids = set()
    for c in structured.get("constraints", []):
        ctype = c.get("type", "").lower()
        for kw, cid in {
            "time_window":"1","temporal":"1","window":"1",
            "capacity":"2","resource":"3","machine":"3",
            "precedence":"4","sequence":"4","global":"5",
            "all_diff":"5","gc_content":"5","dynamic":"6","fault":"6",
        }.items():
            if kw in ctype:
                ids.add(cid)
    return ",".join(sorted(ids)) if ids else "base"


def _save(result: Dict, task_id: str) -> str:
    """Save result as both JSON and a CSV row appended to the master results file."""
    import pandas as pd

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, f"{task_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ── CSV — same column schema as CP_weitiao140.xlsx ────────
    # Columns: Number | CP natural language | CP code | Constraint | Problem types | CP formal language
    final   = result.get("final_output", {})
    meta    = result.get("metadata", {})
    updates = result.get("dynamic_updates", [])

    # Build one row per "version" of the problem (base + each constraint update)
    rows = []

    # Row 1 — base result (before any dynamic updates)
    stage3 = result.get("pipeline_stages", {}).get("stage3_cp_code", {})
    base_code = stage3.get("final_code", final.get("cp_code", ""))
    base_label = final.get("constraint_label", "base")
    # If there were dynamic updates the base label should be reverted
    if updates:
        base_label = "base"

    rows.append({
        "Number":               result.get("task_id", ""),
        "CP natural language":  result["input"].get("natural_language", ""),
        "CP code":              base_code,
        "Constraint":           base_label,
        "Problem types":        final.get("problem_type", ""),
        "CP formal language":   final.get("cp_formal_language", ""),
        "Validation score":     stage3.get("validation_report", {}).get("score",
                                final.get("validation_score", "")),
        "Timestamp":            result.get("timestamp", ""),
        "Source":               meta.get("update_type", "pipeline"),
    })

    # Rows for each dynamic constraint update
    for i, upd in enumerate(updates, 1):
        rows.append({
            "Number":               f"{result.get('task_id','')}_update{i}",
            "CP natural language":  (result["input"].get("natural_language", "") +
                                     f"\n\nAdditional constraint: {upd.get('constraint_nl','')}"),
            "CP code":              upd.get("updated_code", ""),
            "Constraint":           upd.get("constraint_id", str(i)),
            "Problem types":        final.get("problem_type", ""),
            "CP formal language":   final.get("cp_formal_language", ""),
            "Validation score":     upd.get("validation", {}).get("score", ""),
            "Timestamp":            upd.get("timestamp", ""),
            "Source":               "dynamic_constraint_adaptation",
        })

    new_df = pd.DataFrame(rows, columns=[
        "Number", "CP natural language", "CP code",
        "Constraint", "Problem types", "CP formal language",
        "Validation score", "Timestamp", "Source",
    ])

    # Append to master CSV (one file accumulates all runs)
    master_csv = os.path.join(OUTPUT_DIR, "cpam_results.csv")
    if os.path.exists(master_csv):
        existing = pd.read_csv(master_csv, encoding="utf-8-sig")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(master_csv, index=False, encoding="utf-8-sig")

    # Also save a per-task CSV
    task_csv = os.path.join(OUTPUT_DIR, f"{task_id}.csv")
    new_df.to_csv(task_csv, index=False, encoding="utf-8-sig")

    print(f"  Saved JSON : {json_path}")
    print(f"  Saved CSV  : {task_csv}")
    print(f"  Master CSV : {master_csv}  ({len(combined)} total rows)")
    return json_path


def _banner(text: str) -> None:
    w = 64
    print("\n" + "═"*w + f"\n  {text}\n" + "═"*w)


def run(
    nl_input: str,
    kb: Optional[KnowledgeBase] = None,
    verbose: bool = True,
    auto_update_kb: bool = True,
    tag: str = "",
) -> Dict:
    """Run the full CPAM-LLM pipeline from NL to CP code. Returns full result dict."""
    if kb is None:
        kb = get_knowledge_base()

    t0 = time.time()
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    task_id = f"task_{ts}_{tag}" if tag else f"task_{ts}"

    _banner(f"CPAM-LLM Pipeline  [{task_id}]")
    print(f"Input (first 300 chars):\n{nl_input[:300]}{'...' if len(nl_input)>300 else ''}")
    kb_info = kb.summary()
    print(f"\nKnowledge base: {kb_info['total_records']} records | "
          f"{kb_info['total_concepts']} concepts | "
          f"{kb_info['total_attributes']} attributes")

    result: Dict = {
        "task_id":   task_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "input":     {"natural_language": nl_input},
        "knowledge_base_summary": kb_info,
        "pipeline_stages": {},
        "final_output": {},
        "dynamic_updates": [],
        "metadata": {
            "total_time_seconds": 0,
            "kb_updated": False,
            "validation_passed": False,
            "iteration_rounds": 0,
        },
    }

    try:
        structured = stage1(nl_input, kb, verbose=verbose)
        result["pipeline_stages"]["stage1_nl_structuring"] = {"status":"success","output":structured}
    except Exception as e:
        result["pipeline_stages"]["stage1_nl_structuring"] = {"status":"error","error":str(e)}
        result["metadata"]["total_time_seconds"] = round(time.time()-t0, 2)
        _save(result, task_id); raise

    try:
        math_model = stage2(structured, kb, verbose=verbose)
        result["pipeline_stages"]["stage2_math_model"] = {"status":"success","output":math_model}
    except Exception as e:
        result["pipeline_stages"]["stage2_math_model"] = {"status":"error","error":str(e)}
        result["metadata"]["total_time_seconds"] = round(time.time()-t0, 2)
        _save(result, task_id); raise

    try:
        final_code, val_report, history = run_loop(structured, math_model, kb, verbose=verbose)
        result["pipeline_stages"]["stage3_cp_code"] = {
            "status": "success",
            "final_code": final_code,
            "validation_report": val_report,
            "iteration_history": history,
        }
        result["metadata"]["validation_passed"] = val_report["overall_valid"]
        result["metadata"]["iteration_rounds"]  = len(history)
    except Exception as e:
        result["pipeline_stages"]["stage3_cp_code"] = {"status":"error","error":str(e)}
        result["metadata"]["total_time_seconds"] = round(time.time()-t0, 2)
        _save(result, task_id); raise

    # Post-processing: build formal language description and assemble final_output.
    # Wrapped in try/except so that even a failure here returns a usable result
    # (the code was already generated successfully above).
    try:
        formal_lang      = _build_formal_language(structured, math_model)
        # Prefer the constraint_label that Stage 1 already computed deterministically
        # (the align path derives it from the CATALOG regex signatures, so it matches
        # the dataset's real optional-id numbering, e.g. DNA '1,4'). Only fall back to
        # the keyword-based guesser when Stage 1 did not provide one. The old guesser
        # used a generic keyword->id map that does NOT match per-type optional ids and
        # would mislabel results (e.g. it tagged DNA as '5' from a 'gc_content' base
        # constraint, even though DNA has no id 5 and gc_content is a base constraint).
        stage1_label = structured.get("constraint_label")
        if stage1_label:
            constraint_label = stage1_label
        else:
            constraint_label = _constraint_label(structured)
    except Exception as e:
        print(f"\n  ⚠  Post-processing warning: {e} — using fallback formal language.")
        formal_lang      = f"(formal language generation failed: {e})"
        constraint_label = "base"

    result["final_output"] = {
        "problem_type":       structured.get("problem_type", "Unknown"),
        "problem_summary":    structured.get("problem_summary", ""),
        "cp_natural_language":nl_input,
        "cp_formal_language": formal_lang,
        "math_model":         math_model,
        "cp_code":            final_code,
        "validation_score":   val_report.get("score", 0.0),
        "constraint_label":   constraint_label,
        "_structured":        structured,
    }
    result["metadata"]["total_time_seconds"] = round(time.time()-t0, 2)

    if auto_update_kb:
        rec = {
            "problem_type":   structured.get("problem_type", "Unknown"),
            "nl_text":        nl_input,
            "cp_code":        final_code,
            "constraint":     constraint_label,
            "constraint_desc":formal_lang,
            "math_model":     math_model,
        }
        kb.try_auto_update(rec, val_report, nl_input)

    path = _save(result, task_id)
    _print_summary(result, path)
    return result


def add_constraint(
    new_constraint_nl: str,
    existing_result: Dict,
    kb: Optional[KnowledgeBase] = None,
    verbose: bool = True,
    auto_update_kb: bool = True,
) -> Dict:
    """
    Dynamic constraint adaptation (paper §dynamic constraint adaptation).
    Takes the result of a previous run() call and injects a new constraint.
    Returns a new result dict for the updated problem.
    """
    if kb is None:
        kb = get_knowledge_base()

    t0 = time.time()
    orig_id   = existing_result.get("task_id", "unknown")
    ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    update_id = f"{orig_id}_update_{ts}"

    _banner(f"Dynamic Constraint Adaptation  [{update_id}]")
    print(f"  Parent task: {orig_id}")
    print(f"  New constraint: {new_constraint_nl[:120]}")

    existing_code       = existing_result["final_output"]["cp_code"]
    existing_structured = _get_structured(existing_result)
    existing_model      = existing_result["final_output"]["math_model"]
    existing_label      = existing_result["final_output"]["constraint_label"]

    updated_code, updated_struct, updated_model = adapt_constraint(
        new_constraint_nl, existing_code,
        existing_structured, existing_model, kb, verbose=verbose,
    )

    print("\n[Re-validation after constraint injection]")
    val_report = validate(updated_code, updated_model, updated_struct, verbose=verbose)

    if not val_report["overall_valid"] and verbose:
        print("  ⚠ Validation flagged issues — code injected but may need manual review.")

    c_id      = _constraint_id(new_constraint_nl)
    new_label = _new_label(existing_label, c_id)
    try:
        formal_lang = _build_formal_language(updated_struct, updated_model)
    except Exception as e:
        print(f"  ⚠  Formal language build warning: {e}")
        formal_lang = f"(formal language unavailable: {e})"

    result: Dict = {
        "task_id":         update_id,
        "timestamp":       datetime.datetime.now().isoformat(),
        "parent_task_id":  orig_id,
        "new_constraint_nl": new_constraint_nl,
        "input": {
            "natural_language": existing_result["input"]["natural_language"],
            "added_constraint": new_constraint_nl,
        },
        "pipeline_stages": {
            "dynamic_adaptation": {
                "status": "success",
                "updated_code": updated_code,
                "new_constraint_formal": updated_model["constraints_section"][-1],
                "validation_report": val_report,
            }
        },
        "dynamic_updates": existing_result.get("dynamic_updates", []) + [{
            "constraint_nl": new_constraint_nl,
            "constraint_id": c_id,
            "timestamp":     datetime.datetime.now().isoformat(),
            "updated_code":  updated_code,
            "validation":    val_report,
        }],
        "final_output": {
            "problem_type":        updated_struct.get("problem_type",""),
            "problem_summary":     updated_struct.get("problem_summary","") + f" + [{new_constraint_nl[:60]}]",
            "cp_natural_language": (existing_result["input"]["natural_language"] +
                                    f"\n\nAdditional constraint: {new_constraint_nl}"),
            "cp_formal_language":  formal_lang,
            "math_model":          updated_model,
            "cp_code":             updated_code,
            "validation_score":    val_report.get("score", 0.0),
            "constraint_label":    new_label,
            "_structured":         updated_struct,
        },
        "metadata": {
            "total_time_seconds": round(time.time()-t0, 2),
            "kb_updated": False,
            "validation_passed": val_report["overall_valid"],
            "iteration_rounds": 1,
            "update_type": "dynamic_constraint_adaptation",
        },
    }

    if auto_update_kb:
        rec = {
            "problem_type":   updated_struct.get("problem_type", "Unknown"),
            "nl_text":        result["input"]["natural_language"] + f"\n+[{new_constraint_nl}]",
            "cp_code":        updated_code,
            "constraint":     new_label,
            "constraint_desc":formal_lang,
        }
        added, reason = kb.try_auto_update(rec, val_report, new_constraint_nl)
        result["metadata"]["kb_updated"] = added

    path = _save(result, update_id)
    _print_summary(result, path)
    return result


def _print_math_model(math_model: Dict) -> None:
    """Display the mathematical model using CPLEX mathematical notation."""
    W = 64
    print("\n" + "─"*W)
    print(f"  Mathematical Model: {math_model.get('model_title','')}")
    print(f"  Type: {math_model.get('model_type','')}  |  {math_model.get('complexity_note','')}")
    print("─"*W)

    params = math_model.get("parameters_section", [])
    if params:
        print("\n  Parameters:")
        for p in params:
            print(f"    {p.get('symbol',''):20s}  {p.get('description','')}")

    variables = math_model.get("variables_section", [])
    if variables:
        print("\n  Decision Variables:")
        for v in variables:
            print(f"    {v.get('symbol',''):20s}  [{v.get('docplex_type','')}]"
                  f"  {v.get('description','')}")
            dom = v.get("domain_or_size", "")
            if dom:
                print(f"    {'':20s}  domain: {dom}")

    constraints = math_model.get("constraints_section", [])
    if constraints:
        print("\n  Constraints:")
        for c in constraints:
            cid   = c.get("id", "")
            name  = c.get("name", "")
            mexpr = c.get("math_expression", "")
            dexpr = c.get("docplex_expression", "")
            desc  = c.get("description", "")
            print(f"\n    [{cid}] {name}")
            print(f"         Math  :  {mexpr}")
            print(f"         Code  :  {dexpr}")
            if desc:
                print(f"         Note  :  {desc}")

    obj = math_model.get("objective_section", {})
    if obj:
        print("\n  Objective Function:")
        direction = obj.get("direction", "").upper()
        mexpr     = obj.get("math_expression", "")
        dexpr     = obj.get("docplex_expression", "")
        desc      = obj.get("description", "")
        print(f"    {direction}  {mexpr}")
        print(f"    Code  :  {dexpr}")
        if desc:
            print(f"    Note  :  {desc}")
    print("\n" + "─"*W)


def _print_summary(result: Dict, path: str) -> None:
    meta  = result["metadata"]
    final = result["final_output"]
    _banner("CPAM-LLM Complete")
    print(f"  Task ID         : {result['task_id']}")
    if "parent_task_id" in result:
        print(f"  Parent Task     : {result['parent_task_id']}")
        print(f"  Update type     : {meta.get('update_type','dynamic_constraint_adaptation')}")
    print(f"  Problem type    : {final.get('problem_type')}")
    print(f"  Summary         : {final.get('problem_summary','')[:80]}")
    print(f"  Constraint label: {final.get('constraint_label')}")
    print(f"  Validation score: {final.get('validation_score',0):.2f}/1.00")
    print(f"  Valid           : {'✓' if meta['validation_passed'] else '✗'}")
    print(f"  Iterations      : {meta['iteration_rounds']}")
    print(f"  KB updated      : {'✓' if meta['kb_updated'] else '—'}")
    print(f"  Total time      : {meta['total_time_seconds']:.1f}s")
    print(f"  Output JSON     : {path}")
    csv_path = path.replace(".json", ".csv")
    if os.path.exists(csv_path):
        print(f"  Output CSV      : {csv_path}")
    print("═"*64)

    # ── Mathematical Model ────────────────────────────────────
    math_model = final.get("math_model", {})
    if math_model:
        print("\n[Mathematical Model  (CPLEX notation)]")
        _print_math_model(math_model)
    else:
        print("\n[Mathematical Model: not available in this result]")

    # ── Generated CP Code ─────────────────────────────────────
    print("\n[Generated CP Code  (docplex.cp)]")
    print("─"*64)
    code = final.get("cp_code", "")
    print(code[:3000])
    if len(code) > 3000:
        print(f"... [{len(code)-3000} more chars — see output file]")
    print("─"*64)


def _show_menu() -> str:
    """Display the post-generation action menu and return user choice."""
    print("\n" + "┌" + "─"*54 + "┐")
    print("│  What would you like to do next?  (Enter 'q' to quit) │")
    print("│                                                        │")
    print("│  [1]  Modify code                                      │")
    print("│  [2]  Add new requirement / constraint                 │")
    print("│  [3]  Solve / Execute (run CP solver)                  │")
    print("│  [5]  Save current result to knowledge base            │")
    print("│  [4]  End conversation  (or press 'q')                 │")
    print("└" + "─"*54 + "┘")
    while True:
        choice = input("  Enter choice (1/2/3/5/4 or q): ").strip().lower()
        if choice == "q":
            return "4"
        if choice in ("1", "2", "3", "4", "5"):
            return choice
        print("  Please enter 1, 2, 3, 5, 4, or q.")


def _save_to_kb(current_result: Dict, kb: KnowledgeBase) -> None:
    """
    Option [5]: explicitly save (or not) the current result into the knowledge
    base. The user decides — this is how previously-unseen constraints get
    persisted. Adding to the KB inserts the new problem into the formal context
    and updates the concept lattice (incremental, Eq. 23-24).
    """
    if not current_result:
        print("  Nothing to save yet.")
        return
    fo     = current_result.get("final_output", {})
    struct = fo.get("_structured", {})
    ptype  = fo.get("problem_type", struct.get("problem_type", "Unknown"))
    score  = fo.get("validation_score", 0.0)
    label  = fo.get("constraint_label", "")
    nl     = fo.get("cp_natural_language", "") or \
             current_result.get("input", {}).get("natural_language", "")

    print("\n  ── Save to knowledge base ──")
    print(f"    Problem type : {ptype}")
    print(f"    Constraints  : {label or '(base)'}")
    print(f"    Valid score  : {score:.2f}")
    ans = input("  Save this result to the knowledge base? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("  Not saved. Knowledge base unchanged.")
        return

    rec = {
        "problem_type":    ptype,
        "nl_text":         nl,
        "cp_code":         fo.get("cp_code", ""),
        "constraint":      label,
        "constraint_desc": fo.get("cp_formal_language", ""),
        "math_model":      fo.get("math_model", {}),
    }
    try:
        added, reason = kb.add_record_interactive(rec)
        if added:
            print(f"  ✓ Saved to knowledge base. {reason}")
            print(f"    KB now: {len(kb.records)} records, "
                  f"{len(kb.lattice.concepts)} concepts, "
                  f"{len(kb.context.attributes)} attributes.")
        else:
            print(f"  Not saved: {reason}")
    except Exception as e:
        print(f"  ⚠  Could not save to KB: {str(e)[:120]}")


def _modify_code(current_result: Dict, kb: KnowledgeBase,
                 verbose: bool = True) -> Dict:
    """
    Handle option [1]: user describes targeted code modifications.
    Uses the LLM to apply changes to the existing code, then re-validates.
    """
    from llm.client import chat_for_stage, system
    import re as _re

    print("\n  Describe the code modification you want:")
    print("  (e.g. 'Change the time limit to 60', 'Use model instead of mdl', etc.)")
    mod_desc = input("  > ").strip()
    if not mod_desc:
        print("  No modification entered — returning unchanged.")
        return current_result

    final      = current_result["final_output"]
    code       = final.get("cp_code", "")
    math_model = final.get("math_model", {})

    prompt = f"""Modify the following docplex.cp Python code as described.

Requested modification:
{mod_desc}

Current code:
{code}

Rules:
- Apply only the requested change(s).
- Keep all existing logic intact unless explicitly asked to change it.
- No comments. No markdown. Return only the complete modified Python code."""

    raw = chat_for_stage("feedback_fix", [{"role": "system", "content": system("code_modifier")},
                {"role": "user",   "content": prompt}], temperature=0.1)

    # strip fences
    m = _re.search(r'```(?:python)?\n?(.*?)```', raw, _re.DOTALL)
    updated_code = m.group(1).strip() if m else raw.strip()

    print("\n[Re-validation after code modification]")
    from pipeline.validator import validate
    structured = final.get("_structured",
                 current_result.get("pipeline_stages", {})
                               .get("stage1_nl_structuring", {})
                               .get("output", {}))
    val_report = validate(updated_code, math_model, structured, verbose=verbose)

    import copy, time, datetime
    updated = copy.deepcopy(current_result)
    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_id  = f"{current_result.get('task_id','task')}_mod_{ts}"
    updated["task_id"]   = new_id
    updated["timestamp"] = datetime.datetime.now().isoformat()
    updated["final_output"]["cp_code"]          = updated_code
    updated["final_output"]["validation_score"] = val_report.get("score", 0.0)
    updated["metadata"]["validation_passed"]    = val_report["overall_valid"]
    updated["metadata"]["update_type"]          = "code_modification"
    updated.setdefault("modifications", []).append({
        "description": mod_desc,
        "timestamp":   datetime.datetime.now().isoformat(),
        "score":       val_report.get("score", 0.0),
    })

    path = _save(updated, new_id)

    print("\n[Modified Code]")
    print("─"*64)
    print(updated_code[:2000])
    if len(updated_code) > 2000:
        print(f"... [{len(updated_code)-2000} more chars]")
    print("─"*64)
    print(f"  Validation score: {val_report.get('score',0):.2f}/1.00  "
          f"{'✓' if val_report['overall_valid'] else '✗'}")
    print(f"  Saved: {path}")
    return updated



def _get_solver_time_limit() -> int:
    """Return solver TimeLimit from config (can be overridden at runtime)."""
    return SOLVER_TIME_LIMIT


def run_interactive(
    nl_input: str,
    kb: Optional[KnowledgeBase] = None,
    verbose: bool = True,
    auto_update_kb: bool = True,
    tag: str = "",
    data_dir: Optional[str] = None,
) -> Optional[Dict]:
    """
    Run the pipeline and enter an interactive menu loop:
      [1] Modify code
      [2] Add new requirement / constraint
      [3] Solve / Execute (run CP solver with real data)
      [4] End conversation

    The menu appears even when the pipeline partially fails — in that case
    only options [2] (retry with added detail) and [4] are available.

    Args:
        data_dir: Path to data directory containing dataset files.
                  Defaults to the project data/ folder.
    """
    if kb is None:
        kb = get_knowledge_base()

    # In interactive mode, saving to the KB is an explicit user choice (menu
    # option [5]) — never automatic. This lets the user decide whether to keep
    # a previously-unseen constraint or discard it.
    auto_update_kb = False

    # Resolve data directory (default: project data/ folder)
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    current: Optional[Dict] = None
    try:
        current = run(nl_input, kb=kb, verbose=verbose,
                      auto_update_kb=auto_update_kb, tag=tag)
    except Exception as e:
        print(f"\n  ⚠  Pipeline error: {e}")
        print("  The menu will still appear so you can retry or exit.")

    while True:
        if current is None:
            # Pipeline failed — show limited menu
            print("\n" + "┌" + "─"*52 + "┐")
            print("│  Pipeline did not complete successfully.          │")
            print("│                                                   │")
            print("│  [2]  Retry with additional requirement           │")
            print("│  [4]  End conversation                            │")
            print("└" + "─"*52 + "┘")
            choice = ""
            while choice not in ("2", "4", "q"):
                choice = input("  Enter choice (2/4 or q): ").strip().lower()
            if choice == "q":
                choice = "4"
        else:
            choice = _show_menu()

        if choice == "1":
            current = _modify_code(current, kb, verbose=verbose)

        elif choice == "2":
            if current is None:
                # Retry from scratch with a supplementary note
                print("\n  Describe what to add / clarify and press Enter:")
                nc = input("  > ").strip()
                new_nl = nl_input + (f"\n\nAdditional detail: {nc}" if nc else "")
                try:
                    current = run(new_nl, kb=kb, verbose=verbose,
                                  auto_update_kb=auto_update_kb, tag=tag)
                except Exception as e:
                    print(f"  ⚠  Still failed: {e}")
            else:
                print("\n  Describe the new requirement or constraint to add:")
                nc = input("  > ").strip()
                if nc:
                    current = add_constraint(nc, current, kb=kb,
                                             verbose=verbose,
                                             auto_update_kb=auto_update_kb)
                else:
                    print("  No input — returning to menu.")

        elif choice == "3":
            # [3] Solve / Execute
            from pipeline.solver import run_solve_interactive
            current = run_solve_interactive(
                current, data_dir=data_dir,
                time_limit=_get_solver_time_limit(),
                verbose=verbose,
            )

        elif choice == "5":
            _save_to_kb(current, kb)

        elif choice == "4":
            print("\n  Conversation ended.")
            if current:
                print(f"  Final task ID: {current.get('task_id')}")
                final_csv = os.path.join(OUTPUT_DIR, "cpam_results.csv")
                if os.path.exists(final_csv):
                    print(f"  Master CSV   : {final_csv}")
                solve_results = current.get("solve_results", [])
                if solve_results:
                    print(f"  Solve runs   : {len(solve_results)}")
                    for sr in solve_results:
                        icon = "✓" if sr["status"]=="solved" else ("○" if sr["status"]=="no_solution" else "✗")
                        obj  = f"  obj={sr['objective']}" if sr.get("objective") is not None else ""
                        print(f"    {icon} {sr['status']}{obj}  ({sr['solve_time']:.1f}s)  → {sr['json_path']}")
            break

    return current
