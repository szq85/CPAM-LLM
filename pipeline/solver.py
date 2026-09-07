from __future__ import annotations
import os, sys, re, json, time, datetime, subprocess, tempfile, shutil, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from typing import Dict, List, Optional, Tuple

from config import OUTPUT_DIR

DATA_FILE_MAP: Dict[str, List[str]] = {
    "Aircraft Skin Processing":    ["fjsp_location_newdataset_1.fjs"],
    "Battery Pack Design":         ["battery_data.txt"],
    "Charging Station Location":   ["plant_location.data"],
    "DNA Sequence Design":         ["word.txt"],
    "VRP":                         ["25c101.txt"],
}

DEFAULT_TIMEOUT = 120


def _resolve_data_path(code: str, data_dir: str, problem_type: str = "") -> str:
    """
    Replace every data path in the code with an absolute path under data_dir.

    Covers all 4 path forms seen in the dataset:
      1. os.path.dirname(os.path.abspath(__file__)) + "/data/xxx"
      2. os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/xxx")
      3. os.path.join("data", "xxx")
      4. open("data/xxx", ...)
    """
    abs_data = os.path.abspath(data_dir)

    code = re.sub(
        r'os\.path\.dirname\s*\(\s*os\.path\.abspath\s*\(\s*__file__\s*\)\s*\)'
        r'\s*\+\s*["\']/?data/([^"\']+)["\']',
        lambda m: f'r"{os.path.join(abs_data, m.group(1))}"',
        code,
    )
    code = re.sub(
        r'os\.path\.join\s*\(\s*os\.path\.dirname\s*\(\s*os\.path\.abspath\s*\(\s*__file__\s*\)\s*\)\s*'
        r',\s*["\']data/([^"\']+)["\']\s*\)',
        lambda m: f'r"{os.path.join(abs_data, m.group(1))}"',
        code,
    )
    code = re.sub(
        r'os\.path\.join\s*\(\s*["\']data["\']\s*,\s*["\']([^"\']+)["\']\s*\)',
        lambda m: f'r"{os.path.join(abs_data, m.group(1))}"',
        code,
    )
    code = re.sub(
        r'open\s*\(\s*["\']data/([^"\']+)["\']\s*,',
        lambda m: f'open(r"{os.path.join(abs_data, m.group(1))}",',
        code,
    )
    _ALL_DATA_FILES = [
        "fjsp_location_newdataset_1.fjs",
        "battery_data.txt",
        "plant_location.data",
        "word.txt",
        "25c101.txt",
    ]
    for fname in _ALL_DATA_FILES:
        code = re.sub(
            r'(["\'])(?![/\\])' + re.escape(fname) + r'\1',
            lambda m, f=fname: f'r"{os.path.join(abs_data, f)}"',
            code,
        )



    return code


def _inject_result_capture(code: str) -> str:
    """
    Inject standardized output markers after the solve() call:
      __CPAM_SOLVE_START__  ... solution output ...  __CPAM_SOLVE_END__
    so the solve-result text can be extracted precisely.
    """
    solve_marker = '\nprint("__CPAM_SOLVE_START__")\n'
    end_marker   = '\nprint("__CPAM_SOLVE_END__")\n'

    code = re.sub(
        r'(msol\s*=\s*(?:mdl|model)\.solve\s*\([^)]*\))',
        r'\1' + solve_marker,
        code,
        count=1,
    )
    code = code.rstrip() + end_marker
    return code


def _patch_time_limit(code: str, time_limit: int) -> str:
    """Override the TimeLimit argument of solve() in the code."""
    code = re.sub(
        r'((?:mdl|model)\.solve\s*\([^)]*TimeLimit\s*=\s*)\d+',
        lambda m: m.group(1) + str(time_limit),
        code,
    )
    return code


def execute_code(
    code: str,
    data_dir: str,
    problem_type: str = "",
    time_limit: int = 60,
    process_timeout: int = DEFAULT_TIMEOUT,
) -> Dict:
    """
    Execute CP code in a separate subprocess; return a result dict:
      {
        "status":       "solved" | "no_solution" | "error" | "timeout",
        "objective":    float | None,
        "solve_time":   float,       # seconds
        "stdout":       str,
        "stderr":       str,
        "solution_text": str,        # extracted solution-output text
        "error_msg":    str,
      }
    """
    t_start = time.time()

    patched = _resolve_data_path(code, data_dir, problem_type)
    patched = _patch_time_limit(patched, time_limit)
    patched = _inject_result_capture(patched)

    tmp_dir = tempfile.mkdtemp(prefix="cpam_solve_")
    script_path = os.path.join(tmp_dir, "solve_script.py")
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(patched)

        work_dir = os.path.dirname(os.path.abspath(data_dir))
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=process_timeout,
            cwd=work_dir,
        )
        stdout  = proc.stdout or ""
        stderr  = proc.stderr or ""
        elapsed = round(time.time() - t_start, 2)

        return _parse_output(stdout, stderr, elapsed)

    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - t_start, 2)
        return {
            "status":        "timeout",
            "objective":     None,
            "solve_time":    elapsed,
            "stdout":        "",
            "stderr":        f"Process killed after {process_timeout}s timeout",
            "solution_text": "",
            "error_msg":     f"Execution timed out after {process_timeout}s",
        }
    except Exception as e:
        elapsed = round(time.time() - t_start, 2)
        return {
            "status":        "error",
            "objective":     None,
            "solve_time":    elapsed,
            "stdout":        "",
            "stderr":        str(e),
            "solution_text": "",
            "error_msg":     str(e),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_output(stdout: str, stderr: str, elapsed: float) -> Dict:
    """Parse the subprocess output and extract the solve status and objective value."""
    result = {
        "status":        "error",
        "objective":     None,
        "solve_time":    elapsed,
        "stdout":        stdout,
        "stderr":        stderr,
        "solution_text": "",
        "error_msg":     "",
    }

    start_m = re.search(r'__CPAM_SOLVE_START__', stdout)
    end_m   = re.search(r'__CPAM_SOLVE_END__',   stdout)
    if start_m:
        sol_text = stdout[start_m.end(): end_m.start() if end_m else len(stdout)]
        result["solution_text"] = sol_text.strip()
    else:
        result["solution_text"] = stdout.strip()

    low = stdout.lower()
    has_traceback = "traceback (most recent call last)" in stderr.lower()
    no_sol_markers = (
        "no feasible solution",
        "no solution found",
        "model has no solution",  # Solver engine output.
        "search completed, model has no solution",
    )
    if (not has_traceback) and any(m in low for m in no_sol_markers):
        result["status"] = "no_solution"
    elif (not has_traceback) and ("solution found" in low or "objective value" in low):
        result["status"] = "solved"
    elif stderr:
        if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
            result["status"]    = "error"
            result["error_msg"] = _extract_import_error(stderr)
        elif "SyntaxError" in stderr:
            result["status"]    = "error"
            result["error_msg"] = _extract_syntax_error(stderr)
        elif "FileNotFoundError" in stderr:
            result["status"]    = "error"
            result["error_msg"] = _extract_file_error(stderr)
        elif "Error" in stderr or "Traceback" in stderr:
            result["status"]    = "error"
            result["error_msg"] = _extract_runtime_error(stderr)
        else:
            result["status"]    = "error"
            result["error_msg"] = stderr[:400]

    obj_patterns = [
        r'Objective value\s*[^:]*:\s*([\d.eE+\-]+)',
        r'Objective\s*[:\-=]\s*([\d.eE+\-]+)',
        r'(?:makespan|Makespan)\s*[:\-=]\s*([\d.eE+\-]+)',
        r'(?:total|Total)\s+(?:cost|distance|loss)\s*[:\-=]\s*([\d.eE+\-]+)',
        r'(?:minimum|Maximum|Optimal)\s+(?:value|cost)\s*[:\-=]\s*([\d.eE+\-]+)',
    ]
    for pat in obj_patterns:
        m = re.search(pat, stdout, re.IGNORECASE)
        if m:
            try:
                result["objective"] = float(m.group(1))
                break
            except ValueError:
                pass

    return result


def _extract_import_error(stderr: str) -> str:
    for line in stderr.splitlines():
        if "ModuleNotFoundError" in line or "ImportError" in line:
            if "docplex" in line or "cplex" in line.lower():
                return ("docplex/CPLEX not installed.\n"
                        "Run: pip install docplex\n"
                        "Full solving needs IBM CPLEX Optimization Studio (free for academics).\n"
                        "Download: https://www.ibm.com/academic/home")
            return line
    return stderr[:300]


def _extract_syntax_error(stderr: str) -> str:
    lines = stderr.splitlines()
    for i, line in enumerate(lines):
        if "SyntaxError" in line:
            ctx = lines[max(0, i-2):i+2]
            return "\n".join(ctx)
    return stderr[:300]


def _extract_file_error(stderr: str) -> str:
    for line in stderr.splitlines():
        if "FileNotFoundError" in line or "No such file" in line:
            return f"Data file not found: {line}\nPlease ensure the required data file exists under data/."
    return stderr[:300]


def _extract_runtime_error(stderr: str) -> str:
    lines = [l for l in stderr.splitlines() if l.strip()]
    return "\n".join(lines[-5:]) if lines else stderr[:400]


def check_data_files(problem_type: str, data_dir: str) -> Tuple[bool, List[str]]:
    """Check whether the data file required by the given problem type exists."""
    needed  = DATA_FILE_MAP.get(problem_type, [])
    missing = [f for f in needed if not os.path.exists(os.path.join(data_dir, f))]
    return (len(missing) == 0), missing


def list_available_data(data_dir: str) -> Dict[str, List[str]]:
    """List the available data files in data_dir, grouped by problem type."""
    available: Dict[str, List[str]] = {}
    for ptype, files in DATA_FILE_MAP.items():
        avail = [f for f in files if os.path.exists(os.path.join(data_dir, f))]
        if avail:
            available[ptype] = avail
    return available


def preview_path_resolution(code: str, data_dir: str) -> str:
    """Return a preview of the code after path correction (debugging only)."""
    patched = _resolve_data_path(code, data_dir)
    lines = patched.split('\n')[:30]
    return '\n'.join(lines)


def save_solve_result(
    pipeline_result: Dict,
    solve_result: Dict,
    task_id: str,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Save the solve result:
      - {task_id}_solve.json     full result (code + solve output)
      - {task_id}_solve.csv      this task's solve row
      - cpam_results.csv         appended to the master table (Solve status / Objective columns)
    Returns the saved JSON path.
    """
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    full = copy.deepcopy(pipeline_result)
    full["solve_result"]    = solve_result
    full["solve_timestamp"] = datetime.datetime.now().isoformat()

    json_path = os.path.join(output_dir, f"{task_id}_solve.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    final      = pipeline_result.get("final_output", {})
    nl_input   = pipeline_result.get("input", {}).get("natural_language", "")
    cp_code    = final.get("cp_code", "")
    constraint = final.get("constraint_label", "")
    ptype      = final.get("problem_type", "")
    formal     = final.get("cp_formal_language", "")
    val_score  = final.get("validation_score", "")

    new_row = pd.DataFrame([{
        "Number":              f"{task_id}_solve",
        "CP natural language": nl_input,
        "CP code":             cp_code,
        "Constraint":          constraint,
        "Problem types":       ptype,
        "CP formal language":  formal,
        "Validation score":    val_score,
        "Solve status":        solve_result.get("status", ""),
        "Objective value":     solve_result.get("objective", ""),
        "Solve time (s)":      solve_result.get("solve_time", ""),
        "Solution text":       (solve_result.get("solution_text", "") or
                                solve_result.get("stdout", ""))[:1000],
        "Error message":       solve_result.get("error_msg", "")[:500],
        "Timestamp":           full["solve_timestamp"],
        "Source":              "solver",
    }])

    master_csv = os.path.join(output_dir, "cpam_results.csv")
    if os.path.exists(master_csv):
        existing = pd.read_csv(master_csv, encoding="utf-8-sig")
        for col in new_row.columns:
            if col not in existing.columns:
                existing[col] = ""
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(master_csv, index=False, encoding="utf-8-sig")

    solve_csv = os.path.join(output_dir, f"{task_id}_solve.csv")
    new_row.to_csv(solve_csv, index=False, encoding="utf-8-sig")

    print(f"  Saved solve JSON : {json_path}")
    print(f"  Saved solve CSV  : {solve_csv}")
    print(f"  Master CSV       : {master_csv}")
    return json_path


# ──────────────────────────────────────────────────────────────
# solving
# ──────────────────────────────────────────────────────────────

def batch_solve(
    pipeline_result: Dict,
    data_dir: str,
    time_limit: int = 60,
    verbose: bool = True,
) -> List[Dict]:
    """Batch-solve the generated code in pipeline_result; return a list of results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    final   = pipeline_result.get("final_output", {})
    code    = final.get("cp_code", "")
    ptype   = final.get("problem_type", "")
    task_id = pipeline_result.get("task_id", "unknown")

    if not code:
        print("  ⚠  No CP code found in pipeline result.")
        return []

    needed_files = DATA_FILE_MAP.get(ptype, [])
    if not needed_files:
        needed_files = ["__inline__"]

    results = []
    for fname in needed_files:
        if fname == "__inline__":
            data_file = ""
        else:
            data_file = os.path.join(data_dir, fname)
            if not os.path.exists(data_file):
                print(f"  ⚠  Data file not found: {data_file} — skipping")
                continue

        if verbose:
            print(f"\n  Solving with data: {fname if fname != '__inline__' else '(inline)'}")

        solve_res = execute_code(
            code, data_dir=data_dir, problem_type=ptype, time_limit=time_limit,
        )
        solve_res["data_file"] = fname
        results.append(solve_res)

        if verbose:
            _print_solve_result(solve_res)

        batch_task_id = f"{task_id}_batch_{fname.replace('.', '_')}"
        save_solve_result(pipeline_result, solve_res, batch_task_id)

    return results


# ──────────────────────────────────────────────────────────────
#
# ──────────────────────────────────────────────────────────────

def _print_solve_result(solve_res: Dict) -> None:
    status = solve_res.get("status", "unknown")
    icons  = {"solved": "✓", "no_solution": "○", "error": "✗", "timeout": "⏱"}
    icon   = icons.get(status, "?")

    print(f"\n  {icon} Solve status   : {status.upper()}")
    if solve_res.get("objective") is not None:
        print(f"  Objective value : {solve_res['objective']}")
    print(f"  Solve time      : {solve_res.get('solve_time', 0):.1f}s")

    sol_text = solve_res.get("solution_text", "")
    if sol_text:
        print(f"\n  ── Solution output ──────────────────────────────────")
        lines = sol_text.splitlines()
        for ln in lines[:40]:
            print(f"  {ln}")
        if len(lines) > 40:
            print(f"  ... [{len(lines)-40} more lines — see saved file]")
        print(f"  ─────────────────────────────────────────────────────")

    if solve_res.get("error_msg"):
        print(f"\n  ⚠  Error:\n")
        for ln in solve_res["error_msg"].splitlines():
            print(f"     {ln}")


def run_solve_interactive(
    current_result: Dict,
    data_dir: str,
    time_limit: int = 60,
    verbose: bool = True,
) -> Dict:
    """
    Interactive solve entry (menu [3] Solve):
      - check data files
      - show path-correction preview
      - run the solve
      - save and return the updated result dict
    """
    W = 64
    print("\n" + "═" * W)
    print("  [Solve]  Executing CP code with CPLEX solver")
    print("═" * W)

    final   = current_result.get("final_output", {})
    code    = final.get("cp_code", "")
    ptype   = final.get("problem_type", "")
    task_id = current_result.get("task_id", "unknown")

    if not code:
        print("  ⚠  No generated code available.")
        return current_result

    all_ok, missing = check_data_files(ptype, data_dir)
    avail = list_available_data(data_dir)
    print(f"\n  Problem type : {ptype}")
    print(f"  Data dir     : {data_dir}")
    if avail:
        print(f"  Available data files:")
        for pt, files in avail.items():
            print(f"    [{pt}] {', '.join(files)}")
    if missing:
        print(f"\n  ⚠  Missing data files: {missing}")
        print(f"  Place them in: {data_dir}")
        print(f"  Attempting execution anyway (code may have inline data)...\n")
    else:
        print(f"  ✓ All required data files found\n")

    tl_input = input(f"  Solver time limit in seconds (default {time_limit}): ").strip()
    try:
        time_limit = int(tl_input) if tl_input else time_limit
    except ValueError:
        pass
    print(f"  Using TimeLimit = {time_limit}s\n")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    solve_task_id = f"{task_id}_solve_{ts}"

    print("  Running solver...")
    solve_res = execute_code(
        code, data_dir=data_dir, problem_type=ptype, time_limit=time_limit,
    )

    _print_solve_result(solve_res)

    json_path = save_solve_result(current_result, solve_res, solve_task_id)

    updated = copy.deepcopy(current_result)
    updated.setdefault("solve_results", []).append({
        "task_id":    solve_task_id,
        "timestamp":  datetime.datetime.now().isoformat(),
        "data_dir":   data_dir,
        "time_limit": time_limit,
        "status":     solve_res["status"],
        "objective":  solve_res["objective"],
        "solve_time": solve_res["solve_time"],
        "json_path":  json_path,
    })
    print(f"\n  Solve result saved → {json_path}")
    print("═" * W)
    return updated
