from __future__ import annotations
from typing import Dict, List

# attribute name -> {"desc","math","cp"}
FORMAL: Dict[str, Dict[str, str]] = {

  # ── problem-type anchors & objectives (solution properties) ───────────────
  "aircraft_skin_processing": {"desc":"Flexible job-shop scheduling of aircraft-skin operations on multiple robots/machines.",
     "math":"problem class: FJSP; jobs J, operations O_{j,s}, machines M","cp":"# interval vars O[j,s]; machine assignment M_s(O[j,s])"},
  "battery_pack_design": {"desc":"Reconfigurable photovoltaic battery pack: choose active modules per period.",
     "math":"problem class: reconfiguration; modules (i,j), periods k","cp":"# s[i,j,k] = 1 if module (i,j) active in period k"},
  "charging_station_location": {"desc":"EV charging-station facility location and demand allocation.",
     "math":"problem class: facility location; sites F, demand areas C","cp":"# open[f] in {0,1}; assign cust[c] -> f"},
  "dna_sequence_design": {"desc":"DNA nanostructure sequence design under biophysical constraints.",
     "math":"problem class: sequence design; sequences W_i, length L, bases {A,T,G,C}=={0,1,2,3}","cp":"# W[i][p] in {0,1,2,3} (A,T,G,C)"},
  "vrp": {"desc":"Unmanned-vehicle routing from a depot to customers.",
     "math":"problem class: VRP; vehicles V, locations N, customers C, depot D","cp":"# arc[v,i,j] in {0,1}; order u[v,i]"},
  "minimize_makespan":      {"desc":"Minimize the maximum completion time.","math":"min  max_{j,s} endOf(O_{j,s})","cp":"mdl.add(mdl.minimize(mdl.max([mdl.end_of(O[j,s]) for j,s in ops])))"},
  "minimize_energy_loss":   {"desc":"Minimize total net energy loss over all activated modules.","math":"min  scalProd(s_{i,j,k}, C_{i,j,k}+R_{i,j,k})","cp":"mdl.add(mdl.minimize(mdl.scal_prod(s, [C[i,j,k]+R[i,j,k] for ...])))"},
  "minimize_total_cost":    {"desc":"Minimize fixed construction cost plus users' travel cost.","math":"min  scalProd(fixedcost_f, open_f) + sum_c element(cust_c, cost_c)","cp":"mdl.add(mdl.minimize(mdl.scal_prod(fixedcost, open) + mdl.sum(...)))"},
  "minimize_total_distance":{"desc":"Minimize total travelled distance.","math":"min  scalProd(arc_{v,i,j}, dist_{i,j})","cp":"mdl.add(mdl.minimize(mdl.scal_prod(arc, dist)))"},
  "feasibility":            {"desc":"Feasibility problem — no objective; any solution satisfying all constraints.","math":"find any assignment satisfying all constraints","cp":"# no objective; mdl.solve() seeks feasibility"},

  # ── DNA constraints ───────────────────────────────────────────────────────
  "gc_content":              {"desc":"GC content exactly 50% (G+C count == L/2).","math":"count(W_i, G) + count(W_i, C) == L/2","cp":"mdl.add(mdl.count(W[i],2) + mdl.count(W[i],3) == L//2)"},
  "watson_crick_pairing":    {"desc":"Bases obey Watson–Crick complementarity A↔T, G↔C.","math":"comp(A)=T, comp(T)=A, comp(G)=C, comp(C)=G","cp":"# complement via element([1,0,3,2], base)"},
  "sequence_diversity":      {"desc":"Any two sequences differ in ≥4 positions (Hamming ≥ 4).","math":"sum_j [ W_x[j] != W_y[j] ] >= 4   (x != y)","cp":"mdl.add(mdl.sum(W[x][j]!=W[y][j] for j in range(L)) >= 4)"},
  "reverse_complementarity": {"desc":"≥4 mismatches vs the reverse complement of any sequence.","math":"sum_j [ element([3,2,1,0], W_x[j]) != W_y[L-j+1] ] >= 4","cp":"mdl.add(mdl.sum(mdl.element([3,2,1,0],W[x][j])!=W[y][L-1-j] for j in range(L)) >= 4)"},
  "terminal_gc_stability":   {"desc":"Each sequence must end in C or G (stronger terminal binding).","math":"W_i[L-1] in {G,C}","cp":"mdl.add(mdl.allowed_assignments(W[i][L-1], [2,3]))"},
  "base_composition_balance":{"desc":"Across all sequences, |count(b1)-count(b2)| ≤ 2 for any two bases.","math":"|count_all(b) - count_all(b')| <= 2  ∀ b,b'","cp":"mdl.add(mdl.abs(tot[b]-tot[b2]) <= 2)"},
  "dinucleotide_repetition_limit":{"desc":"No dinucleotide (2-mer) appears more than once across all positions.","math":"forall dinucleotide d: occurrences(d) <= 1","cp":"mdl.add(mdl.count([dimer[k] for k in K], d) <= 1)"},
  "forbidden_motif":         {"desc":"The subsequence ACTG must not occur in any sequence.","math":"ACTG ∉ W_i  ∀ i","cp":"mdl.add(mdl.sum(window_eq(W[i],p,[0,3,1,2]) for p)==0)"},

  # ── VRP constraints ───────────────────────────────────────────────────────
  "flow_conservation":     {"desc":"In-degree equals out-degree at every visited customer.","math":"sum_i arc_{v,i,j} = sum_i arc_{v,j,i}  ∀ v,j","cp":"mdl.add(mdl.sum(arc[v,i,j] for i)==mdl.sum(arc[v,j,i] for i))"},
  "depot_departure_return":{"desc":"Each vehicle leaves the depot ≤ once and returns consistently.","math":"sum_j arc_{v,D,j} <= 1 ;  sum_j arc_{v,j,0} <= 1","cp":"mdl.add(mdl.sum(arc[v,D,j] for j)<=1); mdl.add(mdl.sum(arc[v,j,0] for j)<=1)"},
  "vehicle_capacity":      {"desc":"Total demand carried by a vehicle ≤ its capacity Q.","math":"scalProd(d_j, sum_i arc_{v,i,j}) <= Q","cp":"mdl.add(mdl.scal_prod(d, [mdl.sum(arc[v,i,j] for i) for j]) <= Q)"},
  "subtour_elimination":   {"desc":"MTZ order constraint prevents disconnected subtours.","math":"arc_{v,i,j}=1  ⇒  u_{v,j} >= u_{v,i} + 1","cp":"mdl.add(mdl.if_then(arc[v,i,j]==1, u[v,j] >= u[v,i]+1))"},
  "visit_once":            {"desc":"Every customer is visited exactly once.","math":"sum_v sum_i arc_{v,i,j} = 1  ∀ customer j","cp":"mdl.add(mdl.sum(arc[v,i,j] for v,i)==1)"},
  "arc_existence":         {"desc":"Binary arc-usage decision variables define the routes.","math":"arc_{v,i,j} ∈ {0,1}","cp":"arc = {(v,i,j): mdl.binary_var() for ...}"},
  "vehicle_capacity_relaxed":{"desc":"Variant: vehicle-capacity constraint is NOT enforced.","math":"(vehicle_capacity removed)","cp":"# capacity constraint intentionally omitted"},
  "vrp_time_window":       {"desc":"Each customer has ready/due/service times; arrivals respect them.","math":"ready_j <= arrive_{v,j} <= due_j ;  arrive_{v,k} >= arrive_{v,j}+serv_j+t_{j,k}","cp":"mdl.add(arrive[v,j] >= ready[j]); mdl.add(arrive[v,j] <= due[j])"},
  "location_exclusion":    {"desc":"A given location is excluded — no vehicle may enter or leave it.","math":"sum_{v,i} arc_{v,i,e} = 0 ;  sum_{v,j} arc_{v,e,j} = 0","cp":"mdl.add(mdl.sum(arc[v,i,e] for v,i)==0)"},

  # ── Aircraft (FJSP) constraints ───────────────────────────────────────────
  "operation_precedence":          {"desc":"Operations of a job run in their predefined order.","math":"endBeforeStart(O_{j,s-1}, O_{j,s})","cp":"mdl.add(mdl.end_before_start(O[j,s-1], O[j,s]))"},
  "resource_flexibility":          {"desc":"Each operation is assigned to one of its eligible machines.","math":"alternative(O_{j,s}, [C_{j,s,k}]_k)","cp":"mdl.add(mdl.alternative(O[j,s], [C[j,s,k] for k]))"},
  "resource_capacity_no_overlap":  {"desc":"A machine processes at most one operation at a time.","math":"noOverlap({ O_{j,s} | M_s(O_{j,s}) = m })","cp":"mdl.add(mdl.no_overlap([C[j,s,m] for (j,s) on m]))"},
  "non_preemption":                {"desc":"Operations are atomic interval activities (no preemption).","math":"O_{j,s} is a non-preemptive interval of fixed length","cp":"O[j,s] = mdl.interval_var(size=p[j,s])  # atomic"},
  "job_start_time_window":         {"desc":"A job's first operation must start within a time window.","math":"20 <= startOf(O_{1,0}) <= 100","cp":"mdl.add(mdl.start_of(O[1,0]) >= 20); mdl.add(mdl.start_of(O[1,0]) <= 100)"},
  "job_temporal_bounds":           {"desc":"All operations of a job lie within [lb, ub].","math":"startOf(O_{2,*}) >= 10 ;  endOf(O_{2,*}) <= 500","cp":"mdl.add(mdl.start_of(O[2,s]) >= 10); mdl.add(mdl.end_of(O[2,s]) <= 500)"},
  "inter_job_precedence":          {"desc":"All operations of one job precede all of another.","math":"endBeforeStart(O_{2,last}, O_{3,first})","cp":"mdl.add(mdl.end_before_start(O[2,last], O[3,0]))"},
  "operation_level_precedence":    {"desc":"A specific operation must follow a specific operation of another job.","math":"endBeforeStart(O_{3,2}, O_{4,1})","cp":"mdl.add(mdl.end_before_start(O[3,1], O[4,0]))"},
  "operation_end_time_window":     {"desc":"A specific operation must finish within a time window.","math":"500 <= endOf(O_{5,1}) <= 600","cp":"mdl.add(mdl.end_of(O[5,0]) >= 500); mdl.add(mdl.end_of(O[5,0]) <= 600)"},
  "sequence_dependent_setup":      {"desc":"Setup time when a machine switches between different jobs.","math":"transition time = 20 on same machine when job changes","cp":"mdl.add(mdl.no_overlap(seq[m], transition_matrix))"},

  # ── Battery constraints ───────────────────────────────────────────────────
  "parallel_activation_consistency":{"desc":"Number of active modules per period equals p_k.","math":"sum_{i,j} s_{k,i,j} = p_k","cp":"mdl.add(mdl.sum(s[k,i,j] for i,j) == p[k])"},
  "battery_pack_voltage_range":    {"desc":"Pack voltage stays within [V_lk, V_uk] per period.","math":"V_lk * p_k <= scalProd(s_{k,i,j}, V_{k,i,j}) <= V_uk * p_k","cp":"mdl.add(V_l[k]*p[k] <= mdl.scal_prod(s_k, V_k)); mdl.add(mdl.scal_prod(s_k,V_k) <= V_u[k]*p[k])"},
  "max_module_current":            {"desc":"Per-period load current within active-module capacity.","math":"I_load_k <= p_k * I_max","cp":"mdl.add(I_load[k] <= p[k]*I_max)"},
  "load_demand_satisfaction":      {"desc":"Activated configuration meets the period's load demand.","math":"selected modules satisfy load voltage & current each period","cp":"# enforced jointly by voltage_range & max_module_current"},
  "module_usage_balancing":        {"desc":"Most-used and least-used module counts differ by ≤ 2.","math":"max_{i,j} use_{i,j} - min_{i,j} use_{i,j} <= 2","cp":"mdl.add(mdl.max(use) - mdl.min(use) <= 2)"},
  "switch_count_limit":            {"desc":"≤ 6 module switch-state changes between consecutive periods.","math":"sum_{i,j} | s_{k,i,j} - s_{k-1,i,j} | <= 6","cp":"mdl.add(mdl.sum(mdl.abs(s[k,i,j]-s[k-1,i,j]) for i,j) <= 6)"},
  "spatial_block_activation_limit":{"desc":"In every 2×2 block of positions, ≤ 2 modules active at once.","math":"sum_{(i,j) in block} s_{k,i,j} <= 2","cp":"mdl.add(mdl.sum(s[k,i,j] for (i,j) in block) <= 2)"},
  "faulty_module_isolation":       {"desc":"A faulty module stays offline in every period.","math":"s_{k,1,1} = 0  ∀ k","cp":"mdl.add(s[k,1,1] == 0)  # for all k"},

  # ── Charging-station constraints ──────────────────────────────────────────
  "site_opening":   {"desc":"Binary decision whether each candidate site is opened.","math":"open_f ∈ {0,1}","cp":"open = {f: mdl.binary_var() for f in F}"},
  "site_capacity":  {"desc":"A station's load cannot exceed its capacity.","math":"load_f <= capacity_f","cp":"mdl.add(load[f] <= capacity[f])"},
  "demand_coverage":{"desc":"Every demand area's charging demand is fully served.","math":"each area c is assigned to an open station covering its demand","cp":"mdl.add(mdl.sum(assign[c,f] for f)==1)"},
  "demand_allocation":{"desc":"Station load is the packed demand of its assigned areas.","math":"pack(load_f, cust_c, demand_c)","cp":"mdl.add(mdl.pack(load, cust, demand))"},
  "demand_area_expansion":{"desc":"Variant: number of demand areas expanded (e.g. 30→40).","math":"|C| increased (30 -> 40)","cp":"# enlarge demand-area index set C"},
  "travel_cost_limit":{"desc":"Travel cost between an area and its station ≤ a bound.","math":"cost(c, assigned_f) <= 20","cp":"mdl.add(mdl.element(cust[c], cost_c) <= 20)"},
  "budget_limit":   {"desc":"Total fixed cost of opened stations within a budget range.","math":"20000 <= scalProd(fixedcost_f, open_f) <= 25000","cp":"mdl.add(mdl.scal_prod(fixedcost, open) >= 20000); mdl.add(mdl.scal_prod(fixedcost, open) <= 25000)"},
  "operational_capacity_derating":{"desc":"Operating load ≤ 90% of design capacity.","math":"load_f <= 0.9 * capacity_f","cp":"mdl.add(load[f] <= 0.9*capacity[f])"},
  "site_diversity": {"desc":"Among most-similar-cost site pairs, restrict simultaneous selection.","math":"for nearest-10% cost pairs (f,f'): open_f + open_{f'} <= 1","cp":"mdl.add(open[f] + open[f2] <= 1)  # similar-cost pairs"},
}


def formalize(attrs: List[str]) -> List[Dict[str, str]]:
    """Expand a list of attribute names into their formal representations,
    skipping any without a known formalization."""
    out = []
    for a in attrs:
        f = FORMAL.get(a)
        if f:
            out.append({"name": a, **f})
    return out


def as_prompt_block(attrs: List[str], header: str = "Formal constraint templates") -> str:
    """Render formal templates as a prompt block for the model/code stages."""
    items = formalize(attrs)
    if not items:
        return ""
    lines = [f"{header} (from the concept-lattice knowledge base):"]
    for it in items:
        lines.append(f"- {it['name']}: {it['desc']}")
        lines.append(f"    math: {it['math']}")
        lines.append(f"    cp  : {it['cp']}")
    return "\n".join(lines)
