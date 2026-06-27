from __future__ import annotations
import re
from typing import Dict, List, Set, Optional

# scenario -> {anchor, objective, base[list], optional[list of dict]}
#   optional dict: {id:int|None, name:str, regex:str, relax:str|None}
CATALOG: Dict[str, dict] = {
 "Aircraft Skin Processing": {
   "anchor": "aircraft_skin_processing",
   "objective": "minimize_makespan",
   "base": ["resource_flexibility", "operation_precedence",
            "resource_capacity_no_overlap", "non_preemption"],
   "optional": [
     {"id":1,"name":"job_start_time_window",
      "regex":r"restricted time window|must commence within|Job ?1 must start|start between time \d+ and"},
     {"id":2,"name":"job_temporal_bounds",
      "regex":r"strict temporal boundaries|must start after time|Job ?2[^.]*(?:start after|end before)|operations of Job ?2[^.]*time"},
     {"id":3,"name":"inter_job_precedence",
      "regex":r"inter-?job precedence|Job ?3[^.]*(?:after|only after)[^.]*Job ?2|all operations of Job ?3[^.]*after"},
     {"id":4,"name":"operation_level_precedence",
      "regex":r"fine-grained precedence|first operation of Job ?4[^.]*after[^.]*(?:second )?operation of Job ?3"},
     {"id":5,"name":"operation_end_time_window",
      "regex":r"operation-level temporal constraint|completion window is enforced for Job ?5|end time of (?:the|its) first operation|completion time of the first operation[^.]*(?:500|600)|Job ?5[^.]*(?:initial|first) operation"},
     {"id":6,"name":"sequence_dependent_setup",
      "regex":r"[Ss]equence-dependent setup|setup time is required whenever|setup time of \d+ units|when (?:the )?same machine switches"},
   ],
 },
 "Battery Pack Design": {
   "anchor": "battery_pack_design",
   "objective": "minimize_energy_loss",
   "base": ["parallel_activation_consistency", "battery_pack_voltage_range",
            "max_module_current", "load_demand_satisfaction"],
   "optional": [
     {"id":1,"name":"module_usage_balancing",
      "regex":r"usage counts of all modules[^.]*balanced|most-used and least-used|difference between (?:the )?most[- ]used and least[- ]used"},
     {"id":2,"name":"switch_count_limit",
      "regex":r"module switch states may change|no more than \d+ (?:module )?switch|switch states[^.]*between consecutive periods"},
     {"id":3,"name":"spatial_block_activation_limit",
      "regex":r"2 ?by ?2 block|2x2 block|every 2[x ]?2[^.]*block|block of neighbou?ring module"},
     {"id":4,"name":"faulty_module_isolation",
      "regex":r"faulty and must remain offline|module[^.]*faulty|must remain offline in every period"},
   ],
 },
 "Charging Station Location": {
   "anchor": "charging_station_location",
   "objective": "minimize_total_cost",
   "base": ["site_opening", "site_capacity", "demand_coverage",
            "demand_allocation"],
   "optional": [
     {"id":1,"name":"demand_area_expansion",
      # cover expanded / increased / grew / scaled up demand areas
      "regex":r"(?:number of\s+)?demand areas?[^.]*\b(?:expanded|increased|grew|grown|raised|scaled up)\b|\bfrom\s*30\s*to\s*40\b"},
     {"id":2,"name":"travel_cost_limit",
      "regex":r"service quality constraint|transportation cost[^.]*(?:shall not exceed|not exceed|no more than|at most|cannot exceed|must not exceed)"},
     {"id":3,"name":"budget_limit",
      "regex":r"budgetary (?:constraint|control)|budget range|sum of fixed costs[^.]*(?:within|between|range)"},
     {"id":4,"name":"operational_capacity_derating",
      # \d+% non- 90% cover 95%/90% etc.percentageupper limit
      "regex":r"operational capacity constraint|load capacity constraint|(?:actual load|operational load|load capacity)[^.]*(?:not exceed|no more than|at most)[^.]*\d+\s*%|not exceed\s*\d+\s*%\s*of[^.]*(?:design )?capacity"},
     {"id":5,"name":"site_diversity",
      "regex":r"strategic diversity constraint|geographic diversity|most similar cost|similar cost characteristics|nearest\s*\d+(?:\.\d+)?\s*%[^.]*(?:pairs?|sites?)|at most one of (?:them|any two)"},
   ],
 },
 "DNA Sequence Design": {
   "anchor": "dna_sequence_design",
   "objective": "feasibility",
   "base": ["gc_content", "watson_crick_pairing", "sequence_diversity",
            "reverse_complementarity"],
   "optional": [
     {"id":1,"name":"terminal_gc_stability",
      "regex":r"terminal stability|end of each DNA sequence must be either C or G|must end (?:with|in) (?:C or G|a C or G)|end with C or G|each (?:DNA )?sequence must end with"},
     {"id":2,"name":"base_composition_balance",
      "regex":r"base composition balance|total counts of any two bases|counts of any two bases differ"},
     {"id":3,"name":"dinucleotide_repetition_limit",
      "regex":r"pattern repetition constraint|dinucleotide|no (?:dinucleotide )?pattern may repeat"},
     {"id":4,"name":"forbidden_motif",
      "regex":r"forbidden motif|subsequence ACTG|forbidden (?:motif|pattern) ACTG|ACTG must not appear"},
   ],
 },
 "VRP": {
   "anchor": "vrp",
   "objective": "minimize_total_distance",
   "base": ["flow_conservation", "depot_departure_return", "vehicle_capacity",
            "subtour_elimination", "visit_once", "arc_existence"],
   "optional": [
     {"id":1,"name":"vehicle_capacity_relaxed",
      "regex":r"vehicle capacity is not enforced|capacity is not (?:enforced|used|applied|binding)|not used as a binding|without (?:a )?capacity constraint|no (?:binding )?capacity constraint", "relax":"vehicle_capacity"},
     {"id":2,"name":"vrp_time_window",
      "regex":r"ready time, due time, and service time|ready time.{0,40}due time|service time window|time window"},
     # id 3 is a parameter-only variant (capacity 200→180): no new TYPE → skip
     {"id":4,"name":"location_exclusion",
      "regex":r"is excluded from the delivery network"},
   ],
 },
}


def all_named_attributes() -> List[str]:
    """The complete attribute set M (sorted): anchors + objectives + every
    base and optional constraint type across all scenarios."""
    s: Set[str] = set()
    for spec in CATALOG.values():
        s.add(spec["anchor"]); s.add(spec["objective"])
        s.update(spec["base"])
        for opt in spec["optional"]:
            s.add(opt["name"])
    return sorted(s)


def attributes_for_type(problem_type: str) -> List[str]:
    """All attributes a problem of this scenario could possibly carry."""
    spec = CATALOG.get(problem_type)
    if not spec:
        return []
    out = {spec["anchor"], spec["objective"], *spec["base"]}
    out.update(o["name"] for o in spec["optional"])
    return sorted(out)


def _parse_label(label: str) -> Optional[Set[int]]:
    """Parse a `Constraint` cell ('base', '1,3', NaN, ...) into a set of ids.
    Returns set() for 'base', None when no usable label is present."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in ("", "nan", "none"):
        return None
    if s == "base":
        return set()
    nums = re.findall(r"\d+", s)
    return {int(n) for n in nums} if nums else set()


def attributes_for(record: dict) -> Set[str]:
    """
    Compute the attribute set (intent contribution) of one record.

    Incidence source priority:
      1. The numbered `Constraint` label (exact) — maps ids → named types.
      2. Fallback: NL signature regexes (for label-less / user-added records).
    """
    pt = (record.get("problem_type") or "").strip()
    spec = CATALOG.get(pt)
    if not spec:
        return set()

    attrs: Set[str] = {spec["anchor"], spec["objective"], *spec["base"]}
    ids = _parse_label(record.get("constraint", ""))

    if ids is not None:
        # label-driven (exact)
        by_id = {o["id"]: o for o in spec["optional"] if o.get("id") is not None}
        for i in ids:
            opt = by_id.get(i)
            if not opt:
                continue            # e.g. VRP id 3 (parameter-only)
            if opt.get("relax"):
                attrs.discard(opt["relax"])
            attrs.add(opt["name"])
    else:
        # NL fallback
        text = " ".join([record.get("nl_text", ""),
                         record.get("constraint_desc", "")])
        for opt in spec["optional"]:
            if re.search(opt["regex"], text):
                if opt.get("relax"):
                    attrs.discard(opt["relax"])
                attrs.add(opt["name"])
    return attrs
