"""CLI for the Step 16 operational workload simulation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from learnpulse.workload_simulation import *


def parser():
    p=argparse.ArgumentParser()
    for name in ("temporal-predictions","temporal-trajectories","temporal-results","episode-strategy-results","threshold-results","modeling-table","student-info"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--fairness-predictions",type=Path)
    p.add_argument("--temporal-policy-results",type=Path,default=Path("experiments/temporal_policy_results.csv"))
    p.add_argument("--calibration-predictions",type=Path,default=Path("experiments/calibration_predictions.parquet"))
    p.add_argument("--fairness-results",type=Path,default=Path("experiments/fairness_results.json"))
    p.add_argument("--observation-days",nargs="+",type=int,default=list(DAYS));p.add_argument("--primary-policy",default="f1_focused")
    p.add_argument("--sensitivity-policies",nargs="*",default=["fixed_0_50","recall_focused","alert_capacity"])
    p.add_argument("--capacity-rates",nargs="+",type=float,default=[.05,.10,.20]);p.add_argument("--fixed-capacities",nargs="*",type=int,default=[])
    p.add_argument("--deduplication-rules",nargs="+",default=list(DEDUP_RULES));p.add_argument("--priority-policies",nargs="+",default=list(PRIORITIES))
    p.add_argument("--bootstrap-samples",type=int,default=1000);p.add_argument("--random-seed",type=int,default=42);p.add_argument("--only-module");p.add_argument("--only-presentation")
    defaults={"results":"experiments/workload_simulation_results.json","scenario-results":"experiments/workload_scenario_results.csv","checkpoint-results":"experiments/workload_checkpoint_results.csv","course-results":"experiments/workload_course_results.csv","subgroup-results":"experiments/workload_subgroup_results.csv","bootstrap-results":"experiments/workload_bootstrap_intervals.csv","case-results":"experiments/workload_case_results.parquet","examples":"experiments/workload_examples.csv","figures-dir":"experiments/figures"}
    for name,value in defaults.items():p.add_argument(f"--{name}",type=Path,default=Path(value))
    return p

def main(argv=None):
    a=parser().parse_args(argv);start=time.time()
    policies=[a.primary_policy,*a.sensitivity_policies]
    if not a.observation_days or any(x not in DAYS for x in a.observation_days) or len(set(a.observation_days))!=len(a.observation_days):raise SystemExit("Observation days must be a unique subset of 14 28 42 56")
    if len(set(policies))!=len(policies) or any(x not in POLICY_COLUMNS for x in policies):raise SystemExit("Invalid or duplicate threshold policy")
    if any(x<=0 or x>1 for x in a.capacity_rates) or any(x<1 for x in a.fixed_capacities):raise SystemExit("Capacity rates must be in (0,1]; fixed capacities positive")
    if any(x not in DEDUP_RULES for x in a.deduplication_rules) or any(x not in PRIORITIES for x in a.priority_policies):raise SystemExit("Invalid queue policy")
    sources=[a.temporal_predictions,a.temporal_trajectories,a.temporal_results,a.episode_strategy_results,a.threshold_results,a.modeling_table,a.student_info]
    if a.fairness_predictions and a.fairness_predictions.exists():sources.append(a.fairness_predictions)
    for optional in (a.temporal_policy_results,a.calibration_predictions,a.fairness_results):
        if optional.exists():sources.append(optional)
    if any(not x.exists() for x in sources):raise SystemExit("A required input is missing")
    before=source_signatures(sources)
    events=load_inputs(a.temporal_predictions,a.modeling_table,a.temporal_trajectories,policies,a.only_module,a.only_presentation,a.observation_days)
    demographics=demographic_lookup(a.fairness_predictions,a.student_info)
    capacities=[Capacity("rate",x,f"rate_{x:.2f}") for x in a.capacity_rates]+[Capacity("fixed",x,f"fixed_{x}") for x in a.fixed_capacities]+[Capacity("unlimited",None,"unlimited")]
    audit=run_simulation(events,capacities,a.deduplication_rules,a.priority_policies,policies,demographics,a.bootstrap_samples,a.random_seed)
    after=source_signatures(sources);integrity={key:before[key]==after[key] for key in before}
    if not all(integrity.values()):raise RuntimeError("A source artifact changed during simulation")
    report={"step":"16_operational_workload_simulation","research_simulation_only":True,"models_refitted":False,"scores_recalibrated":False,"thresholds_reselected":False,
        "configuration":{"observation_days":a.observation_days,"threshold_policies":policies,"capacity_rates":a.capacity_rates,"fixed_capacities":a.fixed_capacities,"deduplication_rules":a.deduplication_rules,"priority_policies":a.priority_policies,"bootstrap_samples":a.bootstrap_samples,"random_seed":a.random_seed,"only_module":a.only_module,"only_presentation":a.only_presentation},
        "source_signatures_before":before,"source_integrity_after":integrity,"prediction_rows":len(events),"scenario_count":len(audit["scenarios"]),"case_rows":len(audit["cases"]),
        "checkpoint_rows":len(audit["checkpoints"]),"course_rows":len(audit["courses"]),"subgroup_rows":len(audit["subgroups"]),"bootstrap_rows":len(audit["bootstrap"]),
        "runtime_seconds":time.time()-start,"limitations":["Checkpoint cycles are not weekly predictions.","Review selection is not an intervention and does not estimate intervention effects.","Day-specific scores and labels are not one invariant longitudinal risk scale.","Demographics are audit-only and never affect ranking."]}
    targets={name.replace("_","-"):None for name in []}|{"results":a.results,"scenario_results":a.scenario_results,"checkpoint_results":a.checkpoint_results,"course_results":a.course_results,"subgroup_results":a.subgroup_results,"bootstrap_results":a.bootstrap_results,"case_results":a.case_results,"examples":a.examples,"figures_dir":a.figures_dir}
    publish(audit,report,targets)
    print(json.dumps({"scenarios":len(audit["scenarios"]),"cases":len(audit["cases"]),"runtime_seconds":time.time()-start,"source_integrity":all(integrity.values())},indent=2))

if __name__=="__main__":main()
