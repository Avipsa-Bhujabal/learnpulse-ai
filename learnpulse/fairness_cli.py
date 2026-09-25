"""CLI for a read-only, strictly held-out subgroup audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from learnpulse.fairness_audit import (
    ATTRIBUTES,
    AuditConfig,
    load_joined_predictions,
    run_audit,
    save_audit,
    validate_saved_thresholds,
)

DEFAULTS={
    "results":Path("experiments/fairness_results.json"),
    "group_metrics":Path("experiments/fairness_group_metrics.csv"),
    "disparities":Path("experiments/fairness_disparities.csv"),
    "intersections":Path("experiments/fairness_intersections.csv"),
    "bootstrap_intervals":Path("experiments/fairness_bootstrap_intervals.csv"),
    "joined_predictions":Path("experiments/fairness_predictions.parquet"),
    "figures_dir":Path("experiments/figures"),
}


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Audit held-out sigmoid predictions by OULAD learner subgroup")
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--student-info",type=Path,required=True)
    parser.add_argument("--threshold-results",type=Path,default=Path("experiments/threshold_results.csv"))
    parser.add_argument("--calibration-results",type=Path,default=Path("experiments/calibration_results.json"))
    parser.add_argument("--observation-days",type=int,nargs="+",default=[28,56])
    parser.add_argument("--calibration-method",choices=["sigmoid"],default="sigmoid")
    parser.add_argument("--primary-policy",choices=["f1_focused"],default="f1_focused")
    parser.add_argument("--sensitivity-policies",nargs="+",choices=["fixed_0_50","recall_focused","alert_capacity"],default=["fixed_0_50","recall_focused","alert_capacity"])
    parser.add_argument("--bootstrap-samples",type=int,default=1000)
    parser.add_argument("--minimum-rows",type=int,default=50)
    parser.add_argument("--minimum-students",type=int,default=25)
    parser.add_argument("--minimum-positives",type=int,default=10)
    parser.add_argument("--minimum-negatives",type=int,default=10)
    parser.add_argument("--minimum-predicted-positives",type=int,default=10)
    parser.add_argument("--random-seed",type=int,default=42)
    parser.add_argument("--only-attribute",choices=ATTRIBUTES)
    parser.add_argument("--only-observation-day",type=int)
    parser.add_argument("--only-module")
    for key,path in DEFAULTS.items():
        parser.add_argument("--"+key.replace("_","-"),type=Path,default=path)
    return parser.parse_args()


def main() -> None:
    args=parse_args()
    config=AuditConfig(days=tuple(args.observation_days),primary_policy=args.primary_policy,
        sensitivity_policies=tuple(args.sensitivity_policies),bootstrap_samples=args.bootstrap_samples,
        minimum_rows=args.minimum_rows,minimum_students=args.minimum_students,
        minimum_positives=args.minimum_positives,minimum_negatives=args.minimum_negatives,
        minimum_predicted_positives=args.minimum_predicted_positives,random_seed=args.random_seed,
        only_attribute=args.only_attribute,only_day=args.only_observation_day,only_module=args.only_module)
    config.validate()
    targets={name:getattr(args,name) for name in DEFAULTS}
    if config.filtered and any(targets[name].resolve()==DEFAULTS[name].resolve() for name in DEFAULTS):
        raise SystemExit("Filtered audits require separate paths for every output and the figures directory")
    source_paths={"predictions":args.predictions,"student_info":args.student_info,
                  "threshold_results":args.threshold_results,"calibration_results":args.calibration_results}
    if any(not path.is_file() for path in source_paths.values()):
        raise SystemExit("A required saved calibration or OULAD source file is missing")
    started=time.perf_counter()
    try:
        rows,join=load_joined_predictions(args.predictions,args.student_info,config)
        join["saved_threshold_validation"]=validate_saved_thresholds(rows,args.threshold_results,
            (config.primary_policy,*config.sensitivity_policies))
        calibration=json.loads(args.calibration_results.read_text(encoding="utf-8"))
        if calibration["configuration"]["filtered_run"] or calibration["runtime"]["outer_model_fits"]!=44:
            raise ValueError("Expected complete strict Step 13 calibration results")
        audit=run_audit(rows,config)
        report=save_audit(audit,rows,join,config,source_paths,targets)
    except (FileNotFoundError,ValueError,RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"prediction_rows":len(rows),"group_rows":report["group_rows"],
        "supported_groups":report["supported_groups"],"suppressed_groups":report["suppressed_groups"],
        "elapsed_seconds":round(time.perf_counter()-started,3),"sources_unchanged":all(report["source_integrity"].values()),
        "filtered_run":config.filtered},sort_keys=True))


if __name__=="__main__": main()
