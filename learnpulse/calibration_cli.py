"""CLI for nested calibration and training-only threshold selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.calibration_experiment import (
    load_parquet_rows,
    refresh_saved_reporting,
    run_calibration_experiment,
    save_calibration_outputs,
)

DEFAULTS={"results":Path("experiments/calibration_results.json"),"holdout_results":Path("experiments/calibration_holdout_results.csv"),"predictions":Path("experiments/calibration_predictions.parquet"),"curves":Path("experiments/calibration_curves.csv"),"threshold_results":Path("experiments/threshold_results.csv"),"policies":Path("experiments/threshold_policies.json"),"figures":Path("experiments/figures")}


def parse_args():
    parser=argparse.ArgumentParser(description="Evaluate nested calibration and thresholds")
    parser.add_argument("--modeling-table",type=Path,required=True)
    parser.add_argument("--observation-days",nargs="+",type=int,default=[28,56])
    parser.add_argument("--calibration-methods",nargs="+",choices=["uncalibrated","sigmoid","isotonic"],default=["uncalibrated","sigmoid","isotonic"])
    parser.add_argument("--inner-folds",type=int,default=5); parser.add_argument("--threshold",type=float,default=.5)
    parser.add_argument("--recall-target",type=float,default=.8); parser.add_argument("--alert-capacity",type=float,default=.1)
    parser.add_argument("--false-negative-cost",type=float,default=5); parser.add_argument("--false-positive-cost",type=float,default=1)
    parser.add_argument("--random-seed",type=int,default=42)
    parser.add_argument("--results",type=Path,default=DEFAULTS["results"]); parser.add_argument("--holdout-results",type=Path,default=DEFAULTS["holdout_results"])
    parser.add_argument("--predictions",type=Path,default=DEFAULTS["predictions"]); parser.add_argument("--calibration-curves",type=Path,default=DEFAULTS["curves"])
    parser.add_argument("--threshold-results",type=Path,default=DEFAULTS["threshold_results"]); parser.add_argument("--threshold-policies",type=Path,default=DEFAULTS["policies"])
    parser.add_argument("--figures-dir",type=Path,default=DEFAULTS["figures"])
    parser.add_argument("--only-module"); parser.add_argument("--only-presentation"); parser.add_argument("--only-observation-day",type=int)
    parser.add_argument("--only-calibration-method",choices=["uncalibrated","sigmoid","isotonic"])
    parser.add_argument("--refresh-reporting-only",action="store_true",help="Correct saved JSON rate rankings and threshold charts without refitting")
    return parser.parse_args()


def main():
    args=parse_args(); filtered=any(value is not None for value in [args.only_module,args.only_presentation,args.only_observation_day,args.only_calibration_method])
    if args.refresh_reporting_only:
        if filtered: raise SystemExit("Reporting refresh cannot be combined with development filters")
        report=refresh_saved_reporting(args.results,args.holdout_results,args.threshold_results,args.predictions,args.figures_dir)
        print(json.dumps({"reporting_only":True,"outer_model_fits_reused":report["runtime"]["outer_model_fits"],"inner_model_fits_reused":report["runtime"]["inner_model_fits"]},sort_keys=True))
        return
    paths={"results":args.results,"holdout_results":args.holdout_results,"predictions":args.predictions,"curves":args.calibration_curves,"threshold_results":args.threshold_results,"policies":args.threshold_policies,"figures":args.figures_dir}
    if filtered and any(paths[name].resolve()==DEFAULTS[name].resolve() for name in paths): raise SystemExit("Filtered runs must use separate output and figures paths")
    days=[args.only_observation_day] if args.only_observation_day is not None else args.observation_days
    methods=[args.only_calibration_method] if args.only_calibration_method else args.calibration_methods
    try:
        rows=load_parquet_rows(args.modeling_table,days)
        result=run_calibration_experiment(rows,days,methods,args.inner_folds,args.threshold,args.recall_target,args.alert_capacity,args.false_negative_cost,args.false_positive_cost,args.random_seed,args.only_module,args.only_presentation)
        config={"modeling_table":str(args.modeling_table.resolve()),"observation_days":days,"calibration_methods":methods,"inner_folds":args.inner_folds,"fixed_threshold":args.threshold,"recall_target":args.recall_target,"alert_capacity":args.alert_capacity,"false_negative_cost":args.false_negative_cost,"false_positive_cost":args.false_positive_cost,"random_seed":args.random_seed,"only_module":args.only_module,"only_presentation":args.only_presentation,"filtered_run":filtered}
        report=save_calibration_outputs(result,args.modeling_table,args.results,args.holdout_results,args.predictions,args.calibration_curves,args.threshold_results,args.threshold_policies,args.figures_dir,config)
    except (ValueError,FileNotFoundError,RuntimeError) as error: raise SystemExit(str(error)) from error
    print(json.dumps({"outer_model_fits":result.report["outer_model_fits"],"inner_model_fits":result.report["inner_model_fits"],"prediction_rows":len(result.predictions),"runtime_seconds":result.report["runtime_seconds"],"filtered_run":filtered,"source_unchanged":report["source_integrity"]["unchanged"]},sort_keys=True))


if __name__=="__main__": main()
