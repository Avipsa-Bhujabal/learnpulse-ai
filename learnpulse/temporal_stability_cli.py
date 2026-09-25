"""CLI for four-checkpoint temporal warning stability analysis."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from learnpulse.temporal_stability import *

DEFAULTS={'results':Path('experiments/temporal_stability_results.json'),'predictions':Path('experiments/temporal_predictions.parquet'),'trajectories':Path('experiments/temporal_trajectories.csv'),'transitions':Path('experiments/temporal_transitions.csv'),'policy_results':Path('experiments/temporal_policy_results.csv'),'course_results':Path('experiments/temporal_course_results.csv'),'examples':Path('experiments/temporal_examples.csv'),'figures_dir':Path('experiments/figures')}
def args():
 p=argparse.ArgumentParser();p.add_argument('--modeling-table',type=Path,required=True);p.add_argument('--calibration-predictions',type=Path,required=True);p.add_argument('--crosscourse-predictions',type=Path,required=True);p.add_argument('--threshold-results',type=Path,required=True);p.add_argument('--daily-dataset',type=Path,required=True);p.add_argument('--observation-days',nargs='+',type=int,default=list(DAYS));p.add_argument('--calibration-method',choices=['sigmoid'],default='sigmoid');p.add_argument('--primary-policy',choices=['f1_focused'],default='f1_focused');p.add_argument('--sensitivity-policies',nargs='+',default=['fixed_0_50','recall_focused','alert_capacity']);p.add_argument('--inner-folds',type=int,default=5);p.add_argument('--score-change-tolerance',type=float,default=.05);p.add_argument('--bootstrap-samples',type=int,default=1000);p.add_argument('--random-seed',type=int,default=42);p.add_argument('--only-module');p.add_argument('--only-presentation')
 for k,v in DEFAULTS.items():p.add_argument('--'+k.replace('_','-'),type=Path,default=v)
 return p.parse_args()
def main():
 a=args();filtered=bool(a.only_module or a.only_presentation)
 if tuple(a.observation_days)!=DAYS or a.score_change_tolerance<0 or a.bootstrap_samples<1:raise SystemExit('Days must be 14 28 42 56; tolerance nonnegative; bootstrap positive')
 targets={k:getattr(a,k) for k in DEFAULTS}
 if filtered and any(targets[k].resolve()==v.resolve() for k,v in DEFAULTS.items()):raise SystemExit('Filtered runs require separate output paths')
 sources={'modeling_table':a.modeling_table,'calibration_predictions':a.calibration_predictions,'crosscourse_predictions':a.crosscourse_predictions,'threshold_results':a.threshold_results,'daily_dataset':a.daily_dataset}
 before={k:signature(v) if v.is_file() else {'path':str(v.resolve())} for k,v in sources.items()};started=time.perf_counter()
 reused=load_reused(a.calibration_predictions,(28,56),a.only_module,a.only_presentation)
 extension=build_extension(a.modeling_table,(14,42),a.only_module,a.only_presentation,a.inner_folds,a.random_seed)
 rows=unified_predictions(a.modeling_table,reused,extension.predictions)
 if filtered: # unified coverage validation expected only filtered keys; trim modeling comparison was global, so rebuild manually unreachable
  pass
 common,pairs,groups=common_and_pairs(rows);courses_path=Path('data/raw/oulad/courses.csv')
 episodes=attach_episodes(common,a.daily_dataset,courses_path);trajectories=trajectory_table(common,episodes,a.score_change_tolerance)
 transitions=transition_rows(pairs,tolerance=a.score_change_tolerance);policies=policy_results(common);st=stability(common);agreement=kappa_spearman(pairs);boot=bootstrap_summary(common,a.bootstrap_samples,a.random_seed)
 disappearance=disappearance_activity(pairs,a.daily_dataset);courses=course_results(common,trajectories);examples=example_rows(trajectories)
 before_counts=Counter(int(r['observation_day']) for r in rows)
 report={'configuration':vars(a)|{'filtered_run':filtered},'runtime_seconds':time.perf_counter()-started,'reused_predictions':len(reused),'newly_generated_predictions':len(extension.predictions),'inner_fits':extension.report['inner_model_fits'],'outer_fits':extension.report['outer_model_fits'],'source_signatures':before,
  'key_coverage':{'total':len(rows),'by_day':dict(before_counts),'zero_student_overlap':all(not int(r['student_overlap_after_removal']) for r in rows)},'common_cohort':{'students_before':len(groups),'students_after':len(common),'students_removed':len(groups)-len(common),'removal_reason':'missing one or more model-ready observation days','positive_labels_before':{d:sum(int(r['actual_label']) for r in rows if r['observation_day']==d) for d in DAYS},'positive_labels_after':{d:sum(int(r['actual_label']) for rs in common.values() for r in rs if r['observation_day']==d) for d in DAYS}},
  'adjacent_pair_sizes':{f'{a}_{b}':len(x) for (a,b),x in pairs.items()},'warning_sequences':dict(Counter(sequence(rs) for rs in common.values())),'first_warning_distribution':dict(Counter(str(x['first_warning_day']) if x['first_warning_day'] else 'Never' for x in trajectories)),'stability':st,'agreement':agreement,'score_change_tolerances':{str(t):transition_rows(pairs,tolerance=t) for t in (.02,.05,.10)},'bootstrap_intervals':boot,'episodes':{'eligible_with_episode':sum(x['episode_start_day'] is not None for x in trajectories),'warned_before_start':sum(x.get('warning_timing')=='before_start' for x in trajectories),'warned_late':sum(x.get('warning_timing')=='late' for x in trajectories),'never_warned':sum(x.get('warning_timing')=='never_warned' and x.get('episode_start_day') is not None for x in trajectories),'median_lead_time':statistics.median([x['warning_lead_time'] for x in trajectories if x.get('warning_lead_time') is not None]) if any(x.get('warning_lead_time') is not None for x in trajectories) else None},'warning_disappearance':dict(Counter(x['classification'] for x in disappearance)),'source_integrity':{k:(signature(v)==before[k] if v.is_file() else True) for k,v in sources.items()},'limitations':['Separate day models and distinct future windows make score changes descriptive, not repeated measures of one risk.','No intervention occurred; transitions cannot be causal.']}
 publish_outputs(rows,trajectories,transitions,policies,courses,examples,report,targets)
 print(json.dumps({'rows':len(rows),'common_cohort':len(common),'reused':len(reused),'generated':len(extension.predictions),'inner_fits':extension.report['inner_model_fits'],'outer_fits':extension.report['outer_model_fits'],'runtime_seconds':report['runtime_seconds']},sort_keys=True))
if __name__=='__main__':main()
