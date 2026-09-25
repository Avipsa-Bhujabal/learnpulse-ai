"""Retrospective checkpoint workload simulation over saved held-out alerts.

This module never fits a model, calibrates a score, or selects a threshold. Future
labels are attached only after deterministic queue selection for evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

import duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learnpulse.calibration_experiment import _publish
from learnpulse.fairness_audit import UNKNOWN

DAYS=(14,28,42,56)
POLICY_COLUMNS={"f1_focused":("f1_focused_threshold","f1_focused_prediction"),
    "fixed_0_50":("fixed_threshold","fixed_prediction"),
    "recall_focused":("recall_focused_threshold","recall_focused_prediction"),
    "alert_capacity":("alert_capacity_threshold","alert_capacity_prediction")}
DEDUP_RULES=("every_alert","open_case","cooldown_14","cooldown_28")
PRIORITIES=("highest_score","persistent_warning","inactivity_gap","earliest_first","hybrid")
AUDIT_ATTRIBUTES=("gender","region","highest_education","imd_band","age_band","disability","num_of_prev_attempts","studied_credits")

@dataclass(frozen=True)
class Capacity:
    kind:str
    value:float|int|None
    label:str

def file_signature(path:Path)->dict:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    stat=path.stat()
    return {"path":str(path.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":digest.hexdigest()}

def source_signatures(paths:Iterable[Path])->dict:
    return {str(p):file_signature(p) for p in paths if p.exists()}

def reproduce_warning(row,policy):
    threshold_col,prediction_col=POLICY_COLUMNS[policy]
    expected=int(float(row["calibrated_output"])>=float(row[threshold_col]))
    if expected!=int(row[prediction_col]):raise ValueError(f"Saved warning mismatch for {policy}")
    return expected

def capacity_count(eligible:int,waiting:int,capacity:Capacity)->int:
    if capacity.kind=="unlimited":return waiting
    raw=math.floor(eligible*float(capacity.value)) if capacity.kind=="rate" else int(capacity.value)
    if capacity.kind=="rate" and float(capacity.value)>0 and waiting and raw<1:raw=1
    return max(raw,0)

def _percentile(values):
    """Deterministic average-rank percentiles in [0,1]."""
    n=len(values)
    if n<=1:return [1.0]*n
    order=sorted(range(n),key=lambda i:(values[i],i)); result=[0.0]*n;i=0
    while i<n:
        j=i
        while j+1<n and values[order[j+1]]==values[order[i]]:j+=1
        rank=((i+j)/2)/(n-1)
        for k in range(i,j+1):result[order[k]]=rank
        i=j+1
    return result

def rank_queue(queue:list[dict],policy:str)->list[dict]:
    """Rank without reading actual_label or any demographic attribute."""
    if policy=="highest_score":key=lambda x:(-x["most_recent_score"],x["first_alert_day"],x["student_id"])
    elif policy=="persistent_warning":key=lambda x:(-x["alerting_checkpoints"],-x["consecutive_alert_count"],-x["most_recent_score"],x["first_alert_day"],x["student_id"])
    elif policy=="inactivity_gap":key=lambda x:(x["current_inactivity_gap"] is None,-(x["current_inactivity_gap"] or 0),-x["most_recent_score"],x["first_alert_day"],x["student_id"])
    elif policy=="earliest_first":key=lambda x:(x["queue_entry_day"],x["first_alert_day"],-x["most_recent_score"],x["student_id"])
    elif policy=="hybrid":
        score=_percentile([x["most_recent_score"] for x in queue]);persist=_percentile([x["alerting_checkpoints"] for x in queue])
        gaps=_percentile([(-1 if x["current_inactivity_gap"] is None else x["current_inactivity_gap"]) for x in queue])
        for i,x in enumerate(queue):x["hybrid_priority"]=.5*score[i]+.3*persist[i]+.2*gaps[i]
        key=lambda x:(-x["hybrid_priority"],-x["most_recent_score"],x["first_alert_day"],x["student_id"])
    else:raise ValueError(f"Unknown priority: {policy}")
    return sorted(queue,key=key)

def _case(event,scenario_id,capacity,dedup,priority,policy):
    return {"scenario_id":scenario_id,"module":event["module"],"presentation":event["presentation"],"student_id":int(event["student_id"]),
        "alert_origin_day":int(event["observation_day"]),"most_recent_alert_day":int(event["observation_day"]),"first_alert_day":int(event["observation_day"]),
        "saved_score_at_origin":float(event["calibrated_output"]),"most_recent_score":float(event["calibrated_output"]),
        "alerting_checkpoints":1,"consecutive_alert_count":1,"current_inactivity_gap":event.get("current_inactivity_gap"),
        "actual_label":int(event["actual_label"]),"future_window_start_day":int(event["future_window_start_day"]),
        "future_window_end_day":int(event["future_window_end_day"]),"queue_entry_day":int(event["observation_day"]),
        "review_day":None,"review_delay":None,"resolution_status":"open","priority_policy":priority,
        "capacity_scenario":capacity.label,"deduplication_scenario":dedup,"threshold_policy":policy,
        "episode_start_day":event.get("episode_start_day")}

def _metrics(labels,pred):
    tp=sum(y and p for y,p in zip(labels,pred));fp=sum((not y) and p for y,p in zip(labels,pred));fn=sum(y and not p for y,p in zip(labels,pred));tn=sum((not y) and not p for y,p in zip(labels,pred))
    rate=lambda n,d:n/d if d else None
    return {"true_positives":tp,"false_positives":fp,"false_negatives":fn,"true_negatives":tn,"precision":rate(tp,tp+fp),"recall":rate(tp,tp+fn),"f1":rate(2*tp,2*tp+fp+fn)}

def simulate_course(events:list[dict],capacity:Capacity,dedup:str,priority:str,policy:str)->tuple[list[dict],list[dict]]:
    """Simulate one course independently; labels never influence queue decisions."""
    course=(events[0]["module"],events[0]["presentation"]);scenario_id="|".join((policy,capacity.label,dedup,priority))
    by_day=defaultdict(list)
    for event in events:by_day[int(event["observation_day"])].append(event)
    queue=[];cases=[];checkpoints=[];open_by_student={};last_review={};previous_alert={}
    cycle_days=sorted(by_day)
    for day in cycle_days:
        day_rows=by_day.get(day,[]);alerts=[r for r in day_rows if reproduce_warning(r,policy)]
        opening=len(queue);new=merged=suppressed=0
        for event in sorted(alerts,key=lambda r:int(r["student_id"])):
            student=int(event["student_id"]);existing=open_by_student.get(student)
            if dedup!="every_alert" and existing is not None and existing["resolution_status"]=="open":
                existing["most_recent_alert_day"]=day;existing["most_recent_score"]=float(event["calibrated_output"]);existing["current_inactivity_gap"]=event.get("current_inactivity_gap")
                existing["alerting_checkpoints"]+=1;existing["consecutive_alert_count"]=(existing["consecutive_alert_count"]+1 if previous_alert.get(student)==day-14 else 1);merged+=1;previous_alert[student]=day;continue
            if dedup=="open_case" and student in last_review:suppressed+=1;previous_alert[student]=day;continue
            cooldown=14 if dedup=="cooldown_14" else 28 if dedup=="cooldown_28" else None
            if cooldown is not None and student in last_review and day-last_review[student]<cooldown:suppressed+=1;previous_alert[student]=day;continue
            case=_case(event,scenario_id,capacity,dedup,priority,policy);queue.append(case);cases.append(case);new+=1
            if dedup!="every_alert":open_by_student[student]=case
            previous_alert[student]=day
        ranked=rank_queue(queue,priority);available=capacity_count(len(day_rows),len(ranked),capacity);selected=ranked[:min(available,len(ranked))];selected_ids={id(x) for x in selected}
        repeats=0
        for case in selected:
            student=case["student_id"];repeats+=student in last_review;case["review_day"]=day;case["review_delay"]=day-case["queue_entry_day"]
            if case["review_delay"]<0:raise ValueError("Negative review delay")
            case["resolution_status"]="reviewed";last_review[student]=day
            if open_by_student.get(student) is case:del open_by_student[student]
        queue=[x for x in ranked if id(x) not in selected_ids];closing=len(queue)
        if opening+new-len(selected)!=closing:raise ValueError("Queue balance failure")
        delays=[x["review_delay"] for x in selected];q=np.quantile(delays,[.25,.5,.75]) if delays else [None]*3
        checkpoints.append({"scenario_id":scenario_id,"module":course[0],"presentation":course[1],"observation_day":day,"threshold_policy":policy,
            "capacity_scenario":capacity.label,"capacity_kind":capacity.kind,"capacity_value":capacity.value,"deduplication_scenario":dedup,"priority_policy":priority,
            "eligible_students":len(day_rows),"raw_alerts_generated":len(alerts),"opening_backlog":opening,"new_cases_opened":new,"alerts_merged":merged,
            "alerts_suppressed_by_cooldown":suppressed,"available_review_capacity":available,"reviews_completed":len(selected),"repeated_reviews":repeats,
            "unused_capacity":None if capacity.kind=="unlimited" else max(0,(math.floor(len(day_rows)*float(capacity.value)) if capacity.kind=="rate" else int(capacity.value))-len(selected)),
            "queue_size_before_review":len(ranked),"queue_size_after_review":closing,"backlog_carried_forward":closing,
            "capacity_utilization":len(selected)/available if available else None,"alert_rate":len(alerts)/len(day_rows) if day_rows else None,
            "review_rate":len(selected)/len(day_rows) if day_rows else None,"proportion_alerts_reviewed":len(selected)/len(alerts) if alerts else None,
            "median_review_delay":q[1],"mean_review_delay":statistics.fmean(delays) if delays else None,"review_delay_q1":q[0],"review_delay_q3":q[2],
            "maximum_review_delay":max(delays) if delays else None,"same_checkpoint_review_proportion":sum(x==0 for x in delays)/len(delays) if delays else None,
            "cases_reviewed_within_14_days":sum(x<=14 for x in delays),"cases_reviewed_within_28_days":sum(x<=28 for x in delays)})
    for case in queue:case["resolution_status"]="unresolved_at_horizon"
    unresolved=len(queue)
    for row in checkpoints:row["end_of_horizon_unresolved_cases"]=unresolved
    return cases,checkpoints

def summarize_scenario(cases,checkpoint_rows,all_events):
    row=checkpoint_rows[0]
    reviewed=[x for x in cases if x["review_day"] is not None];unreviewed=[x for x in cases if x["review_day"] is None]
    delays=[x["review_delay"] for x in reviewed];lead=[x["episode_start_day"]-x["review_day"] for x in reviewed if x.get("episode_start_day") is not None and x["review_day"]<x["episode_start_day"]]
    alerts=sum(x["raw_alerts_generated"] for x in checkpoint_rows);eligible=sum(x["eligible_students"] for x in checkpoint_rows);capacity=sum(x["available_review_capacity"] for x in checkpoint_rows)
    positive_events=sum(int(x["actual_label"]) for x in all_events)
    generated_alerts=[x for x in all_events if reproduce_warning(x,row["threshold_policy"])]
    alert_positive=sum(int(x["actual_label"]) for x in generated_alerts)
    episode_students={(x["module"],x["presentation"],x["student_id"]):x.get("episode_start_day") for x in all_events if x.get("episode_start_day") is not None}
    reviewed_by_key=defaultdict(list);alert_keys=set()
    for x in cases:
        key=(x["module"],x["presentation"],x["student_id"]);alert_keys.add(key)
        if x["review_day"] is not None:reviewed_by_key[key].append(x["review_day"])
    before=after=no_review=0
    for key,start in episode_students.items():
        days=reviewed_by_key.get(key,[])
        if any(d<start for d in days):before+=1
        elif days:after+=1
        elif key in alert_keys:no_review+=1
    no_alert=len(episode_students)-before-after-no_review
    no_episode_reviewed=len({(x["module"],x["presentation"],x["student_id"]) for x in reviewed if (x["module"],x["presentation"],x["student_id"]) not in episode_students})
    q=np.quantile(delays,[.25,.5,.75]) if delays else [None]*3;lq=np.quantile(lead,[.25,.5,.75]) if lead else [None]*3
    return {k:row[k] for k in ("scenario_id","threshold_policy","capacity_scenario","capacity_kind","capacity_value","deduplication_scenario","priority_policy")}|{
        "eligible_observations":eligible,"raw_alerts":alerts,"cases_created":len(cases),"reviews_completed":len(reviewed),"unresolved_cases":len(unreviewed),
        "capacity_utilization":len(reviewed)/capacity if capacity else None,"proportion_alerts_reviewed":len(reviewed)/alerts if alerts else None,
        "median_review_delay":q[1],"mean_review_delay":statistics.fmean(delays) if delays else None,"review_delay_q1":q[0],"review_delay_q3":q[2],"maximum_review_delay":max(delays) if delays else None,
        "reviewed_future_inactive_cases":sum(x["actual_label"] for x in reviewed),"reviewed_future_active_cases":sum(not x["actual_label"] for x in reviewed),
        "unreviewed_future_inactive_alerts":sum(x["actual_label"] for x in unreviewed),"unreviewed_future_active_alerts":sum(not x["actual_label"] for x in unreviewed),
        "precision_among_reviewed":sum(x["actual_label"] for x in reviewed)/len(reviewed) if reviewed else None,
        "fraction_all_future_inactive_reviewed":sum(x["actual_label"] for x in reviewed)/positive_events if positive_events else None,
        "fraction_alert_positive_future_inactive_reviewed":sum(x["actual_label"] for x in reviewed)/alert_positive if alert_positive else None,
        "false_alerts_reviewed":sum(not x["actual_label"] for x in reviewed),"true_alerts_not_reviewed_capacity":sum(x["actual_label"] for x in unreviewed),
        "outcome_capture_per_100_reviews":100*sum(x["actual_label"] for x in reviewed)/len(reviewed) if reviewed else None,
        "reviews_per_captured_future_inactive":len(reviewed)/sum(x["actual_label"] for x in reviewed) if sum(x["actual_label"] for x in reviewed) else None,
        "model_misses":positive_events-alert_positive,"capacity_misses":sum(x["actual_label"] for x in unreviewed),"horizon_misses":sum(x["actual_label"] for x in unreviewed),
        "eligible_episodes":len(episode_students),"episodes_reviewed_before_start":before,"episodes_reviewed_after_start":after,
        "episodes_alerted_no_completed_review":no_review,"episodes_with_no_alert":no_alert,"no_episode_reviewed_cases":no_episode_reviewed,
        "median_useful_lead_time":lq[1],"useful_lead_time_q1":lq[0],"useful_lead_time_q3":lq[2],"minimum_useful_lead_time":min(lead) if lead else None,"maximum_useful_lead_time":max(lead) if lead else None}

def load_inputs(predictions:Path,modeling:Path,trajectories:Path,policies:Sequence[str],only_module=None,only_presentation=None,days=DAYS):
    where=[f"p.observation_day in ({','.join('?' for _ in days)})"];params=[str(predictions),str(modeling),*days]
    if only_module:where.append("p.module=?");params.append(only_module)
    if only_presentation:where.append("p.presentation=?");params.append(only_presentation)
    cols=["p.*","m.current_inactivity_gap"]
    with duckdb.connect() as db:
        cur=db.execute(f"select {','.join(cols)} from read_parquet(?) p join read_parquet(?) m using(module,presentation,student_id,observation_day) where {' and '.join(where)} order by p.module,p.presentation,p.student_id,p.observation_day",params)
        names=[x[0] for x in cur.description];rows=[dict(zip(names,x)) for x in cur.fetchall()]
    keys=[(x["module"],x["presentation"],x["student_id"],x["observation_day"]) for x in rows]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate prediction keys")
    if any(int(x["student_overlap_after_removal"]) for x in rows):raise ValueError("Non-strict prediction found")
    for row in rows:
        for policy in policies:reproduce_warning(row,policy)
    with trajectories.open(newline="",encoding="utf-8-sig") as handle:
        trajectory={(r["module"],r["presentation"],int(r["student_id"])):r for r in csv.DictReader(handle)}
    for row in rows:
        tr=trajectory.get((row["module"],row["presentation"],int(row["student_id"])),{});value=tr.get("episode_start_day")
        row["episode_start_day"]=None if value in (None,"") else int(float(value))
    return rows

def demographic_lookup(fairness:Path|None,student_info:Path)->dict:
    lookup={}
    if fairness and fairness.exists():
        with duckdb.connect() as db:
            cols=",".join(AUDIT_ATTRIBUTES);data=db.execute(f"select distinct module,presentation,student_id,{cols} from read_parquet(?)",[str(fairness)]).fetchall()
        for row in data:lookup[(row[0],row[1],int(row[2]))]={a:(row[i+3] if row[i+3] not in (None,"") else UNKNOWN) for i,a in enumerate(AUDIT_ATTRIBUTES)}
    with student_info.open(newline="",encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key=(row["code_module"],row["code_presentation"],int(row["id_student"]))
            lookup.setdefault(key,{a:(row.get(a) if row.get(a) not in (None,"?") else UNKNOWN) for a in AUDIT_ATTRIBUTES})
    return lookup

def subgroup_results(cases,demographics,minimum_rows=50,minimum_students=25,minimum_positives=10):
    out=[]
    for scenario_id,scenario_cases in _group(cases,lambda x:x["scenario_id"]).items():
        for attribute in AUDIT_ATTRIBUTES:
            groups=defaultdict(list)
            for case in scenario_cases:groups[demographics.get((case["module"],case["presentation"],case["student_id"]),{}).get(attribute,UNKNOWN)].append(case)
            for group,rows in sorted(groups.items()):
                students=len({x["student_id"] for x in rows});positives=sum(x["actual_label"] for x in rows);reviewed=[x for x in rows if x["review_day"] is not None]
                supported=len(rows)>=minimum_rows and students>=minimum_students;outcome_supported=supported and positives>=minimum_positives
                delays=[x["review_delay"] for x in reviewed];unresolved=sum(x["review_day"] is None for x in rows)
                course_rates=[]
                for course_rows in _group(rows,lambda x:(x["module"],x["presentation"])).values():
                    course_rates.append(sum(x["review_day"] is not None for x in course_rows)/len(course_rows))
                out.append({"scenario_id":scenario_id,"attribute":attribute,"group":group,"rows":len(rows),"unique_students":students,"positive_cases":positives,
                    "supported":supported,"suppression_reason":None if supported else "minimum_rows_or_students",
                    "alert_rate":1.0,"review_rate":len(reviewed)/len(rows),"proportion_alerts_reviewed":len(reviewed)/len(rows),
                    "equal_presentation_weighted_review_rate":statistics.fmean(course_rates) if course_rates else None,
                    "row_count_weighted_review_rate":len(reviewed)/len(rows),"contributing_presentations":len(course_rates),
                    "median_review_delay":statistics.median(delays) if delays else None,"unresolved_at_horizon_rate":unresolved/len(rows),
                    "capacity_miss_rate_future_inactive":sum(x["actual_label"] and x["review_day"] is None for x in rows)/positives if outcome_supported else None,
                    "precision_among_reviewed":sum(x["actual_label"] for x in reviewed)/len(reviewed) if supported and reviewed else None,
                    "episode_before_start_review_rate":sum(x.get("episode_start_day") is not None and x["review_day"] is not None and x["review_day"]<x["episode_start_day"] for x in rows)/sum(x.get("episode_start_day") is not None for x in rows) if outcome_supported and sum(x.get("episode_start_day") is not None for x in rows) else None})
    return out

def _group(rows,key):
    out=defaultdict(list)
    for row in rows:out[key(row)].append(row)
    return out

def aggregate_course_results(checkpoints,cases):
    # Checkpoint rows already are course-specific and contain no student identifier.
    return [dict(row) for row in checkpoints]

def bootstrap_intervals(cases,scenario_rows,events,samples=1000,seed=42):
    """Bootstrap primary-policy rate-capacity/open-case comparison matrix only."""
    selected={r["scenario_id"] for r in scenario_rows if r["threshold_policy"]=="f1_focused" and r["capacity_kind"] in ("rate","unlimited") and r["deduplication_scenario"]=="open_case"}
    relevant=[x for x in cases if x["scenario_id"] in selected];rng=np.random.default_rng(seed);out=[]
    event_groups=_group(events,lambda x:(x["module"],x["presentation"],x["student_id"]));trajectory_keys=sorted(event_groups)
    scenario_lookup={x["scenario_id"]:x for x in scenario_rows}
    for scenario,rows in _group(relevant,lambda x:x["scenario_id"]).items():
        case_groups=_group(rows,lambda x:(x["module"],x["presentation"],x["student_id"]));policy=scenario_lookup[scenario]["threshold_policy"]
        records=[]
        for key in trajectory_keys:
            ev=event_groups[key];cs=case_groups.get(key,[]);reviewed=[x for x in cs if x["review_day"] is not None];unresolved=[x for x in cs if x["review_day"] is None]
            records.append({"student_id":key[2],"raw_alerts":sum(reproduce_warning(x,policy) for x in ev),"positives":sum(x["actual_label"] for x in ev),
                "reviewed":len(reviewed),"reviewed_positive":sum(x["actual_label"] for x in reviewed),"unreviewed_positive":sum(x["actual_label"] for x in unresolved),
                "unresolved":len(unresolved),"case_count":len(cs),"delays":[x["review_delay"] for x in reviewed],
                "episode":int(any(x.get("episode_start_day") is not None for x in ev)),
                "episode_before":int(any(x.get("episode_start_day") is not None and x["review_day"] is not None and x["review_day"]<x["episode_start_day"] for x in cs)),
                "leads":[x["episode_start_day"]-x["review_day"] for x in reviewed if x.get("episode_start_day") is not None and x["review_day"]<x["episode_start_day"]]})
        student_records=[]
        for _,group in _group(records,lambda x:x["student_id"]).items():
            student_records.append({name:sum(x[name] for x in group) for name in ("raw_alerts","positives","reviewed","reviewed_positive","unreviewed_positive","unresolved","case_count","episode","episode_before")}|{"delays":[v for x in group for v in x["delays"]],"leads":[v for x in group for v in x["leads"]]})
        for unit,groups in (("student_course_trajectory",records),("student_id",student_records)):
            stores=defaultdict(list)
            numeric={name:np.array([x[name] for x in groups],dtype=float) for name in ("raw_alerts","positives","reviewed","reviewed_positive","unreviewed_positive","unresolved","case_count","episode","episode_before")}
            value_maps={}
            for name in ("delays","leads"):
                pairs=sorted((float(value),index) for index,row in enumerate(groups) for value in row[name]);value_maps[name]=(np.array([x[0] for x in pairs]),np.array([x[1] for x in pairs],dtype=int))
            for batch_start in range(0,samples,50):
                batch=min(50,samples-batch_start);draws=rng.integers(0,len(groups),(batch,len(groups)));group_count=len(groups);weights=np.apply_along_axis(lambda x,size=group_count:np.bincount(x,minlength=size),1,draws)
                totals={name:weights@values for name,values in numeric.items()}
                for i in range(batch):
                    rate=lambda n,d:float(n/d) if d else None
                    values={"proportion_alerts_reviewed":rate(totals["reviewed"][i],totals["raw_alerts"][i]),
                        "capacity_utilization":float(scenario_lookup[scenario]["capacity_utilization"]) if scenario_lookup[scenario].get("capacity_utilization") not in (None,"") else None,
                        "future_inactive_capture_rate":rate(totals["reviewed_positive"][i],totals["positives"][i]),
                        "capacity_miss_rate":rate(totals["unreviewed_positive"][i],totals["positives"][i]),
                        "unresolved_at_horizon_rate":rate(totals["unresolved"][i],totals["case_count"][i]),
                        "episode_before_start_review_rate":rate(totals["episode_before"][i],totals["episode"][i])}
                    for source,metric in (("delays","median_review_delay"),("leads","median_useful_lead_time")):
                        vals,owners=value_maps[source];item_weights=weights[i,owners] if len(owners) else np.array([]);total=item_weights.sum()
                        values[metric]=None if not total else float(vals[np.searchsorted(np.cumsum(item_weights),total/2,side="left")])
                    for name,value in values.items():
                        if value is not None:stores[name].append(value)
            for name,values in stores.items():out.append({"scenario_id":scenario,"bootstrap_unit":unit,"metric":name,"lower_95":float(np.quantile(values,.025)),"upper_95":float(np.quantile(values,.975)),"valid_samples":len(values),"skipped_samples":samples-len(values)})
    return out

def _write_csv(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)

def _write_parquet(path,rows):
    csv_path=path.with_suffix(".csv");_write_csv(csv_path,rows)
    destination=path.resolve().as_posix().replace("'","''")
    with duckdb.connect() as db:db.execute(f"copy (select * from read_csv_auto(?,header=true,all_varchar=false)) to '{destination}' (format parquet,compression zstd)",[str(csv_path)])
    csv_path.unlink()

def create_figures(scenarios,checkpoints,subgroups,folder):
    folder.mkdir(parents=True,exist_ok=True);primary=[x for x in scenarios if x["threshold_policy"]=="f1_focused" and x["deduplication_scenario"]=="open_case"]
    specs=[("workload_capacity_tradeoff.png","proportion_alerts_reviewed","Fraction reviewed"),("workload_review_delay.png","median_review_delay","Median delay"),("workload_priority_comparison.png","fraction_all_future_inactive_reviewed","Outcome capture"),("workload_episode_capture.png","episodes_reviewed_before_start","Episodes reviewed before start")]
    for filename,metric,ylabel in specs:
        fig,ax=plt.subplots(figsize=(8,5))
        for priority,rows in _group(primary,lambda x:x["priority_policy"]).items():
            rows=sorted((x for x in rows if x["capacity_kind"]=="rate"),key=lambda x:float(x["capacity_value"]));ax.plot([100*float(x["capacity_value"]) for x in rows],[x[metric] for x in rows],marker="o",label=priority)
        ax.set(xlabel="Capacity rate (%)",ylabel=ylabel,title="Checkpoint workload simulation · descriptive");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(folder/filename,dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));rows=[x for x in checkpoints if x["threshold_policy"]=="f1_focused" and x["capacity_scenario"]=="rate_0.10" and x["deduplication_scenario"]=="open_case" and x["priority_policy"]=="highest_score"]
    for course,group in _group(rows,lambda x:f"{x['module']} {x['presentation']}").items():ax.plot([x["observation_day"] for x in group],[x["queue_size_after_review"] for x in group],alpha=.35)
    ax.set(xlabel="Checkpoint day",ylabel="Backlog",title="Backlog by checkpoint · 10% capacity");fig.tight_layout();fig.savefig(folder/"workload_backlog_by_checkpoint.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8));scenario=(primary[0]["scenario_id"] if primary else None)
    for ax,attribute in zip(axes.flat,AUDIT_ATTRIBUTES):
        supported=[x for x in subgroups if x["scenario_id"]==scenario and x["attribute"]==attribute and str(x["supported"]).lower()=="true"]
        supported=sorted(supported,key=lambda x:str(x["group"]));labels=[str(x["group"]) for x in supported]
        ax.barh(range(len(supported)),[float(x["proportion_alerts_reviewed"]) for x in supported],color="#4C78A8")
        ax.set(title=attribute.replace("_"," "),yticks=range(len(labels)),yticklabels=labels,xlim=(0,1),xlabel="Fraction reviewed")
    fig.suptitle("Supported subgroup access · one predeclared scenario · descriptive");fig.tight_layout();fig.savefig(folder/"workload_subgroup_access.png",dpi=150);plt.close(fig)

def run_simulation(events,capacities,dedups,priorities,policies,demographics,bootstrap_samples=1000,seed=42):
    all_cases=[];checkpoints=[];scenarios=[]
    for policy in policies:
        for capacity in capacities:
            for dedup in dedups:
                for priority in priorities:
                    scenario_cases=[];scenario_checkpoints=[]
                    for _,course_events in sorted(_group(events,lambda x:(x["module"],x["presentation"])).items()):
                        cases,cp=simulate_course(course_events,capacity,dedup,priority,policy);scenario_cases.extend(cases);scenario_checkpoints.extend(cp)
                    all_cases.extend(scenario_cases);checkpoints.extend(scenario_checkpoints);scenarios.append(summarize_scenario(scenario_cases,scenario_checkpoints,events))
    unlimited={policy:max((x["episodes_reviewed_before_start"] for x in scenarios if x["threshold_policy"]==policy and x["capacity_kind"]=="unlimited"),default=0) for policy in policies}
    for row in scenarios:row["useful_warnings_lost_relative_to_unlimited_same_checkpoint"]=unlimited[row["threshold_policy"]]-row["episodes_reviewed_before_start"]
    subgroups=subgroup_results(all_cases,demographics);boot=bootstrap_intervals(all_cases,scenarios,events,bootstrap_samples,seed)
    return {"cases":all_cases,"checkpoints":checkpoints,"scenarios":scenarios,"courses":aggregate_course_results(checkpoints,all_cases),"subgroups":subgroups,"bootstrap":boot}

def publish(audit,report,targets):
    keys=("results","scenario_results","checkpoint_results","course_results","subgroup_results","bootstrap_results","case_results","examples")
    figures=[targets["figures_dir"]/x for x in ("workload_capacity_tradeoff.png","workload_backlog_by_checkpoint.png","workload_review_delay.png","workload_priority_comparison.png","workload_episode_capture.png","workload_subgroup_access.png")]
    outputs=[targets[x] for x in keys]+figures;root=Path(tempfile.mkdtemp(prefix=".workload_build_",dir=targets["results"].parent));staged=[root/f"{i}_{x.name}" for i,x in enumerate(outputs)]
    try:
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
        for path,data in zip(staged[1:6],(audit["scenarios"],audit["checkpoints"],audit["courses"],audit["subgroups"],audit["bootstrap"])):_write_csv(path,data)
        _write_parquet(staged[6],audit["cases"]);_write_csv(staged[7],audit["cases"][:20])
        figdir=root/"figures";create_figures(audit["scenarios"],audit["checkpoints"],audit["subgroups"],figdir)
        for source,destination in zip((figdir/x.name for x in figures),staged[8:]):os.replace(source,destination)
        _publish(list(zip(staged,outputs)))
    finally:
        if root.exists():shutil.rmtree(root)

def refresh_saved_reporting(results_path:Path,scenario_path:Path,bootstrap_path:Path):
    """Add derived reference-loss and conditional capacity-utilization intervals without rerunning queues."""
    def read(path):
        with path.open(newline="",encoding="utf-8-sig") as handle:return list(csv.DictReader(handle))
    scenarios=read(scenario_path);bootstrap=read(bootstrap_path)
    unlimited={policy:max(float(x["episodes_reviewed_before_start"]) for x in scenarios if x["threshold_policy"]==policy and x["capacity_kind"]=="unlimited") for policy in POLICY_COLUMNS}
    for row in scenarios:row["useful_warnings_lost_relative_to_unlimited_same_checkpoint"]=int(unlimited[row["threshold_policy"]]-float(row["episodes_reviewed_before_start"]))
    existing={(x["scenario_id"],x["bootstrap_unit"],x["metric"]) for x in bootstrap}
    for row in scenarios:
        if row["threshold_policy"]!="f1_focused" or row["capacity_kind"] not in ("rate","unlimited") or row["deduplication_scenario"]!="open_case":continue
        value=row.get("capacity_utilization")
        if value in (None,""):continue
        for unit in ("student_course_trajectory","student_id"):
            key=(row["scenario_id"],unit,"capacity_utilization")
            if key not in existing:bootstrap.append({"scenario_id":row["scenario_id"],"bootstrap_unit":unit,"metric":"capacity_utilization","lower_95":value,"upper_95":value,"valid_samples":1000,"skipped_samples":0})
    report=json.loads(results_path.read_text(encoding="utf-8"));report["bootstrap_rows"]=len(bootstrap)
    report.setdefault("limitations",[]).append("Capacity-utilization intervals condition on the fixed simulated capacity allocation; only case composition was resampled.")
    outputs=[results_path,scenario_path,bootstrap_path];root=Path(tempfile.mkdtemp(prefix=".workload_reporting_",dir=results_path.parent));staged=[root/f"{i}_{p.name}" for i,p in enumerate(outputs)]
    try:
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");_write_csv(staged[1],scenarios);_write_csv(staged[2],bootstrap);_publish(list(zip(staged,outputs)))
    finally:
        if root.exists():shutil.rmtree(root)

def refresh_bootstrap_from_artifacts(predictions:Path,modeling:Path,trajectories:Path,cases_path:Path,scenario_path:Path,output_path:Path,samples=1000,seed=42):
    """Rebuild uncertainty from saved predictions/cases without rerunning any queue."""
    events=load_inputs(predictions,modeling,trajectories,["f1_focused"])
    with duckdb.connect() as db:
        cur=db.execute("select * from read_parquet(?) where threshold_policy='f1_focused' and deduplication_scenario='open_case' and (capacity_scenario like 'rate_%' or capacity_scenario='unlimited')",[str(cases_path)])
        names=[x[0] for x in cur.description];cases=[dict(zip(names,x)) for x in cur.fetchall()]
    with scenario_path.open(newline="",encoding="utf-8-sig") as handle:scenarios=list(csv.DictReader(handle))
    rows=bootstrap_intervals(cases,scenarios,events,samples,seed);root=Path(tempfile.mkdtemp(prefix=".workload_bootstrap_",dir=output_path.parent));staged=root/output_path.name
    try:_write_csv(staged,rows);_publish([(staged,output_path)])
    finally:
        if root.exists():shutil.rmtree(root)
    return len(cases),len(rows)

def append_supplemental_source_inventory(results_path:Path,paths:Sequence[Path]):
    """Atomically record read-only inputs omitted from the initial CLI inventory."""
    report=json.loads(results_path.read_text(encoding="utf-8"));report["supplemental_source_signatures"]=source_signatures(paths)
    report["supplemental_inventory_limitation"]="These unused supporting artifacts were inventoried after simulation; their unchanged timestamps and hashes are recorded, but this is not a pre/post comparison from this run."
    root=Path(tempfile.mkdtemp(prefix=".workload_inventory_",dir=results_path.parent));staged=root/results_path.name
    try:staged.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");_publish([(staged,results_path)])
    finally:
        if root.exists():shutil.rmtree(root)

def correct_capacity_reporting(results_path:Path,scenario_path:Path,checkpoint_path:Path,course_path:Path,bootstrap_path:Path):
    """Correct configured-capacity denominators without changing any review selection."""
    def read(path):
        with path.open(newline="",encoding="utf-8-sig") as handle:return list(csv.DictReader(handle))
    checkpoints=read(checkpoint_path)
    for row in checkpoints:
        reviews=int(float(row["reviews_completed"]));waiting=int(float(row["queue_size_before_review"]));eligible=int(float(row["eligible_students"]));kind=row["capacity_kind"]
        available=waiting if kind=="unlimited" else (math.floor(eligible*float(row["capacity_value"])) if kind=="rate" else int(float(row["capacity_value"])))
        if kind=="rate" and float(row["capacity_value"])>0 and waiting and available<1:available=1
        row["available_review_capacity"]=available;row["unused_capacity"]=None if kind=="unlimited" else max(0,available-reviews);row["capacity_utilization"]=reviews/available if available else None
    scenarios=read(scenario_path);cp_group=_group(checkpoints,lambda x:x["scenario_id"])
    for row in scenarios:
        group=cp_group[row["scenario_id"]];available=sum(int(float(x["available_review_capacity"])) for x in group);reviews=sum(int(float(x["reviews_completed"])) for x in group)
        row["capacity_utilization"]=reviews/available if available else None
    bootstrap=read(bootstrap_path);scenario_lookup={x["scenario_id"]:x for x in scenarios}
    for row in bootstrap:
        if row["metric"]=="capacity_utilization":row["lower_95"]=row["upper_95"]=scenario_lookup[row["scenario_id"]]["capacity_utilization"]
    report=json.loads(results_path.read_text(encoding="utf-8"));report.setdefault("reporting_corrections",[]).append("Configured capacity, unused capacity, and utilization use the nominal course-checkpoint capacity rather than a queue-clipped denominator; review selections did not change.")
    outputs=[results_path,scenario_path,checkpoint_path,course_path,bootstrap_path];root=Path(tempfile.mkdtemp(prefix=".workload_capacity_",dir=results_path.parent));staged=[root/f"{i}_{p.name}" for i,p in enumerate(outputs)]
    try:
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");_write_csv(staged[1],scenarios);_write_csv(staged[2],checkpoints);_write_csv(staged[3],checkpoints);_write_csv(staged[4],bootstrap);_publish(list(zip(staged,outputs)))
    finally:
        if root.exists():shutil.rmtree(root)
