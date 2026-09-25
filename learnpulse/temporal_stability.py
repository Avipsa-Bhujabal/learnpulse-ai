"""Temporal stability analysis for held-out, day-specific LearnPulse warnings."""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, statistics, tempfile
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Sequence
import duckdb, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from learnpulse.calibration_experiment import (
    _publish,
    load_parquet_rows,
    run_calibration_experiment,
)

DAYS=(14,28,42,56)
POLICIES={"f1_focused":"f1_focused_prediction","fixed_0_50":"fixed_prediction",
          "recall_focused":"recall_focused_prediction","alert_capacity":"alert_capacity_prediction"}
STRATEGY_EARLIEST_DAY={"single":14,"two_consecutive":28,"two_of_three":42,"persistent_or_late":28}

def signature(path:Path)->dict:
    h=hashlib.sha256(path.read_bytes()).hexdigest(); s=path.stat()
    return {"path":str(path.resolve()),"size":s.st_size,"mtime_ns":s.st_mtime_ns,"sha256":h}

def sequence(rows,policy="f1_focused"):
    index={int(r["observation_day"]):int(r[POLICIES[policy]]) for r in rows}
    if set(index)!=set(DAYS): raise ValueError("A common trajectory must contain exactly four observation days")
    return "".join(str(index[d]) for d in DAYS)

def trajectory_flags(bits:str,scores:Sequence[float],tolerance=.0):
    warned=[i for i,x in enumerate(bits) if x=="1"]
    first=DAYS[warned[0]] if warned else None
    persistent="11" in bits; temporary=any(bits[i]=="1" and "0" in bits[i+1:] for i in range(3)) and bits.count("1")==1
    recurrent="101" in bits or "1001" in bits or "1011" in bits or "1101" in bits
    alternating=bits in ("1010","0101")
    diffs=np.diff(scores); escalation=bool(len(diffs) and np.all(diffs>tolerance)); deescalation=bool(len(diffs) and np.all(diffs < -tolerance))
    longest=max((len(x) for x in bits.split("0")),default=0); reversals=sum(a!=b for a,b in zip(bits,bits[1:]))
    if bits=="0000": primary="Never warned"
    elif recurrent: primary="Recurrent warning"
    elif persistent: primary="Persistent warning"
    elif temporary: primary="Temporary warning"
    else: primary=f"First warned Day {first}"
    return {"warning_sequence":bits,"first_warning_day":first,"never_warned":not warned,"warning_count":len(warned),
        "longest_warning_run":longest,"persisted_through_day56":bits[-1]=="1" and persistent,"warning_reversals":reversals,
        "primary_category":primary,"persistent_warning":persistent,"temporary_warning":temporary,
        "recurrent_warning":recurrent,"alternating_warning":alternating,"warning_escalation":escalation,"warning_deescalation":deescalation}

def common_and_pairs(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r["module"],r["presentation"],int(r["student_id"]))].append(r)
    common={k:sorted(v,key=lambda x:x["observation_day"]) for k,v in groups.items() if {x["observation_day"] for x in v}==set(DAYS)}
    pairs={}
    for a,b in zip(DAYS,DAYS[1:]):
        pairs[(a,b)]={k:sorted([x for x in v if x["observation_day"] in (a,b)],key=lambda x:x["observation_day"]) for k,v in groups.items() if {a,b}<={x["observation_day"] for x in v}}
    return common,pairs,groups

def transition_rows(pairs,policy="f1_focused",tolerance=.05):
    out=[]
    for (a,b),cohort in pairs.items():
        grouped=defaultdict(list)
        for key,rs in cohort.items():
            left,right=rs; state=f"{left[POLICIES[policy]]}->{right[POLICIES[policy]]}"
            diff=float(right["calibrated_output"])-float(left["calibrated_output"])
            grouped[state].append((key,right,diff))
        total=sum(map(len,grouped.values()))
        for state,items in sorted(grouped.items()):
            later=[x[1] for x in items]; positives=sum(int(x["actual_label"]) for x in later); warnings=sum(int(x[POLICIES[policy]]) for x in later)
            tp=sum(int(x["actual_label"]) and int(x[POLICIES[policy]]) for x in later)
            diffs=np.array([x[2] for x in items]); q=np.quantile(diffs,[.25,.5,.75])
            out.append({"from_day":a,"to_day":b,"transition":state,"count":len(items),"percentage":len(items)/total,
                "later_positive_rate":positives/len(items),"mean_later_score":statistics.fmean(float(x["calibrated_output"]) for x in later),
                "later_precision":tp/warnings if warnings else None,"later_recall_contribution":tp/positives if positives else None,
                "median_score_change":float(q[1]),"score_change_q1":float(q[0]),"score_change_q3":float(q[2]),
                "increasing":float(np.mean(diffs>tolerance)),"decreasing":float(np.mean(diffs < -tolerance)),"stable":float(np.mean(np.abs(diffs)<=tolerance))})
    return out

def strategy_state(bits,index,strategy):
    if DAYS[index] < STRATEGY_EARLIEST_DAY.get(strategy, 10**9):
        raise ValueError(f"{strategy} is not operational on Day {DAYS[index]}")
    if strategy=="single": return bits[index]=="1"
    if strategy=="two_consecutive": return index>=1 and bits[index-1:index+1]=="11"
    if strategy=="two_of_three": return index>=2 and bits[index-2:index+1].count("1")>=2
    if strategy=="persistent_or_late": return (index>=1 and bits[index-1:index+1]=="11") or (index==3 and bits[3]=="1")
    raise ValueError(strategy)

def binary_metrics(labels,pred):
    tp=sum(y and p for y,p in zip(labels,pred)); fp=sum((not y) and p for y,p in zip(labels,pred)); fn=sum(y and not p for y,p in zip(labels,pred)); tn=sum((not y) and not p for y,p in zip(labels,pred))
    rate=lambda n,d:n/d if d else None
    rec=rate(tp,tp+fn); spec=rate(tn,tn+fp); prec=rate(tp,tp+fp)
    return {"warnings":sum(pred),"alert_rate":sum(pred)/len(pred),"true_positives":tp,"false_positives":fp,"false_negatives":fn,"true_negatives":tn,
        "precision":prec,"recall":rec,"specificity":spec,"false_positive_rate":rate(fp,fp+tn),"f1":rate(2*tp,2*tp+fp+fn),"balanced_accuracy":(rec+spec)/2 if rec is not None and spec is not None else None}

def policy_results(common):
    out=[]
    for day_index,day in enumerate(DAYS):
        labels=[int(rs[day_index]["actual_label"]) for rs in common.values()]
        single=[strategy_state(sequence(rs),day_index,"single") for rs in common.values()]
        base=binary_metrics(labels,single)
        for strategy in ("single","two_consecutive","two_of_three","persistent_or_late"):
            if day < STRATEGY_EARLIEST_DAY[strategy]: continue
            pred=[strategy_state(sequence(rs),day_index,strategy) for rs in common.values()]; m=binary_metrics(labels,pred)
            out.append({"observation_day":day,"strategy":strategy,"eligible_rows":len(labels),"positive_cases":sum(labels),**m,
                "warnings_prevented":base["warnings"]-m["warnings"],"false_positives_prevented":base["false_positives"]-m["false_positives"],
                "true_positives_lost":base["true_positives"]-m["true_positives"],"false_negatives_added":m["false_negatives"]-base["false_negatives"],
                "earliest_possible_day":STRATEGY_EARLIEST_DAY[strategy]})
    return out

def stability(common):
    bits=[sequence(rs) for rs in common.values()]; warned=[b for b in bits if "1" in b]
    adjacent=[(b[i],b[i+1]) for b in bits for i in range(3)]
    return {"trajectories":len(bits),"warning_persistence_rate":sum("11" in b for b in warned)/len(warned) if warned else None,
        "warning_reversal_rate":sum(any(a!=c for a,c in zip(b,b[1:])) for b in bits)/len(bits),"recurrent_warning_rate":sum(trajectory_flags(b,[0,0,0,0])["recurrent_warning"] for b in bits)/len(bits),
        "warned_exactly_once":sum(b.count("1")==1 for b in bits)/len(bits),"warned_at_least_twice":sum(b.count("1")>=2 for b in bits)/len(bits),
        "warned_consecutively":sum("11" in b for b in bits)/len(bits),"adjacent_agreement":sum(a==c for a,c in adjacent)/len(adjacent)}

def kappa_spearman(pairs):
    out=[]
    for (a,b),cohort in pairs.items():
        x=[int(rs[0]["f1_focused_prediction"]) for rs in cohort.values()]; y=[int(rs[1]["f1_focused_prediction"]) for rs in cohort.values()]
        sx=[float(rs[0]["calibrated_output"]) for rs in cohort.values()]; sy=[float(rs[1]["calibrated_output"]) for rs in cohort.values()]
        k=None if len(set(x))<2 and len(set(y))<2 else float(cohen_kappa_score(x,y)); rho=float(spearmanr(sx,sy).statistic) if len(set(sx))>1 and len(set(sy))>1 else None
        out.append({"from_day":a,"to_day":b,"cohort_size":len(cohort),"cohens_kappa":k,"spearman_score_correlation":rho})
    return out

def build_extension(modeling:Path,days,only_module=None,only_presentation=None,inner_folds=5,seed=42):
    rows=load_parquet_rows(modeling,days)
    return run_calibration_experiment(rows,days,["sigmoid"],inner_folds,.5,.8,.1,5,1,seed,only_module,only_presentation)

def load_reused(path:Path,days=(28,56),only_module=None,only_presentation=None):
    where=["calibration_method='sigmoid'",f"observation_day in ({','.join('?' for _ in days)})"]; params=[str(path),*days]
    if only_module: where.append("module=?");params.append(only_module)
    if only_presentation: where.append("presentation=?");params.append(only_presentation)
    with duckdb.connect() as c:
        cur=c.execute(f"select * from read_parquet(?) where {' and '.join(where)} order by observation_day,module,presentation,student_id",params)
        cols=[d[0] for d in cur.description];return [dict(zip(cols,x)) for x in cur.fetchall()]

def unified_predictions(modeling:Path,reused,generated):
    rows=[]
    for source,values in (("reused_step13",reused),("generated_temporal_extension",generated)):
        for r in values:
            if r["calibration_method"]!="sigmoid":continue
            x=dict(r);x["calibration_source"]="sigmoid_inner_oof";x["prediction_provenance"]=source;rows.append(x)
    keys=[(r["module"],r["presentation"],int(r["student_id"]),int(r["observation_day"])) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError("Duplicate temporal prediction keys")
    with duckdb.connect() as c:
        meta={(m,p,int(s),int(d)):(int(y),int(a),int(b)) for m,p,s,d,y,a,b in c.execute("select module,presentation,student_id,observation_day,cast(future_inactivity as integer),future_window_start_day,future_window_end_day from read_parquet(?) where observation_day in (14,28,42,56)",[str(modeling)]).fetchall()}
    courses={(k[0],k[1]) for k in keys};meta={k:v for k,v in meta.items() if (k[0],k[1]) in courses}
    if set(keys)!=set(meta):raise ValueError(f"Prediction key coverage mismatch: predictions={len(keys)} expected={len(meta)}")
    for r,key in zip(rows,keys):
        y,a,b=meta[key]
        if int(r["actual_label"])!=y:raise ValueError("Label parity failure")
        r["future_window_start_day"]=a;r["future_window_end_day"]=b
        if r["outer_holdout_id"]!=f"{r['module']}_{r['presentation']}" or int(r["student_overlap_after_removal"]):raise ValueError("Non-held-out temporal prediction")
        for policy,col in POLICIES.items():
            threshold=float(r[policy.replace("fixed_0_50","fixed")+"_threshold"]); expected=int(float(r["calibrated_output"])>=threshold)
            if expected!=int(r[col]):raise ValueError("Threshold reproduction failure")
    return sorted(rows,key=lambda r:(r["module"],r["presentation"],r["student_id"],r["observation_day"]))

def episode_for(days_with_activity,course_end,start=15,length=14):
    active=set(days_with_activity)
    for day in range(start,course_end-length+2):
        if not any(d in active for d in range(day,day+length)):
            return {"episode_eligible":True,"episode_start_day":day,"episode_end_day":day+length-1}
    return {"episode_eligible":course_end>=start+length-1,"episode_start_day":None,"episode_end_day":None}

def attach_episodes(trajectories, daily_dataset:Path, courses:Path):
    lengths={}
    with courses.open(newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):lengths[(r["code_module"],r["code_presentation"])]=int(r["module_presentation_length"])-1
    keys=set(trajectories)
    with duckdb.connect() as c:
        activity=defaultdict(list)
        for m,p,s,d in c.execute("select module,presentation,student_id,day from read_parquet(?,hive_partitioning=true) where day>=15 order by 1,2,3,4",[str(daily_dataset/"**/*.parquet")]).fetchall():
            key=(m,p,int(s))
            if key in keys:activity[key].append(int(d))
    output={}
    for key,rs in trajectories.items():
        ep=episode_for(activity.get(key,[]),lengths.get((key[0],key[1]),-1)); bits=sequence(rs)
        first=next((d for d,ch in zip(DAYS,bits) if ch=="1"),None); start=ep["episode_start_day"]
        prior=next((d for d,ch in zip(DAYS,bits) if ch=="1" and (start is None or d<start)),None)
        ep.update({"first_prior_warning_day":prior,"warning_lead_time":start-prior if start is not None and prior is not None else None,
            "warning_timing":"before_start" if prior is not None else "late" if start is not None and first is not None else "never_warned"})
        output[key]=ep
    return output

def disappearance_activity(pairs,daily_dataset):
    keys=[]
    for (a,b),cohort in pairs.items():
        for key,rs in cohort.items():
            if int(rs[0]["f1_focused_prediction"])==1 and int(rs[1]["f1_focused_prediction"])==0:keys.append((key,a,b,int(rs[1]["f1_focused_prediction"])))
    intervals=defaultdict(list)
    for key,a,b,_ in keys:intervals[key].append((a,b))
    active=defaultdict(lambda:[0,0])
    with duckdb.connect() as c:
        for m,p,s,d,clicks in c.execute("select module,presentation,student_id,day,total_clicks from read_parquet(?,hive_partitioning=true)",[str(daily_dataset/"**/*.parquet")]).fetchall():
            key=(m,p,int(s))
            for a,b in intervals.get(key,()):
                if b<int(d)<=b+14:active[(key,a,b)][0]+=1;active[(key,a,b)][1]+=int(clicks)
    out=[]
    for key,a,b,_ in keys:
        n,clicks=active[(key,a,b)];out.append({"module":key[0],"presentation":key[1],"student_id":key[2],"from_day":a,"to_day":b,
            "activity_days_next_14":n,"clicks_next_14":clicks,"classification":"warning disappeared with subsequent activity" if n else "warning disappeared without subsequent activity"})
    return out

def trajectory_table(common,episodes,tolerance=.05):
    out=[]
    for key,rs in common.items():
        bits=sequence(rs); flags=trajectory_flags(bits,[float(r["calibrated_output"]) for r in rs],tolerance)
        out.append({"module":key[0],"presentation":key[1],"student_id":key[2],**flags,
            **{f"day{r['observation_day']}_score":r["calibrated_output"] for r in rs},
            **{f"day{r['observation_day']}_label":r["actual_label"] for r in rs},**episodes.get(key,{})})
    return out

def bootstrap_summary(common,samples=1000,seed=42):
    vals=list(common.items());rng=np.random.default_rng(seed); metrics=defaultdict(list)
    if not vals:return []
    for _ in range(samples):
        draw=[vals[i] for i in rng.integers(0,len(vals),len(vals))]; cohort={f"{j}_{k}":v for j,(k,v) in enumerate(draw)}
        st=stability(cohort); policies=policy_results(cohort)
        for name in ("warning_persistence_rate","warning_reversal_rate","recurrent_warning_rate"):
            if st[name] is not None: metrics[("stability",None,name)].append(st[name])
        for row in policies:
            if row["observation_day"] in (28,42,56):
                for name in ("precision","recall","f1","false_positives_prevented","true_positives_lost"):
                    if row[name] is not None:metrics[(row["strategy"],row["observation_day"],name)].append(row[name])
    return [{"strategy":k[0],"observation_day":k[1],"metric":k[2],"lower_95":float(np.quantile(v,.025)),"upper_95":float(np.quantile(v,.975)),"valid_samples":len(v),"skipped_samples":samples-len(v)} for k,v in metrics.items() if v]

def _write_csv(path,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def _write_parquet(path,rows):
    tmp=path.with_suffix('.csv');_write_csv(tmp,rows);dest=path.resolve().as_posix().replace("'","''")
    with duckdb.connect() as c:c.execute(f"copy (select * from read_csv_auto(?,header=true)) to '{dest}' (format parquet,compression zstd)",[str(tmp)])
    tmp.unlink()

def plots(transitions,trajectories,policies,figdir):
    # transition heatmaps
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(a,b) in zip(axes,zip(DAYS,DAYS[1:])):
        z=np.zeros((2,2));rows=[r for r in transitions if r["from_day"]==a]
        for r in rows:i,j=map(int,r["transition"].split('->'));z[i,j]=r["percentage"]
        ax.imshow(z,vmin=0,vmax=1,cmap="Blues");ax.set(xticks=[0,1],yticks=[0,1],xlabel=f"Day {b}",ylabel=f"Day {a}",title=f"{a}→{b}")
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{z[i,j]:.1%}",ha="center",va="center")
    fig.suptitle("F1-focused sigmoid warnings · separately fitted day models · descriptive");fig.tight_layout();fig.savefig(figdir/'temporal_warning_transitions.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for cls in sorted({r['primary_category'] for r in trajectories}):
        rs=[r for r in trajectories if r['primary_category']==cls]
        if len(rs)<20:continue
        arr=np.array([[float(r[f'day{d}_score']) for d in DAYS] for r in rs]);med=np.median(arr,axis=0);q1=np.quantile(arr,.25,axis=0);q3=np.quantile(arr,.75,axis=0)
        ax.plot(DAYS,med,marker='o',label=cls);ax.fill_between(DAYS,q1,q3,alpha=.12)
    ax.set(xlabel='Observation day',ylabel='Sigmoid-calibrated output',title='Aggregate score trajectories · separate day models');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/'temporal_score_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,metric in zip(axes,('precision','recall','f1','alert_rate')):
        for strategy in sorted({r['strategy'] for r in policies}):
            rs=[r for r in policies if r['strategy']==strategy];ax.plot([r['observation_day'] for r in rs],[r[metric] for r in rs],marker='o',label=strategy)
        ax.set(title=metric.replace('_',' '),xlabel='Day',ylim=(0,1))
    axes[0].legend(fontsize=7);fig.suptitle('Strategy trade-offs · F1-focused sigmoid source warnings');fig.tight_layout();fig.savefig(figdir/'temporal_policy_tradeoffs.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));labels=['14','28','42','56','Never'];counts=Counter(str(r['first_warning_day']) if r['first_warning_day'] else 'Never' for r in trajectories);ax.bar(labels,[counts[x] for x in labels],color='#376996');ax.set(title='First warning day · common cohort · descriptive',ylabel='Trajectories');fig.tight_layout();fig.savefig(figdir/'first_warning_day.png',dpi=150);plt.close(fig)

def course_results(common,trajectories):
    output=[]
    grouped=defaultdict(dict)
    for key,rs in common.items():grouped[(key[0],key[1])][key]=rs
    lookup={(r['module'],r['presentation'],int(r['student_id'])):r for r in trajectories}
    for (m,p),cohort in sorted(grouped.items()):
        st=stability(cohort);tr=[lookup[k] for k in cohort];episodes=[x for x in tr if x.get('episode_start_day') not in (None,'')];detected=[x for x in episodes if x.get('warning_timing')=='before_start'];leads=[x['warning_lead_time'] for x in detected]
        output.append({'module':m,'presentation':p,'trajectory_cohort_size':len(cohort),**{f'day{d}_warning_prevalence':sum(int(rs[i]['f1_focused_prediction']) for rs in cohort.values())/len(cohort) for i,d in enumerate(DAYS)},
            **st,'first_warning_distribution':dict(Counter(str(x['first_warning_day']) if x['first_warning_day'] else 'Never' for x in tr)),
            'eligible_inactivity_episodes':len(episodes),'episode_detection_rate':len(detected)/len(episodes) if episodes else None,
            'median_lead_time':statistics.median(leads) if leads else None})
    return output

def _as_bool(value):
    """Parse booleans persisted by the CSV reporting layer."""
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}

def _as_optional_int(value):
    return None if value in (None, "", "None", "null") else int(float(value))

def load_trajectory_rows(path:Path):
    """Load the completed trajectory artifact without changing it."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        row["student_id"]=int(row["student_id"])
        for day in DAYS:
            row[f"day{day}_label"]=int(row[f"day{day}_label"])
        row["episode_start_day"]=_as_optional_int(row.get("episode_start_day"))
        row["episode_end_day"]=_as_optional_int(row.get("episode_end_day"))
    return rows

def corrected_episode_summary(trajectories):
    """Summarize mutually exclusive episode outcomes; no-episode alerts stay separate."""
    eligible=[r for r in trajectories if r.get("episode_start_day") is not None]
    before=sum(r.get("warning_timing")=="before_start" for r in eligible)
    after=sum(r.get("warning_timing")=="late" for r in eligible)
    never=len(eligible)-before-after
    no_episode=[r for r in trajectories if r.get("episode_start_day") is None]
    false_alert=sum("1" in str(r["warning_sequence"]) for r in no_episode)
    if before+after+never != len(eligible):
        raise ValueError("Episode categories do not sum to eligible episodes")
    return {"eligible_episode_count":len(eligible),"warned_before_episode_start":before,
        "warned_after_episode_start":after,"eligible_episode_never_warned":never,
        "no_eligible_episode":len(no_episode),"false_alert_trajectory_without_episode":false_alert}

def first_strategy_day(bits, strategy):
    """Return a strategy's first valid alert checkpoint, or None."""
    for index,day in enumerate(DAYS):
        if day < STRATEGY_EARLIEST_DAY[strategy]:
            continue
        if strategy_state(bits,index,strategy):
            return day
    return None

def _summary(values):
    values=[float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:return {"median":None,"mean":None,"q1":None,"q3":None,"minimum":None,"maximum":None}
    q=np.quantile(values,[.25,.5,.75])
    return {"median":float(q[1]),"mean":float(np.mean(values)),"q1":float(q[0]),"q3":float(q[2]),
        "minimum":float(min(values)),"maximum":float(max(values))}

def confirmation_delay_rows(trajectories):
    """Persist confirmation delays relative to each trajectory's first single warning."""
    output=[]
    for strategy in ("two_consecutive","two_of_three","persistent_or_late"):
        delays=[];confirmed=0
        for row in trajectories:
            bits=str(row["warning_sequence"]); first=first_strategy_day(bits,"single"); confirmed_day=first_strategy_day(bits,strategy)
            if confirmed_day is not None:
                confirmed+=1
                if first is not None:delays.append(confirmed_day-first)
        stats=_summary(delays)
        output.append({"strategy":strategy,"confirmed_trajectories":confirmed,
            "earliest_valid_day":STRATEGY_EARLIEST_DAY[strategy],"median_confirmation_delay":stats["median"],
            "mean_confirmation_delay":stats["mean"],"confirmation_delay_q1":stats["q1"],
            "confirmation_delay_q3":stats["q3"],"minimum_confirmation_delay":stats["minimum"],
            "maximum_confirmation_delay":stats["maximum"],"trajectories_never_confirmed":len(trajectories)-confirmed})
    return output

def episode_strategy_rows(trajectories):
    """Compute strategy-specific episode timing without crediting late/negative lead time."""
    first_single={id(r):first_strategy_day(str(r["warning_sequence"]),"single") for r in trajectories}
    output=[]
    single_useful=0
    intermediate={}
    for strategy in STRATEGY_EARLIEST_DAY:
        before=after=never=false_alert=0; leads=[]; delays=[]
        for row in trajectories:
            alert=first_strategy_day(str(row["warning_sequence"]),strategy); start=row.get("episode_start_day")
            if start is None:
                false_alert += alert is not None
                continue
            if alert is None:never+=1
            elif alert < start:
                before+=1;leads.append(start-alert)
                base=first_single[id(row)]
                if base is not None:delays.append(alert-base)
            else:
                after+=1  # deliberately no useful lead time for late warnings
        if strategy=="single":single_useful=before
        intermediate[strategy]=(before,after,never,false_alert,leads,delays)
    eligible=sum(r.get("episode_start_day") is not None for r in trajectories); no_episode=len(trajectories)-eligible
    for strategy,(before,after,never,false_alert,leads,delays) in intermediate.items():
        if before+after+never != eligible:raise ValueError("Strategy episode categories do not sum")
        stats=_summary(leads); delay=_summary(delays)
        output.append({"strategy":strategy,"eligible_episodes":eligible,"warned_before_episode_start":before,
            "warned_after_episode_start":after,"never_warned":never,"no_eligible_episode":no_episode,
            "false_alert_trajectories_without_episode":false_alert,"median_lead_time":stats["median"],
            "mean_lead_time":stats["mean"],"lead_time_q1":stats["q1"],"lead_time_q3":stats["q3"],
            "minimum_nonnegative_lead_time":stats["minimum"],"maximum_nonnegative_lead_time":stats["maximum"],
            "useful_warnings_lost_relative_to_single":single_useful-before,
            "median_delay_relative_to_single":delay["median"]})
    return output

def course_strategy_results(trajectories):
    """Return day-aligned policy metrics for every course presentation."""
    grouped=defaultdict(list)
    for row in trajectories:grouped[(row["module"],row["presentation"])].append(row)
    output=[]
    for (module,presentation),rows in sorted(grouped.items()):
        for index,day in enumerate(DAYS):
            labels=[int(r[f"day{day}_label"]) for r in rows]
            single=[strategy_state(str(r["warning_sequence"]),index,"single") for r in rows]
            baseline=binary_metrics(labels,single)
            for strategy,earliest in STRATEGY_EARLIEST_DAY.items():
                if day<earliest:continue
                pred=[strategy_state(str(r["warning_sequence"]),index,strategy) for r in rows]; metrics=binary_metrics(labels,pred)
                output.append({"module":module,"presentation":presentation,"observation_day":day,"strategy":strategy,
                    "rows":len(rows),"positive_cases":sum(labels),**metrics,
                    "false_positives_prevented":baseline["false_positives"]-metrics["false_positives"],
                    "true_positives_lost":baseline["true_positives"]-metrics["true_positives"]})
    return output

def course_macro_summary(course_rows):
    """Macro summarize presentations, omitting undefined values rather than treating them as zero."""
    metrics=("alert_rate","precision","recall","f1","false_positive_rate","false_positives_prevented","true_positives_lost")
    grouped=defaultdict(list)
    for row in course_rows:grouped[(row["observation_day"],row["strategy"])].append(row)
    out=[]
    for (day,strategy),rows in sorted(grouped.items()):
        for metric in metrics:
            vals=[float(r[metric]) for r in rows if r.get(metric) not in (None,"")]
            out.append({"observation_day":day,"strategy":strategy,"metric":metric,
                "mean":statistics.fmean(vals) if vals else None,"sample_standard_deviation":statistics.stdev(vals) if len(vals)>1 else None,
                "median":statistics.median(vals) if vals else None,"contributing_presentations":len(vals)})
    return out

def _weighted_median(values,weights):
    mask=np.isfinite(values)&(weights>0)
    if not np.any(mask):return None
    values=values[mask];weights=weights[mask];order=np.argsort(values);values=values[order];weights=weights[order]
    return float(values[np.searchsorted(np.cumsum(weights),weights.sum()/2,side="left")])

def reporting_bootstrap(trajectories,samples=1000,seed=42):
    """Bootstrap reporting estimates at trajectory and strict cross-course student levels."""
    n=len(trajectories); rng=np.random.default_rng(seed); bits=[str(r["warning_sequence"]) for r in trajectories]
    first=np.array([first_strategy_day(b,"single") or 0 for b in bits]); episode=np.array([float(r["episode_start_day"]) if r.get("episode_start_day") is not None else np.nan for r in trajectories])
    alerts={s:np.array([first_strategy_day(b,s) or 0 for b in bits]) for s in STRATEGY_EARLIEST_DAY}
    leads={s:np.where((alerts[s]>0)&(alerts[s]<episode),episode-alerts[s],np.nan) for s in alerts}
    delays={s:np.where((alerts[s]>0)&(first>0),alerts[s]-first,np.nan) for s in ("two_consecutive","two_of_three","persistent_or_late")}
    value_orders={};
    for name,values in [(f"lead:{s}",v) for s,v in leads.items()]+[(f"delay:{s}",v) for s,v in delays.items()]:
        valid=np.flatnonzero(np.isfinite(values));value_orders[name]=(valid[np.argsort(values[valid])],values)
    store=defaultdict(list)
    student_values=sorted({r["student_id"] for r in trajectories});student_index={x:i for i,x in enumerate(student_values)}
    trajectory_student=np.array([student_index[r["student_id"]] for r in trajectories])
    labels={d:np.array([int(r[f"day{d}_label"]) for r in trajectories]) for d in DAYS}
    predictions={(s,d):np.array([int(strategy_state(bits[i],DAYS.index(d),s)) for i in range(n)]) for s in STRATEGY_EARLIEST_DAY for d in DAYS if d>=STRATEGY_EARLIEST_DAY[s]}
    warned=first>0;persistent=np.array(["11" in b for b in bits]);reversal=np.array([any(a!=c for a,c in zip(b,b[1:])) for b in bits])
    recurrent=np.array(["101" in b or "1001" in b or "1011" in b or "1101" in b for b in bits])
    def collect(scope,weights):
        def ordered_median(name):
            order,values=value_orders[name];ordered_weights=weights[order];total=ordered_weights.sum()
            return None if total==0 else float(values[order[np.searchsorted(np.cumsum(ordered_weights),total/2,side="left")]])
        denom=weights.sum()
        for day in (*DAYS,0):
            name=f"first_warning_day_{day}" if day else "never_warned"
            target=(first==day) if day else (first==0);store[(scope,None,None,name)].append(float(np.dot(weights,target)/denom))
        for strategy in ("two_consecutive","two_of_three","persistent_or_late"):
            value=ordered_median(f"delay:{strategy}")
            if value is not None:store[(scope,strategy,None,"confirmation_delay")].append(value)
        for strategy in STRATEGY_EARLIEST_DAY:
            value=ordered_median(f"lead:{strategy}")
            if value is not None:store[(scope,strategy,None,"median_episode_lead_time")].append(value)
        if scope=="strict_student_id":
            if np.dot(weights,warned):store[(scope,None,None,"persistence")].append(float(np.dot(weights,warned&persistent)/np.dot(weights,warned)))
            store[(scope,None,None,"reversal")].append(float(np.dot(weights,reversal)/denom));store[(scope,None,None,"recurrence")].append(float(np.dot(weights,recurrent)/denom))
            for (strategy,day),pred in predictions.items():
                y=labels[day];tp=np.dot(weights,(pred==1)&(y==1));fp=np.dot(weights,(pred==1)&(y==0));fn=np.dot(weights,(pred==0)&(y==1))
                base=predictions[("single",day)];base_fp=np.dot(weights,(base==1)&(y==0));base_tp=np.dot(weights,(base==1)&(y==1))
                for metric,value in (("precision",tp/(tp+fp) if tp+fp else None),("recall",tp/(tp+fn) if tp+fn else None),("f1",2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None),("false_positive_reduction",base_fp-fp),("true_positive_loss",base_tp-tp)):
                    if value is not None:store[(scope,strategy,day,metric)].append(float(value))
    for _ in range(samples):
        collect("student_course_trajectory",np.bincount(rng.integers(0,n,n),minlength=n))
        id_weights=np.bincount(rng.integers(0,len(student_values),len(student_values)),minlength=len(student_values));collect("strict_student_id",id_weights[trajectory_student])
    out=[]
    for key,values in sorted(store.items(),key=lambda x:str(x[0])):
        out.append({"bootstrap_unit":key[0],"strategy":key[1],"observation_day":key[2],"metric":key[3],
            "lower_95":float(np.quantile(values,.025)),"upper_95":float(np.quantile(values,.975)),
            "valid_samples":len(values),"skipped_samples":samples-len(values)})
    return out

def recursive_parquet_manifest(directory:Path):
    """Create a deterministic prospective integrity baseline for a Parquet directory."""
    files=[]
    for path in sorted(directory.rglob("*.parquet"),key=lambda p:p.relative_to(directory).as_posix()):
        stat=path.stat();digest=hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
        files.append({"relative_path":path.relative_to(directory).as_posix(),"size":stat.st_size,
            "modification_timestamp_ns":stat.st_mtime_ns,"sha256":digest.hexdigest()})
    return {"directory":str(directory.resolve()),"baseline_type":"prospective_from_step15_reporting_correction",
        "limitation":"No pre-run recursive hash was saved, so this cannot prove byte-for-byte equality with the directory used during the original Step 15 run.","files":files}

def plot_policy_tradeoffs(policies,path:Path):
    """Render the corrected trade-off chart from saved/derived policy metrics only."""
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,metric in zip(axes,("precision","recall","f1","alert_rate")):
        for strategy in sorted({r["strategy"] for r in policies}):
            rows=sorted((r for r in policies if r["strategy"]==strategy),key=lambda r:r["observation_day"])
            ax.plot([r["observation_day"] for r in rows],[r[metric] for r in rows],marker="o",label=strategy.replace("_"," "))
        ax.set(title=metric.replace("_"," "),xlabel="Observation day",ylim=(0,1),xticks=DAYS)
    axes[0].legend(fontsize=7);fig.suptitle("Strategy trade-offs · F1-focused sigmoid source warnings · valid days only")
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def pooled_policy_results(course_rows):
    """Pool course confusion counts into corrected day/strategy policy results."""
    grouped=defaultdict(list)
    for row in course_rows:grouped[(row["observation_day"],row["strategy"])].append(row)
    output=[]
    for (day,strategy),rows in sorted(grouped.items()):
        tp=sum(r["true_positives"] for r in rows);fp=sum(r["false_positives"] for r in rows)
        fn=sum(r["false_negatives"] for r in rows);tn=sum(r["true_negatives"] for r in rows);total=tp+fp+fn+tn
        rate=lambda n,d:n/d if d else None
        single=grouped[(day,"single")];single_fp=sum(r["false_positives"] for r in single);single_tp=sum(r["true_positives"] for r in single)
        output.append({"observation_day":day,"strategy":strategy,"eligible_rows":total,"positive_cases":tp+fn,
            "warnings":tp+fp,"alert_rate":rate(tp+fp,total),"true_positives":tp,"false_positives":fp,
            "false_negatives":fn,"true_negatives":tn,"precision":rate(tp,tp+fp),"recall":rate(tp,tp+fn),
            "specificity":rate(tn,tn+fp),"false_positive_rate":rate(fp,fp+tn),"f1":rate(2*tp,2*tp+fp+fn),
            "balanced_accuracy":((rate(tp,tp+fn)+rate(tn,tn+fp))/2 if rate(tp,tp+fn) is not None and rate(tn,tn+fp) is not None else None),
            "warnings_prevented":sum(r["warnings"] for r in single)-(tp+fp),
            "false_positives_prevented":single_fp-fp,"true_positives_lost":single_tp-tp,
            "false_negatives_added":fn-sum(r["false_negatives"] for r in single),
            "earliest_possible_day":STRATEGY_EARLIEST_DAY[strategy]})
    return output

def refresh_reporting_artifacts(*,results_path:Path,trajectories_path:Path,daily_dataset:Path,
        policy_path:Path,course_path:Path,chart_path:Path,confirmation_path:Path,
        episode_strategy_path:Path,bootstrap_path:Path,macro_path:Path,manifest_path:Path,
        bootstrap_samples:int=1000,seed:int=42):
    """Atomically refresh derived Step 15 reports without fitting or changing predictions."""
    trajectories=load_trajectory_rows(trajectories_path)
    episode=corrected_episode_summary(trajectories);confirmations=confirmation_delay_rows(trajectories)
    episode_strategies=episode_strategy_rows(trajectories);courses=course_strategy_results(trajectories)
    macro=course_macro_summary(courses);policies=pooled_policy_results(courses)
    if any(r["strategy"]=="persistent_or_late" and r["observation_day"]<28 for r in policies):
        raise ValueError("persistent_or_late cannot be emitted before Day 28")
    bootstrap=reporting_bootstrap(trajectories,bootstrap_samples,seed);manifest=recursive_parquet_manifest(daily_dataset)
    report=json.loads(results_path.read_text(encoding="utf-8"));report["episodes"]=episode
    report["confirmation_delays"]=confirmations;report["episode_strategy_results"]=episode_strategies
    report["reporting_only_correction"]={"models_refitted":False,"scores_recalibrated":False,
        "thresholds_reselected":False,"temporal_predictions_regenerated":False,
        "bootstrap_samples":bootstrap_samples,"random_seed":seed,
        "daily_activity_manifest":str(manifest_path)}
    outputs=[results_path,policy_path,course_path,chart_path,confirmation_path,episode_strategy_path,bootstrap_path,macro_path,manifest_path]
    root=Path(tempfile.mkdtemp(prefix=".temporal_reporting_",dir=results_path.parent));staged=[root/f"{i}_{p.name}" for i,p in enumerate(outputs)]
    try:
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
        _write_csv(staged[1],policies);_write_csv(staged[2],courses);plot_policy_tradeoffs(policies,staged[3])
        _write_csv(staged[4],confirmations);_write_csv(staged[5],episode_strategies);_write_csv(staged[6],bootstrap);_write_csv(staged[7],macro)
        staged[8].write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        _publish(list(zip(staged,outputs)))
    finally:
        if root.exists():shutil.rmtree(root)
    return {"episodes":episode,"confirmation_delays":confirmations,"episode_strategy_results":episode_strategies,
        "policies":policies,"course_rows":courses,"macro_rows":macro,"bootstrap_rows":bootstrap,"manifest_files":len(manifest["files"])}

def example_rows(trajectories):
    categories={'consistently_low':lambda r:max(float(r[f'day{d}_score']) for d in DAYS)<.25,
        'consistently_high':lambda r:min(float(r[f'day{d}_score']) for d in DAYS)>.5,
        'low_to_high':lambda r:float(r['day56_score'])-float(r['day14_score'])>.3,
        'high_to_low':lambda r:float(r['day14_score'])-float(r['day56_score'])>.3,
        'temporary_warning':lambda r:r['temporary_warning'],'persistent_warning':lambda r:r['persistent_warning'],
        'recurrent_warning':lambda r:r['recurrent_warning'],'early_true_warning':lambda r:r['first_warning_day']==14 and int(r['day14_label'])==1,
        'early_false_warning':lambda r:r['first_warning_day']==14 and int(r['day14_label'])==0,
        'late_or_missed_episode':lambda r:r.get('episode_start_day') is not None and r.get('warning_timing')!='before_start'}
    out=[]
    for name,predicate in categories.items():
        eligible=[r for r in trajectories if predicate(r)]
        if not eligible:continue
        median=statistics.median(statistics.fmean(float(r[f'day{d}_score']) for d in DAYS) for r in eligible)
        r=min(eligible,key=lambda x:abs(statistics.fmean(float(x[f'day{d}_score']) for d in DAYS)-median))
        out.append({'example_category':name,**r})
    return out

def publish_outputs(rows,trajectories,transitions,policies,courses,examples,report,targets):
    outputs=[targets[k] for k in ('results','predictions','trajectories','transitions','policy_results','course_results','examples')]
    figures=[targets['figures_dir']/x for x in ('temporal_warning_transitions.png','temporal_score_trajectories.png','temporal_policy_tradeoffs.png','first_warning_day.png')];outputs+=figures
    root=Path(tempfile.mkdtemp(prefix='.temporal_build_',dir=targets['results'].parent));staged=[root/f'{i}_{p.name}' for i,p in enumerate(outputs)]
    try:
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8');_write_parquet(staged[1],rows)
        for path,data in zip(staged[2:7],(trajectories,transitions,policies,courses,examples)):_write_csv(path,data)
        figdir=root/'figs';figdir.mkdir();plots(transitions,trajectories,policies,figdir)
        for src,dst in zip([figdir/x.name for x in figures],staged[7:]):os.replace(src,dst)
        _publish(list(zip(staged,outputs)))
    finally:
        if root.exists():shutil.rmtree(root)
