"""Build and use the research-only full-history reference inference artifact."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import joblib
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 average_precision_score,
 brier_score_loss,
 log_loss,
 roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.calibration_experiment import select_thresholds

FEATURES=tuple(PERSONAL_CHANGE_FEATURE_COLUMNS)
SCHEMA_VERSION="reference-personal-change-v1"
FORBIDDEN={"module","presentation","student_id","observation_day","future_inactivity","future_activity_days","future_clicks","future_window_start_day","future_window_end_day","official_withdrawal_day","officially_withdrawn_by_observation_day","officially_withdrawn_by_window_end","final_result","prediction_eligible","outcome_eligible","exclusion_reason","source_partition","dataset_schema_version","gender","region","highest_education","imd_band","age_band","disability","num_of_prev_attempts","studied_credits"}
NONNEGATIVE={"historical_windows_available","historical_average_clicks","historical_standard_deviation_clicks","historical_average_active_days","current_inactivity_gap","longest_previous_inactivity_gap"}
NULLABLE={"historical_standard_deviation_clicks","click_percentage_change_from_personal_average","click_z_score","current_inactivity_gap","longest_previous_inactivity_gap"}

def sha256(path:Path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def logical_fingerprint(dataset_signature,version,seed,folds):
 payload={"artifact":"research_reference_model","version":version,"schema":SCHEMA_VERSION,"features":FEATURES,"seed":seed,"folds":folds,"dataset":dataset_signature,"pipeline":"median+indicator|standard-scaler|balanced-logistic|max_iter=1000|oof-sigmoid"}
 return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def base_pipeline(seed=42):
 return Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",StandardScaler()),("classifier",LogisticRegression(class_weight="balanced",random_state=seed,max_iter=1000))])

def grouped_folds(y,groups,folds=5,seed=42):
 for count in range(min(folds,len(set(groups))),1,-1):
  split=list(StratifiedGroupKFold(count,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
  if all(len(set(y[i] for i in tr))==2 and len(set(y[i] for i in va))==2 for tr,va in split):return split,count
 raise ValueError("Cannot construct grouped folds with both classes")

def load_training(path:Path):
 cols=",".join([*FEATURES,"future_inactivity","student_id","module","presentation","observation_day"])
 with duckdb.connect() as c:
  cur=c.execute(f"select {cols} from read_parquet(?) order by module,presentation,student_id,observation_day",[str(path)]);names=[x[0] for x in cur.description];rows=[dict(zip(names,x)) for x in cur.fetchall()]
 X=np.array([[np.nan if r[f] is None else float(r[f]) for f in FEATURES] for r in rows]);y=np.array([int(r["future_inactivity"]) for r in rows]);groups=np.array([int(r["student_id"]) for r in rows])
 return rows,X,y,groups

def train_reference(rows,X,y,groups,folds=5,seed=42):
 splits,count=grouped_folds(y,groups,folds,seed);oof=np.full(len(y),np.nan);fold_info=[]
 for number,(train,valid) in enumerate(splits):
  pipe=base_pipeline(seed);pipe.fit(X[train],y[train]);oof[valid]=pipe.decision_function(X[valid]);fold_info.append({"fold":number,"training_rows":len(train),"validation_rows":len(valid),"training_positives":int(y[train].sum()),"validation_positives":int(y[valid].sum()),"training_students":len(set(groups[train])),"validation_students":len(set(groups[valid])),"student_overlap":len(set(groups[train])&set(groups[valid])),"iterations":int(pipe.named_steps["classifier"].n_iter_[0])})
 if np.isnan(oof).any():raise ValueError("Every row must receive exactly one OOF score")
 calibrator=LogisticRegression(random_state=seed);calibrator.fit(oof.reshape(-1,1),y);calibrated=calibrator.predict_proba(oof.reshape(-1,1))[:,1]
 final=base_pipeline(seed);final.fit(X,y);thresholds=select_thresholds(calibrated,y,.5,.8,.1,5,1)
 policies={name:float(value["threshold"]) for name,value in thresholds.items()}
 diagnostics={"diagnostic_scope":"grouped OOF internal training diagnostics; not untouched-test performance","fold_count":count,"folds":fold_info,"oof_rows":len(oof),"oof_complete":True,"prevalence":float(y.mean()),"oof_brier":float(brier_score_loss(y,calibrated)),"oof_log_loss":float(log_loss(y,calibrated)),"oof_pr_auc":float(average_precision_score(y,calibrated)),"oof_roc_auc":float(roc_auc_score(y,calibrated)),"final_iterations":int(final.named_steps["classifier"].n_iter_[0]),"final_converged":bool(final.named_steps["classifier"].n_iter_[0]<1000),"threshold_policies":policies}
 return {"pipeline":final,"calibrator":calibrator,"feature_names":FEATURES,"schema_version":SCHEMA_VERSION,"threshold_policies":policies,"research_only":True},diagnostics,calibrated

def validate_record(record,allow_reference=True):
 unknown=set(record)-set(FEATURES)-({"record_reference"} if allow_reference else set())
 forbidden=set(record)&FORBIDDEN
 if forbidden:raise ValueError(f"Forbidden fields: {sorted(forbidden)}")
 if unknown:raise ValueError(f"Unknown fields: {sorted(unknown)}")
 missing=set(FEATURES)-set(record)
 if missing:raise ValueError(f"Missing fields: {sorted(missing)}")
 values=[]
 for name in FEATURES:
  value=record[name]
  if value is None:
   if name not in NULLABLE:raise ValueError(f"{name} may not be null")
   values.append(np.nan);continue
  if isinstance(value,bool) or not isinstance(value,(int,float)):raise ValueError(f"{name} must be numeric")
  if not math.isfinite(float(value)):raise ValueError(f"{name} must be finite")
  if name in NONNEGATIVE and value<0:raise ValueError(f"{name} must be nonnegative")
  if name=="historical_windows_available" and int(value)!=value:raise ValueError(f"{name} must be an integer")
  if name=="historical_average_active_days" and not 0<=value<=7:raise ValueError(f"{name} must be between 0 and 7")
  values.append(float(value))
 return values

def score_records(bundle,records):
 X=np.array([validate_record(x) for x in records]);raw=bundle["pipeline"].decision_function(X);return bundle["calibrator"].predict_proba(raw.reshape(-1,1))[:,1]

def build_reference(modeling_table:Path,artifact_dir:Path,folds=5,seed=42,version="1.0.0"):
 artifact_dir.parent.mkdir(parents=True,exist_ok=True)
 started=time.time();dataset={"path":str(modeling_table.resolve()),"size":modeling_table.stat().st_size,"mtime_ns":modeling_table.stat().st_mtime_ns,"sha256":sha256(modeling_table)};fingerprint=logical_fingerprint(dataset,version,seed,folds)
 rows,X,y,groups=load_training(modeling_table);bundle,diagnostics,reference_scores=train_reference(rows,X,y,groups,folds,seed)
 profile={"features":{},"scores":{"mean":float(np.mean(reference_scores)),"median":float(np.median(reference_scores)),"std":float(np.std(reference_scores)),"quantiles":{str(q):float(np.quantile(reference_scores,q)) for q in (0,.1,.25,.5,.75,.9,1)},"histogram":{"counts":np.histogram(reference_scores,bins=np.linspace(0,1,11))[0].tolist(),"edges":np.linspace(0,1,11).tolist()}}}
 for i,name in enumerate(FEATURES):
  v=X[:,i];valid=v[np.isfinite(v)];profile["features"][name]={"missing_rate":float(np.mean(~np.isfinite(v))),"mean":float(np.mean(valid)) if len(valid) else None,"std":float(np.std(valid)) if len(valid) else None,"median":float(np.median(valid)) if len(valid) else None,"q1":float(np.quantile(valid,.25)) if len(valid) else None,"q3":float(np.quantile(valid,.75)) if len(valid) else None,"min":float(np.min(valid)) if len(valid) else None,"max":float(np.max(valid)) if len(valid) else None,"histogram":{"counts":np.histogram(valid,bins=10)[0].tolist(),"edges":np.histogram(valid,bins=10)[1].tolist()} if len(valid) else None}
 schema={"schema_version":SCHEMA_VERSION,"features":[{"name":x,"type":"number","nullable":x in NULLABLE,"constraints":{"nonnegative":x in NONNEGATIVE}} for x in FEATURES],"forbidden_fields":sorted(FORBIDDEN),"unknown_fields":"rejected"}
 root=Path(tempfile.mkdtemp(prefix=".reference_build_",dir=artifact_dir.parent));
 try:
  joblib.dump(bundle,root/"model.joblib");(root/"feature_schema.json").write_text(json.dumps(schema,indent=2)+"\n");(root/"reference_distributions.json").write_text(json.dumps(profile,indent=2)+"\n");(root/"training_diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
  card="# LearnPulse research reference model\n\n**Research-only artifact. Not approved for institutional deployment or automatic learner decisions.**\n\nPredicts future recorded OULAD VLE inactivity. It does not diagnose disengagement, motivation, failure, or intervention need. Full historical training means this artifact provides no new external-performance evidence. Use existing held-out experiments for scientific claims.\n"
  (root/"MODEL_CARD.md").write_text(card)
  hashes={p.name:sha256(p) for p in root.iterdir() if p.is_file()}
  try:commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
  except Exception:commit=None
  manifest={"artifact_name":"research_reference_model","model_version":version,"artifact_type":"research_reference_model","created_at_utc":datetime.now(timezone.utc).isoformat(),"git_commit":commit,"python_version":platform.python_version(),"dependency_versions":{"scikit_learn":sklearn.__version__,"duckdb":duckdb.__version__,"numpy":np.__version__,"joblib":joblib.__version__},"random_seed":seed,"feature_names":list(FEATURES),"schema_version":SCHEMA_VERSION,"training_dataset_signature":dataset,"training_rows":len(rows),"unique_students":len(set(groups)),"observation_days":sorted({r["observation_day"] for r in rows}),"modules":sorted({r["module"] for r in rows}),"presentations":sorted({f'{r["module"]}_{r["presentation"]}' for r in rows}),"target_definition":"No recorded OULAD VLE activity during days t+1 through t+14","positive_count":int(y.sum()),"class_prevalence":float(y.mean()),"fold_count":diagnostics["fold_count"],"preprocessing":"SimpleImputer(median, add_indicator=True) -> StandardScaler","classifier":{"type":"LogisticRegression","class_weight":"balanced","random_state":seed,"max_iter":1000},"calibration_method":"sigmoid fitted on grouped OOF decision scores","convergence_status":diagnostics["final_converged"],"training_code_version":"step17-v1","artifact_fingerprint":fingerprint,"threshold_policies":diagnostics["threshold_policies"],"file_hashes":hashes,"intended_use":["local reproducible scoring architecture","shadow-mode research","synthetic demonstrations"],"prohibited_uses":["new external performance claims","automatic interventions","psychological diagnosis","punitive decisions","institutional deployment without governance"],"known_limitations":["trained on all eligible historical rows","scores are not guaranteed individual risk percentages","OULAD inactivity is a behavioral proxy"],"build_seconds":time.time()-started}
  (root/"model_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
  backup=None
  if artifact_dir.exists():backup=artifact_dir.with_name(artifact_dir.name+".backup");shutil.move(artifact_dir,backup)
  try:os.replace(root,artifact_dir);root=None
  except Exception:
   if backup and backup.exists():shutil.move(backup,artifact_dir)
   raise
  if backup and backup.exists():shutil.rmtree(backup)
  return manifest
 finally:
  if root and root.exists():shutil.rmtree(root)
