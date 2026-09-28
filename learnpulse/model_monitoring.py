"""Transparent schema, feature, score, and delayed-label monitoring."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
 average_precision_score,
 brier_score_loss,
 confusion_matrix,
 f1_score,
 precision_score,
 recall_score,
 roc_auc_score,
)

from learnpulse.reference_model import FEATURES, validate_record


def expected_calibration_error(labels,scores,bins=10):
 y=np.asarray(labels,dtype=float);s=np.asarray(scores,dtype=float)
 if not len(y):return None
 indices=np.minimum((np.clip(s,0,1)*bins).astype(int),bins-1);value=0.0
 for i in range(bins):
  mask=indices==i
  if np.any(mask):value+=float(np.mean(mask))*abs(float(np.mean(s[mask]))-float(np.mean(y[mask])))
 return float(value)

def psi(reference_counts,current_counts,epsilon=1e-6):
 r=np.array(reference_counts,dtype=float);c=np.array(current_counts,dtype=float);r=r/r.sum();c=c/c.sum();return float(np.sum((c-r)*np.log((c+epsilon)/(r+epsilon))))

def monitor(records,scores,reference,labels=None,aligned=False,threshold=.5):
 schema_errors=[]
 for i,row in enumerate(records):
  try:validate_record(row)
  except ValueError as exc:schema_errors.append({"index":i,"error":str(exc)})
 feature=[];levels=[]
 for name in FEATURES:
  vals=np.array([np.nan if r.get(name) is None else float(r[name]) for r in records]);valid=vals[np.isfinite(vals)];ref=reference["features"][name]
  smd=(float(np.mean(valid))-ref["mean"])/ref["std"] if len(valid) and ref["std"] else None;median_shift=float(np.median(valid)-ref["median"]) if len(valid) else None;missing=float(np.mean(~np.isfinite(vals)));out=float(np.mean((valid<ref["min"])|(valid>ref["max"]))) if len(valid) else None
  level="critical" if smd is not None and abs(smd)>=1 else "warning" if smd is not None and abs(smd)>=.5 else "informational";levels.append(level);feature.append({"feature":name,"missing_rate":missing,"reference_missing_rate":ref["missing_rate"],"missing_rate_difference":missing-ref["missing_rate"],"standardized_mean_difference":smd,"median_shift":median_shift,"out_of_range_rate":out,"level":level})
 scores=np.array(scores,dtype=float);counts=np.histogram(scores,bins=np.linspace(0,1,11))[0];score={"mean":float(np.mean(scores)),"median":float(np.median(scores)),"std":float(np.std(scores)),"quantiles":{str(q):float(np.quantile(scores,q)) for q in (.1,.25,.5,.75,.9)},"histogram":counts.tolist(),"psi":psi(reference["scores"]["histogram"]["counts"],counts),"share_above_threshold":float(np.mean(scores>=threshold))}
 performance=None
 if labels is not None:
  if not aligned:raise ValueError("Delayed labels require exact outcome-window alignment")
  y=np.array(labels,dtype=int);p=(scores>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();performance={"prevalence":float(y.mean()),"pr_auc":float(average_precision_score(y,scores)),"roc_auc":float(roc_auc_score(y,scores)),"brier":float(brier_score_loss(y,scores)),"expected_calibration_error":expected_calibration_error(y,scores),"precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],"alert_rate":float(p.mean())}
 return {"synthetic":False,"rows":len(records),"schema_errors":schema_errors,"feature_metrics":feature,"score_metrics":score,"delayed_outcome_performance":performance,"performance_calculated":performance is not None,"alert_counts":dict((x,levels.count(x)) for x in ("informational","warning","critical")),"research_only":True}
