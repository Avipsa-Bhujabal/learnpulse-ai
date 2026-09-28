"""Local FastAPI scoring service for the research reference model."""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from learnpulse.artifact_registry import validate_artifact
from learnpulse.reference_model import FEATURES, score_records, validate_record

log=logging.getLogger("learnpulse.scoring")
METRICS={"requests":0,"validation_failures":0,"scoring_failures":0,"artifact_load_failures":0,"scored_records":0,"latency_seconds":[],"batch_sizes":[],"missing_features":0,"scores":[],"drift_alerts":0}

def _safe_log(event,status,manifest=None,batch=0,latency=0,error=None,correlation=None):
 log.info(json.dumps({"timestamp":time.time(),"event_type":event,"request_status":status,"model_version":manifest.get("model_version") if manifest else None,"artifact_fingerprint":manifest.get("artifact_fingerprint") if manifest else None,"batch_size":batch,"latency_ms":round(latency*1000,3),"error_category":error,"correlation_id":correlation}))

def load_artifact(path:Path):
 manifest=validate_artifact(path)
 try:bundle=joblib.load(path/"model.joblib")
 except Exception as exc:raise ValueError("Model artifact cannot be loaded") from exc
 if tuple(bundle.get("feature_names",()))!=FEATURES:raise ValueError("Incompatible packaged feature schema")
 return manifest,bundle

def create_app(artifact_dir:Path|None=None,max_batch=1000,max_request_bytes=1_000_000):
 app=FastAPI(title="LearnPulse Research Scoring Service",version="1.0.0",description="Research only; no automatic learner decisions.")
 path=artifact_dir or Path(os.getenv("LEARNPULSE_ARTIFACT_DIR","artifacts/reference_model"));state={"manifest":None,"bundle":None,"error":None}
 try:state["manifest"],state["bundle"]=load_artifact(path)
 except Exception as exc:state["error"]=str(exc);METRICS["artifact_load_failures"]+=1
 @app.middleware("http")
 async def controls(request:Request,call_next):
  started=time.perf_counter();METRICS["requests"]+=1;correlation=request.headers.get("x-correlation-id") or str(uuid.uuid4())
  if request.method=="POST" and request.headers.get("content-type","").split(";")[0]!="application/json":return Response(json.dumps({"detail":"Content-Type must be application/json","correlation_id":correlation}),415,media_type="application/json")
  if int(request.headers.get("content-length") or 0)>max_request_bytes:return Response(json.dumps({"detail":"Request too large","correlation_id":correlation}),413,media_type="application/json")
  try:response=await call_next(request)
  except Exception:
   METRICS["scoring_failures"]+=1;response=Response(json.dumps({"detail":"Controlled internal error","correlation_id":correlation}),500,media_type="application/json")
  elapsed=time.perf_counter()-started;METRICS["latency_seconds"].append(elapsed);response.headers["x-correlation-id"]=correlation;_safe_log(request.url.path,response.status_code,state["manifest"],latency=elapsed,error=None if response.status_code<400 else "request_error",correlation=correlation);return response
 @app.get("/health")
 def health():return {"status":"healthy","model_ready":False,"research_only":True}
 @app.get("/ready")
 def ready(response:Response):
  try:
   manifest,bundle=load_artifact(path);score=score_records(bundle,[synthetic_record()])[0]
   if not math.isfinite(score) or not 0<=score<=1:raise ValueError("Smoke score invalid")
   return {"ready":True,"model_version":manifest["model_version"],"research_only":True}
  except Exception as exc:response.status_code=503;return {"ready":False,"reason":str(exc),"research_only":True}
 @app.get("/version")
 def version():
  if not state["manifest"]:raise HTTPException(503,"Artifact unavailable")
  m=state["manifest"];return {"service_version":"1.0.0","model_version":m["model_version"],"artifact_fingerprint":m["artifact_fingerprint"],"schema_version":m["schema_version"],"build_timestamp":m["created_at_utc"],"research_only":True}
 @app.get("/v1/schema")
 def schema():
  data=json.loads((path/"feature_schema.json").read_text());data["example"]=synthetic_record();return data
 @app.get("/v1/model-card")
 def card():
  if not state["manifest"]:raise HTTPException(503,"Artifact unavailable")
  m=state["manifest"];return {k:m[k] for k in ("artifact_name","model_version","intended_use","prohibited_uses","known_limitations","target_definition","artifact_fingerprint")}|{"research_only":True}
 def parse_payload(payload):
  if not isinstance(payload,dict):raise ValueError("Record must be an object")
  validate_record(payload);return payload
 def decision(score,payload):
  threshold=payload.get("threshold");policy=payload.get("decision_policy")
  if threshold is not None and policy is not None:raise ValueError("Supply threshold or decision_policy, not both")
  source=None
  if policy:
   if policy not in state["manifest"]["threshold_policies"]:raise ValueError("Unknown decision policy")
   threshold=float(state["manifest"]["threshold_policies"][policy]);source="packaged_training_oof_policy"
  elif threshold is not None:
   if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not 0<=threshold<=1:raise ValueError("Threshold must be numeric in [0,1]")
   threshold=float(threshold);source="caller_explicit"
  return (None,None,None,None) if threshold is None else (bool(score>=threshold),threshold,source,policy or "explicit_threshold")
 def result(score,payload):
  d,t,source,policy=decision(score,payload);m=state["manifest"];return {"score":float(score),"decision":d,"threshold":t,"threshold_source":source,"decision_policy":policy,"model_version":m["model_version"],"artifact_fingerprint":m["artifact_fingerprint"],"schema_version":m["schema_version"],"research_only":True,"interpretation":"Research score for future recorded VLE inactivity; not a psychological or causal assessment."}
 @app.post("/v1/validate")
 def validate(payload:dict):
  records=payload.get("records",[payload.get("record",payload)]);errors=[]
  for i,r in enumerate(records):
   try:parse_payload(r)
   except ValueError as exc:errors.append({"index":i,"error":str(exc)})
  return {"valid":not errors,"errors":errors,"research_only":True}
 @app.post("/v1/score")
 def score(payload:dict):
  if not state["bundle"]:raise HTTPException(503,"Artifact unavailable")
  record=payload.get("record",payload);options={k:payload[k] for k in ("threshold","decision_policy") if k in payload}
  try:parse_payload(record);value=score_records(state["bundle"],[record])[0];METRICS["scored_records"]+=1;METRICS["scores"].append(float(value));return result(value,options)
  except ValueError as exc:METRICS["validation_failures"]+=1;raise HTTPException(422,{"category":"validation_error","message":str(exc)})
 @app.post("/v1/score/batch")
 def batch(payload:dict):
  records=payload.get("records")
  if not isinstance(records,list):raise HTTPException(422,"records must be a list")
  if len(records)>max_batch:raise HTTPException(413,f"Batch exceeds maximum {max_batch}")
  options={k:payload[k] for k in ("threshold","decision_policy") if k in payload};output=[];METRICS["batch_sizes"].append(len(records))
  for item in records:
   reference=item.get("record_reference") if isinstance(item,dict) else None;record={k:v for k,v in item.items() if k!="record_reference"} if isinstance(item,dict) else item
   try:parse_payload(record);value=score_records(state["bundle"],[record])[0];row=result(value,options);row["record_reference"]=reference;output.append(row);METRICS["scores"].append(float(value));METRICS["scored_records"]+=1
   except ValueError as exc:METRICS["validation_failures"]+=1;output.append({"record_reference":reference,"error":{"category":"validation_error","message":str(exc)},"research_only":True})
  return {"results":output,"count":len(output),"research_only":True}
 @app.get("/v1/monitoring/summary")
 def monitoring():
  scores=METRICS["scores"];return {"request_count":METRICS["requests"],"scored_records":METRICS["scored_records"],"validation_failures":METRICS["validation_failures"],"scoring_failures":METRICS["scoring_failures"],"artifact_load_failures":METRICS["artifact_load_failures"],"score_count":len(scores),"score_mean":statistics.fmean(scores) if scores else None,"drift_alert_count":METRICS["drift_alerts"],"individual_rows":False,"research_only":True}
 @app.get("/metrics",response_class=PlainTextResponse)
 def metrics():return "\n".join([f"learnpulse_requests_total {METRICS['requests']}",f"learnpulse_validation_failures_total {METRICS['validation_failures']}",f"learnpulse_scoring_failures_total {METRICS['scoring_failures']}",f"learnpulse_scored_records_total {METRICS['scored_records']}",f"learnpulse_artifact_load_failures_total {METRICS['artifact_load_failures']}",f"learnpulse_drift_alerts_total {METRICS['drift_alerts']}"])+"\n"
 return app

def synthetic_record():
 return {"historical_windows_available":14,"historical_average_clicks":50.0,"historical_standard_deviation_clicks":10.0,"click_difference_from_personal_average":-5.0,"click_percentage_change_from_personal_average":-10.0,"click_z_score":-0.5,"historical_average_active_days":4.0,"active_days_difference_from_personal_average":-1.0,"current_inactivity_gap":0,"longest_previous_inactivity_gap":3}

import statistics

app=create_app()
