"""Atomic, bounded batch scoring for CSV and Parquet inputs."""
from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path

import duckdb
import joblib

from learnpulse.artifact_registry import validate_artifact
from learnpulse.reference_model import score_records, sha256, validate_record


def read_batches(path:Path,size=4096):
 with duckdb.connect() as db:
  if path.suffix.lower()==".parquet":cur=db.execute("select * from read_parquet(?)",[str(path)])
  elif path.suffix.lower()==".csv":cur=db.execute("select * from read_csv_auto(?,header=true)",[str(path)])
  else:raise ValueError("Input must be CSV or Parquet")
  names=[x[0] for x in cur.description]
  while True:
   rows=cur.fetchmany(size)
   if not rows:break
   yield [dict(zip(names,x)) for x in rows]

def batch_score(input_path:Path,output_path:Path,artifact_dir:Path,batch_size=4096,replace=False):
 if output_path.exists() and not replace:raise FileExistsError("Output exists; use --replace")
 before=sha256(input_path);manifest=validate_artifact(artifact_dir);bundle=joblib.load(artifact_dir/"model.joblib");rows=[];started=time.time()
 for batch in read_batches(input_path,batch_size):
  records=[];references=[]
  for row in batch:
   references.append(row.pop("record_reference",None));validate_record(row);records.append(row)
  scores=score_records(bundle,records)
  rows.extend({"record_reference":ref,"score":float(score),"model_version":manifest["model_version"],"artifact_fingerprint":manifest["artifact_fingerprint"],"schema_version":manifest["schema_version"],"research_only":True} for ref,score in zip(references,scores))
 if sha256(input_path)!=before:raise RuntimeError("Input changed during scoring")
 output_path.parent.mkdir(parents=True,exist_ok=True);root=Path(tempfile.mkdtemp(prefix=".batch_score_",dir=output_path.parent));staged=root/output_path.name
 try:
  temp_csv=root/"rows.csv"
  with temp_csv.open("w",newline="",encoding="utf8") as f:
   w=csv.DictWriter(f,fieldnames=rows[0].keys() if rows else ["score"]);w.writeheader();w.writerows(rows)
  dest=staged.resolve().as_posix().replace("'","''")
  with duckdb.connect() as db:
   if output_path.suffix.lower()==".parquet":db.execute(f"copy (select * from read_csv_auto(?,header=true)) to '{dest}' (format parquet,compression zstd)",[str(temp_csv)])
   else:os.replace(temp_csv,staged)
  os.replace(staged,output_path)
 finally:
  import shutil
  if root.exists():shutil.rmtree(root)
 run={"input_signature":{"sha256":before,"size":input_path.stat().st_size},"output":str(output_path),"rows":len(rows),"model_version":manifest["model_version"],"artifact_fingerprint":manifest["artifact_fingerprint"],"runtime_seconds":time.time()-started,"research_only":True}
 run_path=output_path.with_suffix(output_path.suffix+".manifest.json");tmp=run_path.with_suffix(run_path.suffix+".tmp");tmp.write_text(json.dumps(run,indent=2)+"\n");os.replace(tmp,run_path);return run
