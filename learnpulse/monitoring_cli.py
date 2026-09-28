import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib

from learnpulse.artifact_registry import validate_artifact
from learnpulse.batch_scoring import read_batches
from learnpulse.model_monitoring import monitor
from learnpulse.reference_model import score_records


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--artifact-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--synthetic",action="store_true");a=p.parse_args(argv)
 validate_artifact(a.artifact_dir);bundle=joblib.load(a.artifact_dir/"model.joblib");ref=json.loads((a.artifact_dir/"reference_distributions.json").read_text());records=[r for b in read_batches(a.input) for r in b];scores=score_records(bundle,records);report=monitor(records,scores,ref);report["synthetic"]=a.synthetic;a.output.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=a.output.name+".",suffix=".tmp",dir=a.output.parent);os.close(fd)
 try:Path(tmp).write_text(json.dumps(report,indent=2)+"\n");os.replace(tmp,a.output)
 finally:
  if Path(tmp).exists():Path(tmp).unlink()
 print(json.dumps({"rows":len(records),"alerts":report["alert_counts"]},indent=2))
if __name__=="__main__":main()
