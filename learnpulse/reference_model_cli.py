import argparse
import json
from pathlib import Path

from learnpulse.artifact_registry import ArtifactRegistry
from learnpulse.reference_model import build_reference


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--modeling-table",type=Path,required=True);p.add_argument("--artifact-dir",type=Path,required=True);p.add_argument("--folds",type=int,default=5);p.add_argument("--random-seed",type=int,default=42);p.add_argument("--model-version",default="1.0.0");a=p.parse_args(argv)
 if not a.modeling_table.is_file() or a.folds<2:raise SystemExit("Valid modeling table and at least two folds required")
 m=build_reference(a.modeling_table,a.artifact_dir,a.folds,a.random_seed,a.model_version);ArtifactRegistry(a.artifact_dir.parent/"registry.json").register(a.artifact_dir)
 print(json.dumps({k:m[k] for k in ("model_version","artifact_fingerprint","training_rows","unique_students","positive_count","class_prevalence","fold_count","convergence_status","build_seconds")},indent=2))
if __name__=="__main__":main()
