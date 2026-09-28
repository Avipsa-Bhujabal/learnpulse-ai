"""Transparent local registry and integrity validation for reference artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from learnpulse.reference_model import SCHEMA_VERSION, sha256

REQUIRED={"artifact_name","model_version","artifact_type","created_at_utc","feature_names","schema_version","training_dataset_signature","training_rows","unique_students","target_definition","fold_count","artifact_fingerprint","file_hashes","intended_use","prohibited_uses","known_limitations"}

def load_manifest(directory:Path):
 path=directory/"model_manifest.json"
 try:data=json.loads(path.read_text(encoding="utf-8"))
 except Exception as exc:raise ValueError("Malformed or missing artifact manifest") from exc
 missing=REQUIRED-set(data)
 if missing:raise ValueError(f"Incomplete manifest: {sorted(missing)}")
 if data["schema_version"]!=SCHEMA_VERSION:raise ValueError(f"Unsupported schema: {data['schema_version']}")
 return data

def validate_artifact(directory:Path):
 manifest=load_manifest(directory)
 for name,expected in manifest["file_hashes"].items():
  path=directory/name
  if not path.is_file() or sha256(path)!=expected:raise ValueError(f"Artifact hash mismatch: {name}")
 return manifest

class ArtifactRegistry:
 def __init__(self,path:Path):self.path=path
 def _read(self):
  if not self.path.exists():return {"current":None,"versions":{}}
  try:return json.loads(self.path.read_text(encoding="utf-8"))
  except Exception as exc:raise ValueError("Malformed registry") from exc
 def register(self,directory:Path,current=True):
  manifest=validate_artifact(directory);data=self._read();version=manifest["model_version"]
  existing=data["versions"].get(version)
  entry={"path":str(directory.resolve()),"fingerprint":manifest["artifact_fingerprint"]}
  if existing and existing!=entry:raise ValueError("Version already registered with different artifact")
  data["versions"][version]=entry
  if current:data["current"]=version
  self.path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=self.path.parent,prefix=".registry_",suffix=".json");os.close(fd);Path(tmp).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n");os.replace(tmp,self.path)
  return entry
 def list_versions(self):return sorted(self._read()["versions"])
 def resolve(self,version):
  data=self._read()
  if version not in data["versions"]:raise KeyError(f"Unknown model version: {version}")
  path=Path(data["versions"][version]["path"]);validate_artifact(path);return path
 def current(self):
  data=self._read()
  if not data["current"]:raise KeyError("No current research reference version")
  return self.resolve(data["current"])
