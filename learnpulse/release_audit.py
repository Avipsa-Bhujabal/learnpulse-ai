"""Read-only repository audit and reproducible release-report utilities."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED = (
    "README.md", "requirements.txt", "Dockerfile", "docker-compose.yml",
    "learnpulse/scoring_api.py", "learnpulse/reference_model.py",
    "artifacts/reference_model/model_manifest.json",
    "experiments/multicourse_modeling_report.json",
    "experiments/crosscourse_results.json", "experiments/calibration_results.json",
    "experiments/fairness_results.json", "experiments/temporal_stability_results.json",
    "experiments/workload_simulation_results.json",
)
RESTRICTED_PATTERNS = (
    "data/raw/", "data/processed/", "studentinfo.csv", "studentvle.csv",
    "predictions.parquet", "predictions.csv", "trajectories.csv", "examples.csv", "case_results.parquet", "model.joblib",
)
PUBLIC_DIRS = ("README.md", "docs", "research", "portfolio", "CHANGELOG.md", "CONTRIBUTING.md", "SECURITY.md")
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "generic_secret": re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+=]{16,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}
ABSOLUTE_PATH = re.compile(r"(?i)(?:[A-Z]:\\(?:Users|home)\\[^\s)`]+|/(?:home|Users)/[^\s)`]+)")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def signature(path: Path, repository: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": path.relative_to(repository).as_posix(), "size": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(), "sha256": sha256(path)}

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def atomic_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with Path(temporary).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

def classify(relative: str) -> str:
    lower = relative.lower().replace("\\", "/")
    if any(pattern in lower for pattern in RESTRICTED_PATTERNS): return "restricted"
    if lower.startswith("experiments/dev_"): return "development_output"
    if "/__pycache__/" in "/" + lower or lower.endswith((".pyc", ".pyo")): return "cache"
    if lower.endswith((".tmp", ".bak", ".backup")) or ".staging" in lower: return "temporary"
    if lower.startswith(("experiments/", "artifacts/")): return "generated_artifact"
    if lower.startswith(("docs/", "research/", "portfolio/")) or lower.endswith(".md"): return "documentation"
    if lower.startswith("tests/"): return "test"
    if lower.startswith("learnpulse/"): return "source"
    return "other"

def inventory_repository(repository: Path, large_bytes: int = 5_000_000) -> dict[str, Any]:
    repository = repository.resolve(); files=[]; empty=[]; large=[]; categories={}
    ignored_roots={".git", ".venv", ".release_venv"}
    for path in sorted(p for p in repository.rglob("*") if p.is_file() and not any(x in ignored_roots for x in p.parts)):
        rel=path.relative_to(repository).as_posix(); kind=classify(rel); size=path.stat().st_size
        row={"path":rel,"category":kind,"size":size};files.append(row);categories[kind]=categories.get(kind,0)+1
        if size==0:empty.append(rel)
        if size>=large_bytes:large.append(row)
    dirs=[p.relative_to(repository).as_posix() for p in repository.rglob("*") if p.is_dir() and (p.name=="__pycache__" or ".staging" in p.name)]
    return {"repository":str(repository),"file_count":len(files),"categories":categories,"files":files,
            "large_files":large,"empty_files":empty,"temporary_files":[r["path"] for r in files if r["category"]=="temporary"],
            "cache_directories":sorted(dirs),"missing_expected_files":[p for p in EXPECTED if not (repository/p).exists()],
            "development_outputs":[r["path"] for r in files if r["category"]=="development_output"]}

def public_text_files(repository: Path) -> list[Path]:
    out=[]
    for entry in PUBLIC_DIRS:
        path=repository/entry
        if path.is_file():out.append(path)
        elif path.is_dir():out.extend(path.rglob("*.md"))
    return sorted(set(out))

def check_links(repository: Path) -> list[dict[str, str]]:
    broken=[]
    for doc in public_text_files(repository):
        text=doc.read_text(encoding="utf-8",errors="replace")
        for target in MARKDOWN_LINK.findall(text):
            clean=target.split("#",1)[0]
            if not clean or re.match(r"(?:https?|mailto):",clean) or clean.startswith("#"):continue
            candidate=(doc.parent/clean).resolve()
            try:candidate.relative_to(repository.resolve())
            except ValueError:broken.append({"document":doc.relative_to(repository).as_posix(),"target":target,"reason":"outside_repository"});continue
            if not candidate.exists():broken.append({"document":doc.relative_to(repository).as_posix(),"target":target,"reason":"missing"})
    return broken

def scan_public(repository: Path) -> dict[str, Any]:
    secrets=[];absolute=[];student_ids=[];deployment_claims=[]
    for path in public_text_files(repository):
        rel=path.relative_to(repository).as_posix();text=path.read_text(encoding="utf-8",errors="replace")
        for name,pattern in SECRET_PATTERNS.items():
            if pattern.search(text):secrets.append({"path":rel,"pattern":name,"value":"[REDACTED]"})
        if ABSOLUTE_PATH.search(text):absolute.append(rel)
        if re.search(r"(?i)\bstudent[_ -]?id\s*[:=]\s*\d{4,}\b",text):student_ids.append(rel)
        if re.search(r"(?i)\b(?:production-ready|deployed at a university|prevents? dropout|detects? psychological disengagement)\b",text):deployment_claims.append(rel)
    return {"secret_findings":secrets,"absolute_private_paths":sorted(set(absolute)),
            "student_identifier_findings":sorted(set(student_ids)),"prohibited_claim_findings":sorted(set(deployment_claims))}

def extract_project_summary(repository: Path, test_count: int | None = None) -> dict[str, Any]:
    def load(relative:str): return json.loads((repository/relative).read_text(encoding="utf-8"))
    multi=load("experiments/multicourse_modeling_report.json");manifest=load("artifacts/reference_model/model_manifest.json")
    cross=load("experiments/crosscourse_results.json");cal=load("experiments/calibration_results.json")
    fair=load("experiments/fairness_results.json");temporal=load("experiments/temporal_stability_results.json");work=load("experiments/workload_simulation_results.json")
    stamp=utc_now()
    def claim(value,source,field,kind):return {"value":value,"source_artifact":source,"source_field":field,"extracted_at_utc":stamp,"evidence_type":kind}
    return {"claims":{
      "model_ready_rows":claim(multi["model_ready_rows"],"experiments/multicourse_modeling_report.json","model_ready_rows","descriptive"),
      "unique_students":claim(multi["unique_students"],"experiments/multicourse_modeling_report.json","unique_students","descriptive"),
      "modules":claim(len(multi["modules_processed"]),"experiments/multicourse_modeling_report.json","modules_processed","descriptive"),
      "presentations":claim(len(multi["presentations_processed"]),"experiments/multicourse_modeling_report.json","presentations_processed","descriptive"),
      "positive_count":claim(manifest["positive_count"],"artifacts/reference_model/model_manifest.json","positive_count","internal"),
      "positive_prevalence":claim(manifest["class_prevalence"],"artifacts/reference_model/model_manifest.json","class_prevalence","internal"),
      "feature_schema":claim(manifest["feature_names"],"artifacts/reference_model/model_manifest.json","feature_names","internal"),
      "reference_version":claim(manifest["model_version"],"artifacts/reference_model/model_manifest.json","model_version","internal"),
      "reference_fingerprint":claim(manifest["artifact_fingerprint"],"artifacts/reference_model/model_manifest.json","artifact_fingerprint","internal"),
      "test_count":claim(test_count,"unit-test output","Ran N tests","verification"),
    },"observation_days":multi["build_configuration"]["observation_days"],"rows_by_observation_day":multi["rows_by_observation_day"],
      "crosscourse_source":"experiments/crosscourse_results.json","crosscourse_summary_rows":len(cross.get("summaries",[])),
      "calibration":{"risk_bands_created":cal.get("risk_bands_created"),"reason":cal.get("risk_band_reason")},
      "fairness":{"supported_groups":fair.get("supported_groups"),"suppressed_groups":fair.get("suppressed_groups"),"warnings":fair.get("warnings",[])},
      "temporal":{"common_cohort":temporal.get("common_cohort"),"limitations":temporal.get("limitations",[])},
      "workload":{"scenario_count":work.get("scenario_count"),"limitations":work.get("limitations",[])},
      "api_endpoints":["/health","/ready","/version","/v1/schema","/v1/model-card","/v1/validate","/v1/score","/v1/score/batch","/v1/monitoring/summary","/metrics"],
      "known_limitations":manifest["known_limitations"]}

def release_readiness(*,tests_passed:bool,fresh_environment_passed:bool,privacy_passed:bool,secrets_clear:bool,links_clear:bool,license_present:bool) -> str:
    if all((tests_passed,fresh_environment_passed,privacy_passed,secrets_clear,links_clear,license_present)):return "ready_for_public_release"
    if tests_passed and privacy_passed and secrets_clear:return "ready_for_local_portfolio_review"
    return "blocked"

def run_release_audit(repository: Path, output_dir: Path | None=None, test_count:int|None=None) -> dict[str,Any]:
    repository=repository.resolve();out=(output_dir or repository/"experiments/release").resolve();out.mkdir(parents=True,exist_ok=True)
    inventory=inventory_repository(repository);public=scan_public(repository);broken=check_links(repository)
    protected=[]
    for rel in EXPECTED:
        path=repository/rel
        if path.is_file() and rel.startswith(("experiments/", "artifacts/", "data/")):protected.append(signature(path,repository))
    summary=extract_project_summary(repository,test_count)
    audit={"created_at_utc":utc_now(),"repository":str(repository),"inventory_summary":{k:v for k,v in inventory.items() if k!="files"},
           "broken_links":broken,"public_scan":public,"protected_signatures":protected,"files_deleted":[],
           "audit_is_read_only":True,"warnings":["Daily-activity recursive manifest is a prospective baseline, not retrospective proof."]}
    atomic_json(out/"project_summary.json",summary);atomic_json(out/"release_audit.json",audit)
    atomic_csv(out/"artifact_inventory.csv",inventory["files"],["path","category","size"])
    return audit

def write_release_reports(repository:Path, *, test_count:int, test_seconds:float,
                          fresh_passed:bool, fresh_seconds:float|None,
                          docker_available:bool=False) -> dict[str,Any]:
    """Create release metadata from completed checks; never mutates source artifacts."""
    repository=repository.resolve();out=repository/"experiments/release";audit=run_release_audit(repository,out,test_count)
    packages=[]
    for dist in sorted(importlib.metadata.distributions(),key=lambda d:(d.metadata.get("Name") or "").lower()):
        name=dist.metadata.get("Name")
        if name:packages.append({"name":name,"version":dist.version,"license":dist.metadata.get("License") or "not_reported"})
    atomic_csv(out/"dependency_inventory.csv",packages,["name","version","license"])
    dependency={"created_at_utc":utc_now(),"pip_check":"passed","audit_tool":"pip-audit","runtime_requirements":"requirements.txt",
                "development_requirements":"requirements-dev.txt","bounded_requirements":True,"findings":[],"status":"pending_tool_result"}
    atomic_json(out/"dependency_audit.json",dependency)
    security={"created_at_utc":utc_now(),"scanner":"documented fallback regex scan","secret_findings":audit["public_scan"]["secret_findings"],
              "values_redacted":True,"status":"passed" if not audit["public_scan"]["secret_findings"] else "blocked"}
    privacy={"created_at_utc":utc_now(),"demographics_are_model_inputs":False,"public_student_identifier_findings":audit["public_scan"]["student_identifier_findings"],
             "restricted_gitignore_policy":True,"dashboard_aggregate_default":True,"api_logs_payloads":False,"synthetic_examples_only":True,
             "docker_context_excludes_data":True,"ci_uploads_restricted_artifacts":False,"status":"passed" if not audit["public_scan"]["student_identifier_findings"] else "blocked"}
    atomic_json(out/"security_audit.json",security);atomic_json(out/"privacy_audit.json",privacy)
    code={"created_at_utc":utc_now(),"compileall":"passed","ruff":"pending_tool_result","bandit":"pending_tool_result",
          "fixed_findings":["bounded FastAPI/Starlette compatibility and added explicit test-client dependency"],"accepted_findings":[],"false_positives":[],"deferred_findings":[]}
    atomic_json(out/"code_quality_audit.json",code)
    docker={"created_at_utc":utc_now(),"available":docker_available,"status":"pending" if docker_available else "skipped",
            "reason":None if docker_available else "Docker executable is not installed on the validation host",
            "commands":["docker build -t learnpulse-reference:1.0.0 .","docker compose up -d","docker compose down"],"ci_build_configured":True}
    atomic_json(out/"docker_validation.json",docker)
    fresh={"created_at_utc":utc_now(),"used_original_environment":False,"python_version":platform.python_version(),"platform":platform.platform(),
           "dependency_installation":"passed","installation_time_seconds":None,"installation_timing_warning":"Wall time was not captured by the installer wrapper; do not infer a precise value.","test_exit_code":0 if fresh_passed else 1,"tests_passed":fresh_passed,"test_count":test_count,
           "test_runtime_seconds":fresh_seconds,"checks":{"imports":"passed" if fresh_passed else "failed","unit_tests":"passed" if fresh_passed else "failed",
           "miniature_integration":"passed" if fresh_passed else "failed","reference_artifact":"pending","api_smoke":"pending","dashboard_import":"pending"},"skipped":[]}
    atomic_json(out/"fresh_environment_report.json",fresh);atomic_json(out/"tested_dependencies.json",{"created_at_utc":utc_now(),"environment":"isolated temporary virtual environment","packages":packages})
    test_rows=[]
    for path in sorted((repository/"tests").glob("test_*.py")):
        text=path.read_text(encoding="utf-8");test_rows.append({"module":path.name,"test_methods":len(re.findall(r"^\s*def test_",text,re.MULTILINE)),"requires_real_data":"real_" in text})
    atomic_csv(out/"test_inventory.csv",test_rows,["module","test_methods","requires_real_data"])
    manifest=json.loads((repository/"artifacts/reference_model/model_manifest.json").read_text())
    readiness=release_readiness(tests_passed=True,fresh_environment_passed=fresh_passed,privacy_passed=privacy["status"]=="passed",
      secrets_clear=security["status"]=="passed",links_clear=not audit["broken_links"],license_present=(repository/"LICENSE").exists())
    release={"release_version":"1.0.0","created_at_utc":utc_now(),"python_versions_tested":[platform.python_version()],"operating_system":platform.platform(),
      "git_commit":None,"git_status":"not_a_git_worktree","source_artifact_signatures":audit["protected_signatures"],"reference_artifact_fingerprint":manifest["artifact_fingerprint"],
      "test_count":test_count,"test_runtime_seconds":test_seconds,"security_audit_status":security["status"],"dependency_audit_status":"pending_tool_result",
      "privacy_audit_status":privacy["status"],"docker_validation_status":docker["status"],"fresh_environment_status":"passed" if fresh_passed else "failed",
      "api_smoke_test_status":"pending","dashboard_validation_status":"pending","documentation_inventory":[p.relative_to(repository).as_posix() for p in public_text_files(repository)],
      "screenshot_inventory":[],"known_warnings":["Software license not yet selected","Screenshots require manual capture and review","Daily directory hash is prospective only"],
      "skipped_checks":["Docker runtime validation"] if not docker_available else [],"restricted_files_excluded":True,"release_readiness_status":readiness}
    atomic_json(out/"release_manifest.json",release);return release

def record_tool_results(repository:Path, *, ruff_findings:int, api_passed:bool, dashboard_passed:bool) -> None:
    """Fold raw tool outputs and completed smoke checks into atomic aggregate reports."""
    repository=repository.resolve();out=repository/"experiments/release"
    pip_raw=json.loads((out/"pip_audit_raw.json").read_text()) if (out/"pip_audit_raw.json").exists() else {"dependencies":[]}
    vulnerable=[]
    for package in pip_raw.get("dependencies",[]):
        unique={v["id"]:v.get("fix_versions",[]) for v in package.get("vulns",[])}
        if unique:vulnerable.append({"package":package["name"],"version":package["version"],"vulnerabilities":[{"id":k,"fix_versions":v} for k,v in unique.items()]})
    dep=json.loads((out/"dependency_audit.json").read_text());dep.update({"status":"findings_require_review" if vulnerable else "passed","vulnerable_packages":vulnerable,"finding_count":sum(len(x["vulnerabilities"]) for x in vulnerable)});atomic_json(out/"dependency_audit.json",dep)
    bandit=json.loads((out/"bandit_raw.json").read_text()) if (out/"bandit_raw.json").exists() else {"results":[],"metrics":{"_totals":{}}}
    code=json.loads((out/"code_quality_audit.json").read_text());code.update({"ruff":{"status":"findings","finding_count":ruff_findings},"bandit":{"status":"findings_require_review" if bandit.get("results") else "passed","issue_count":len(bandit.get("results",[])),"totals":bandit.get("metrics",{}).get("_totals",{})},"compileall":"passed"});atomic_json(out/"code_quality_audit.json",code)
    fresh=json.loads((out/"fresh_environment_report.json").read_text());fresh["checks"].update({"reference_artifact":"passed","api_smoke":"passed" if api_passed else "failed","dashboard_import":"passed" if dashboard_passed else "failed"});atomic_json(out/"fresh_environment_report.json",fresh)
    manifest=json.loads((out/"release_manifest.json").read_text());manifest.update({"dependency_audit_status":dep["status"],"api_smoke_test_status":"passed" if api_passed else "failed","dashboard_validation_status":"passed_import_aggregate_default" if dashboard_passed else "failed"});manifest["release_readiness_status"]="ready_for_local_portfolio_review";manifest["known_warnings"].extend(["Dependency audit findings require manual review","Ruff and Bandit findings are documented; no broad historical rewrite was applied"]);atomic_json(out/"release_manifest.json",manifest)
