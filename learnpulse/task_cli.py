"""Small cross-platform task runner for Step 17 workflows."""
from __future__ import annotations

import argparse
import subprocess
import sys

TASKS = {
    "test": [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    "build-reference-model": [sys.executable, "-m", "learnpulse.reference_model_cli"],
    "validate-artifact": [sys.executable, "-c", "from pathlib import Path; from learnpulse.artifact_registry import validate_artifact; print(validate_artifact(Path('artifacts/reference_model')))"],
    "run-api": [sys.executable, "-m", "uvicorn", "learnpulse.scoring_api:app", "--host", "127.0.0.1", "--port", "8000"],
    "api-smoke-test": [sys.executable, "-m", "unittest", "tests.test_scoring_api", "-v"],
    "batch-score-demo": [sys.executable, "-m", "learnpulse.batch_scoring_cli", "--help"],
    "monitoring-demo": [sys.executable, "-m", "learnpulse.monitoring_cli", "--help"],
    "docker-build": ["docker", "build", "-t", "learnpulse-reference", "."],
    "docker-run": ["docker", "compose", "up", "--build"],
    "release-audit": [sys.executable, "-m", "learnpulse.release_audit_cli", "--repository", ".", "--test-count", "340"],
    "verify-links": [sys.executable, "-m", "learnpulse.release_audit_cli", "--repository", "."],
    "release-manifest": [sys.executable, "-m", "learnpulse.release_audit_cli", "--repository", ".", "--test-count", "340"],
}

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("task", choices=TASKS)
    args = parser.parse_args(); raise SystemExit(subprocess.call(TASKS[args.task]))
if __name__ == "__main__": main()
