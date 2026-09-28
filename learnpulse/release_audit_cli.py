"""CLI for the read-only final repository audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.release_audit import run_release_audit, write_release_reports


def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--repository",type=Path,default=Path("."));p.add_argument("--output-dir",type=Path);p.add_argument("--test-count",type=int);p.add_argument("--finalize",action="store_true");p.add_argument("--test-seconds",type=float,default=0);p.add_argument("--fresh-seconds",type=float);p.add_argument("--fresh-passed",action="store_true")
 a=p.parse_args(argv);result=run_release_audit(a.repository,a.output_dir,a.test_count)
 if a.finalize:write_release_reports(a.repository,test_count=a.test_count or 0,test_seconds=a.test_seconds,fresh_passed=a.fresh_passed,fresh_seconds=a.fresh_seconds)
 print(json.dumps({"missing":result["inventory_summary"]["missing_expected_files"],"broken_links":len(result["broken_links"]),"secret_findings":len(result["public_scan"]["secret_findings"]),"read_only":True},indent=2))
if __name__=="__main__":main()
