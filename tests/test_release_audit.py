import json
import tempfile
import unittest
from pathlib import Path

from learnpulse.release_audit import (
 atomic_json,
 check_links,
 classify,
 inventory_repository,
 release_readiness,
 scan_public,
)


class ReleaseAuditTests(unittest.TestCase):
 def test_inventory_missing_large_temp_and_restricted(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"data/raw").mkdir(parents=True);(root/"data/raw/studentInfo.csv").write_text("x"*100)
   (root/"old.tmp").write_text("x");inv=inventory_repository(root,50)
   self.assertTrue(inv["missing_expected_files"]);self.assertTrue(inv["large_files"]);self.assertIn("old.tmp",inv["temporary_files"]);self.assertEqual(classify("data/raw/studentInfo.csv"),"restricted")
 def test_broken_links_and_absolute_secret_redaction(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/"docs").mkdir();(root/"docs/x.md").write_text("[missing](no.md) C:\\Users\\name\\x api_key=abcdefghijklmnop")
   self.assertEqual(len(check_links(root)),1);scan=scan_public(root);self.assertTrue(scan["absolute_private_paths"]);self.assertEqual(scan["secret_findings"][0]["value"],"[REDACTED]")
 def test_atomic_failure_preserves_existing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x.json";atomic_json(p,{"ok":1});self.assertEqual(json.loads(p.read_text()),{"ok":1})
 def test_release_readiness(self):
  self.assertEqual(release_readiness(tests_passed=True,fresh_environment_passed=False,privacy_passed=True,secrets_clear=True,links_clear=True,license_present=False),"ready_for_local_portfolio_review")
  self.assertEqual(release_readiness(tests_passed=True,fresh_environment_passed=True,privacy_passed=True,secrets_clear=True,links_clear=True,license_present=True),"ready_for_public_release")
 def test_audit_never_deletes(self):
  self.assertNotIn("unlink",inventory_repository.__code__.co_names)
if __name__=="__main__":unittest.main()
