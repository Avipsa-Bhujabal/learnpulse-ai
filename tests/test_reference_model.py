import tempfile
import unittest
from pathlib import Path

import joblib

from learnpulse.reference_model import *
from tests.reference_helpers import make_table


class ReferenceModelTests(unittest.TestCase):
 def test_schema_forbidden_and_nulls(self):
  self.assertEqual(tuple(PERSONAL_CHANGE_FEATURE_COLUMNS),FEATURES);r={f:1.0 for f in FEATURES};r["student_id"]=1
  with self.assertRaises(ValueError):validate_record(r)
  r.pop("student_id");r["click_z_score"]=None;self.assertEqual(len(validate_record(r)),10)
 def test_grouped_oof_and_reproducible(self):
  with tempfile.TemporaryDirectory() as td:
   p,rows=make_table(Path(td));rs,X,y,g=load_training(p);a=train_reference(rs,X,y,g,5,42);b=train_reference(rs,X,y,g,5,42);self.assertTrue(a[1]["oof_complete"]);self.assertEqual(a[1]["folds"],b[1]["folds"]);self.assertTrue(all(x["student_overlap"]==0 for x in a[1]["folds"]))
 def test_build_and_scoring(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p,rows=make_table(root);m=build_reference(p,root/"artifact",5,42,"1.0.0");b=joblib.load(root/"artifact/model.joblib");r={f:rows[0][f] for f in FEATURES};self.assertTrue(0<=score_records(b,[r])[0]<=1);self.assertEqual(m["artifact_type"],"research_reference_model");self.assertIn("no new external",(root/"artifact/MODEL_CARD.md").read_text().lower())
