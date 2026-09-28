import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from learnpulse.reference_model import build_reference
from learnpulse.scoring_api import create_app, synthetic_record
from tests.reference_helpers import make_table


class ApiTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();root=Path(self.t.name);p,_=make_table(root);self.a=root/"a";build_reference(p,self.a,3,42,"1");self.c=TestClient(create_app(self.a,max_batch=2))
 def tearDown(self):self.t.cleanup()
 def test_endpoints_and_default_no_decision(self):
  for url in ("/health","/ready","/version","/v1/schema","/v1/model-card","/v1/monitoring/summary","/metrics"):self.assertEqual(self.c.get(url).status_code,200)
  x=self.c.post("/v1/score",json={"record":synthetic_record()});self.assertEqual(x.status_code,200);self.assertIsNone(x.json()["decision"]);self.assertTrue(x.json()["research_only"])
  y=self.c.post("/v1/score",json={"record":synthetic_record(),"threshold":.5});self.assertIsInstance(y.json()["decision"],bool)
 def test_validation_batch_and_rejection(self):
  bad=synthetic_record();bad["gender"]="F";self.assertEqual(self.c.post("/v1/score",json={"record":bad}).status_code,422)
  self.assertEqual(self.c.post("/v1/score/batch",json={"records":[synthetic_record()]*3}).status_code,413)
  x=self.c.post("/v1/score/batch",json={"records":[dict(synthetic_record(),record_reference="a"),dict(synthetic_record(),record_reference="b")]});self.assertEqual([z["record_reference"] for z in x.json()["results"]],["a","b"])
 def test_corruption_not_ready(self):
  (self.a/"model.joblib").write_bytes(b"bad");self.assertEqual(self.c.get("/ready").status_code,503)
