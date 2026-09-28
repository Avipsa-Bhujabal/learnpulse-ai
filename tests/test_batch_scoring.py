import csv
import tempfile
import unittest
from pathlib import Path

from learnpulse.batch_scoring import batch_score
from learnpulse.reference_model import FEATURES, build_reference
from tests.reference_helpers import make_table


class BatchTests(unittest.TestCase):
 def test_atomic_input_preserved_and_replace(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p,rows=make_table(root);a=root/"a";build_reference(p,a,3,42,"1");inp=root/"score.csv"
   with inp.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=["record_reference",*FEATURES]);w.writeheader();w.writerow({"record_reference":"x",**{k:rows[0][k] for k in FEATURES}})
   before=inp.read_bytes();out=root/"out.parquet";r=batch_score(inp,out,a);self.assertEqual(r["rows"],1);self.assertEqual(before,inp.read_bytes())
   with self.assertRaises(FileExistsError):batch_score(inp,out,a)
