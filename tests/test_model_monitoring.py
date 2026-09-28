import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from learnpulse.model_monitoring import monitor, psi
from learnpulse.reference_model import FEATURES, build_reference
from tests.reference_helpers import make_table


class MonitoringTests(unittest.TestCase):
 def test_drift_unlabeled_and_alignment(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p,rows=make_table(root);a=root/"a";build_reference(p,a,3,42,"1");ref=json.loads((a/"reference_distributions.json").read_text());records=[{f:rows[i][f] for f in FEATURES} for i in range(10)];scores=np.linspace(.1,.9,10);x=monitor(records,scores,ref);self.assertFalse(x["performance_calculated"]);self.assertEqual(len(x["feature_metrics"]),10)
   with self.assertRaises(ValueError):monitor(records,scores,ref,[0,1]*5,False)
   y=monitor(records,scores,ref,[0,1]*5,True);self.assertTrue(y["performance_calculated"])
 def test_psi(self):self.assertAlmostEqual(psi([50,50],[50,50]),0)
