import tempfile
import unittest
from pathlib import Path

from learnpulse.artifact_registry import *
from learnpulse.reference_model import build_reference
from tests.reference_helpers import make_table


class RegistryTests(unittest.TestCase):
 def test_register_resolve_tamper_unknown(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);p,_=make_table(root);a=root/"a";build_reference(p,a,3,42,"1.0.0");reg=ArtifactRegistry(root/"registry.json");reg.register(a);self.assertEqual(reg.current(),a.resolve());self.assertEqual(reg.list_versions(),["1.0.0"])
   with self.assertRaises(KeyError):reg.resolve("9")
   (a/"feature_schema.json").write_text("tamper")
   with self.assertRaises(ValueError):validate_artifact(a)
