import tempfile
import unittest
from pathlib import Path

from learnpulse.temporal_stability import *


def rows(bits='0111',scores=(.1,.4,.6,.8),student=1):
 return [{'module':'AAA','presentation':'2013J','student_id':student,'observation_day':d,'actual_label':int(i%2==0),'calibrated_output':scores[i],**{v:int(bits[i]) for v in POLICIES.values()}} for i,d in enumerate(DAYS)]
class TemporalTests(unittest.TestCase):
 def test_sequences_and_missing(self):
  self.assertEqual(sequence(rows()),'0111')
  with self.assertRaises(ValueError):sequence(rows()[:3])
 def test_categories(self):
  self.assertTrue(trajectory_flags('0111',[.1,.2,.3,.4])['persistent_warning'])
  self.assertTrue(trajectory_flags('1000',[.4,.3,.2,.1])['temporary_warning'])
  self.assertTrue(trajectory_flags('1010',[.1,.2,.3,.4])['recurrent_warning'])
  self.assertTrue(trajectory_flags('1010',[.1,.2,.3,.4])['alternating_warning'])
  self.assertTrue(trajectory_flags('0000',[.1,.2,.3,.4],.05)['warning_escalation'])
  self.assertTrue(trajectory_flags('0000',[.4,.3,.2,.1],.05)['warning_deescalation'])
 def test_first_and_never(self):
  self.assertEqual(trajectory_flags('0100',[0]*4)['first_warning_day'],28)
  self.assertTrue(trajectory_flags('0000',[0]*4)['never_warned'])
 def test_common_and_pairs(self):
  common,pairs,_=common_and_pairs(rows()+rows()[:3])
  self.assertEqual(len(common),1);self.assertEqual(len(pairs[(14,28)]),1)
 def test_transition_counts(self):
  _,pairs,_=common_and_pairs(rows('0101'))
  t=transition_rows(pairs);self.assertEqual(sum(x['count'] for x in t),3);self.assertAlmostEqual(sum(x['percentage'] for x in t if x['from_day']==14),1)
 def test_score_tolerance(self):
  _,pairs,_=common_and_pairs(rows('0000',(.1,.13,.25,.1)))
  t=transition_rows(pairs,tolerance=.05);self.assertEqual(next(x for x in t if x['from_day']==14)['stable'],1)
 def test_strategies(self):
  self.assertTrue(strategy_state('1100',1,'two_consecutive'));self.assertTrue(strategy_state('1010',2,'two_of_three'));self.assertTrue(strategy_state('0001',3,'persistent_or_late'))
 def test_day_specific_labels(self):
  p=policy_results({('A','P',1):rows('1111')});self.assertEqual(next(x for x in p if x['observation_day']==14 and x['strategy']=='single')['positive_cases'],1);self.assertEqual(next(x for x in p if x['observation_day']==28 and x['strategy']=='single')['positive_cases'],0)
 def test_confirmation_cost(self):
  p=policy_results({('A','P',1):rows('1011')});r=next(x for x in p if x['observation_day']==42 and x['strategy']=='two_consecutive');self.assertGreaterEqual(r['warnings_prevented'],0);self.assertGreaterEqual(r['true_positives_lost'],0)
 def test_episode(self):
  self.assertEqual(episode_for([15,16,31],60)['episode_start_day'],17);self.assertFalse(episode_for([],20)['episode_eligible'])
 def test_no_negative_lead(self):
  common={('AAA','2013J',1):rows('0001')};ep={('AAA','2013J',1):{'episode_start_day':30,'episode_end_day':43,'episode_eligible':True,'first_prior_warning_day':None,'warning_lead_time':None,'warning_timing':'late'}};self.assertIsNone(trajectory_table(common,ep)[0]['warning_lead_time'])
 def test_metrics_and_degenerate(self):
  m=binary_metrics([1,0],[1,1]);self.assertEqual(m['precision'],.5);self.assertIsNone(binary_metrics([1],[1])['specificity'])
 def test_stability(self):
  s=stability({('A','P',1):rows('1110'),('A','P',2):rows('1010',student=2)});self.assertEqual(s['warning_persistence_rate'],.5);self.assertEqual(s['recurrent_warning_rate'],.5)
 def test_kappa_spearman(self):
  _,pairs,_=common_and_pairs(rows('0000')+rows('1111',student=2));x=kappa_spearman(pairs);self.assertEqual(x[0]['cohens_kappa'],1)
 def test_bootstrap_reproducible(self):
  c={('A','P',1):rows('1111'),('A','P',2):rows('0000',student=2)};self.assertEqual(bootstrap_summary(c,10,42),bootstrap_summary(c,10,42))
 def fixture_trajectories(self):
  return [
   {'module':'A','presentation':'P','student_id':1,'warning_sequence':'1100','warning_timing':'before_start','episode_start_day':40,**{f'day{d}_label':int(d==28) for d in DAYS}},
   {'module':'A','presentation':'P','student_id':2,'warning_sequence':'0001','warning_timing':'late','episode_start_day':40,**{f'day{d}_label':int(d==56) for d in DAYS}},
   {'module':'A','presentation':'P','student_id':3,'warning_sequence':'1000','warning_timing':'before_start','episode_start_day':None,**{f'day{d}_label':0 for d in DAYS}},
   {'module':'A','presentation':'P','student_id':4,'warning_sequence':'0000','warning_timing':'never_warned','episode_start_day':35,**{f'day{d}_label':0 for d in DAYS}},
  ]
 def test_episode_summary_excludes_no_episode_alert(self):
  x=corrected_episode_summary(self.fixture_trajectories());self.assertEqual(x['eligible_episode_count'],3);self.assertEqual(x['warned_before_episode_start'],1);self.assertEqual(x['false_alert_trajectory_without_episode'],1);self.assertEqual(x['warned_before_episode_start']+x['warned_after_episode_start']+x['eligible_episode_never_warned'],3)
 def test_strategy_valid_days(self):
  with self.assertRaises(ValueError):strategy_state('1111',0,'persistent_or_late')
  p=policy_results({('A','P',1):rows('1111')});self.assertFalse(any(x['strategy']=='persistent_or_late' and x['observation_day']==14 for x in p))
 def test_confirmation_and_lead_persisted(self):
  tr=self.fixture_trajectories();c=confirmation_delay_rows(tr);self.assertEqual({x['strategy'] for x in c},{'two_consecutive','two_of_three','persistent_or_late'})
  e=episode_strategy_rows(tr);late=next(x for x in e if x['strategy']=='single');self.assertEqual(late['warned_after_episode_start'],1);self.assertGreaterEqual(late['minimum_nonnegative_lead_time'],0)
 def test_course_day_alignment_and_macro_null(self):
  course=course_strategy_results(self.fixture_trajectories());self.assertFalse(any(x['strategy']=='two_of_three' and x['observation_day']<42 for x in course))
  macro=course_macro_summary([{'observation_day':14,'strategy':'single','alert_rate':.1,'precision':None,'recall':.2,'f1':.15,'false_positive_rate':None,'false_positives_prevented':0,'true_positives_lost':0},{'observation_day':14,'strategy':'single','alert_rate':.3,'precision':.5,'recall':.4,'f1':.3,'false_positive_rate':.2,'false_positives_prevented':1,'true_positives_lost':1}]);r=next(x for x in macro if x['metric']=='precision');self.assertEqual(r['mean'],.5);self.assertEqual(r['contributing_presentations'],1)
 def test_reporting_bootstraps_and_multiplicity(self):
  out=reporting_bootstrap(self.fixture_trajectories(),20,42);self.assertTrue(any(x['metric']=='first_warning_day_14' for x in out));self.assertTrue(all('valid_samples' in x and 'skipped_samples' in x for x in out));self.assertTrue(any(x['bootstrap_unit']=='strict_student_id' for x in out))
 def test_manifest_deterministic(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.parquet';p.write_bytes(b'not-really-parquet');self.assertEqual(recursive_parquet_manifest(Path(td)),recursive_parquet_manifest(Path(td)))
if __name__=='__main__':unittest.main()
