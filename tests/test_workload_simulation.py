import copy
import tempfile
import unittest
from pathlib import Path

from learnpulse.workload_simulation import *
from learnpulse.workload_simulation import _case


def event(student,day,warning=1,label=0,score=None,module="A",presentation="P",gap=2,episode=70):
 score=(.7 if warning else .3) if score is None else score
 return {"module":module,"presentation":presentation,"student_id":student,"observation_day":day,"actual_label":label,"calibrated_output":score,
  "f1_focused_threshold":.5,"fixed_threshold":.5,"recall_focused_threshold":.4,"alert_capacity_threshold":.8,
  "f1_focused_prediction":warning,"fixed_prediction":warning,"recall_focused_prediction":warning,"alert_capacity_prediction":int(score>=.8),
  "future_window_start_day":day+1,"future_window_end_day":day+14,"current_inactivity_gap":gap,"episode_start_day":episode,"student_overlap_after_removal":0}

class WorkloadTests(unittest.TestCase):
 def full(self):return [event(s,d,warning=int(s%2==0),label=int(s==2),score=.9 if s%2==0 else .1,gap=None if s==3 else s) for d in DAYS for s in range(1,5)]
 def test_warning_reproduction_and_failure(self):
  self.assertEqual(reproduce_warning(event(1,14),"f1_focused"),1);x=event(1,14);x["f1_focused_prediction"]=0
  with self.assertRaises(ValueError):reproduce_warning(x,"f1_focused")
 def test_capacity_and_minimum(self):
  self.assertEqual(capacity_count(10,3,Capacity("rate",.05,"x")),1);self.assertEqual(capacity_count(100,10,Capacity("rate",.2,"x")),20);self.assertEqual(capacity_count(1,1,Capacity("unlimited",None,"x")),1)
 def test_deterministic_priorities(self):
  q=[_case(event(i,14,score=.5+i*.1,gap=None if i==1 else i),"x",Capacity("rate",.1,"x"),"every_alert","highest_score","f1_focused") for i in (1,2,3)]
  self.assertEqual([x["student_id"] for x in rank_queue(copy.deepcopy(q),"highest_score")],[3,2,1]);self.assertEqual(rank_queue(copy.deepcopy(q),"inactivity_gap")[-1]["student_id"],1)
  self.assertEqual([x["student_id"] for x in rank_queue(copy.deepcopy(q),"earliest_first")],[3,2,1]);self.assertTrue(all("hybrid_priority" in x for x in rank_queue(q,"hybrid")))
 def test_persistence_order_and_labels_not_used(self):
  q=[_case(event(i,14,label=i%2),"x",Capacity("rate",.1,"x"),"every_alert","persistent_warning","f1_focused") for i in (1,2)];q[1]["alerting_checkpoints"]=2
  self.assertEqual(rank_queue(copy.deepcopy(q),"persistent_warning")[0]["student_id"],2);q2=copy.deepcopy(q);[x.update(actual_label=1-x["actual_label"]) for x in q2];self.assertEqual([x["student_id"] for x in rank_queue(q,"persistent_warning")],[x["student_id"] for x in rank_queue(q2,"persistent_warning")])
 def test_backlog_and_unresolved(self):
  cases,cp=simulate_course(self.full(),Capacity("fixed",1,"fixed_1"),"every_alert","highest_score","f1_focused");self.assertTrue(cp[0]["queue_size_after_review"]>0);self.assertTrue(any(x["resolution_status"]=="unresolved_at_horizon" for x in cases));self.assertTrue(all(x["review_delay"]>=0 for x in cases if x["review_delay"] is not None))
 def test_open_case_merge(self):
  cases,cp=simulate_course(self.full(),Capacity("fixed",0,"fixed_0"),"open_case","highest_score","f1_focused");self.assertEqual(len(cases),2);self.assertEqual(sum(x["alerts_merged"] for x in cp),6)
 def test_cooldown_and_repeats(self):
  ev=[event(2,d) for d in DAYS];c14,cp14=simulate_course(ev,Capacity("unlimited",None,"u"),"cooldown_14","highest_score","f1_focused");c28,cp28=simulate_course(ev,Capacity("unlimited",None,"u"),"cooldown_28","highest_score","f1_focused")
  self.assertEqual(len(c14),4);self.assertEqual(len(c28),2);self.assertGreater(sum(x["alerts_suppressed_by_cooldown"] for x in cp28),0);self.assertGreater(sum(x["repeated_reviews"] for x in cp14),0)
 def test_unlimited_same_checkpoint(self):
  cases,cp=simulate_course(self.full(),Capacity("unlimited",None,"u"),"every_alert","highest_score","f1_focused");self.assertTrue(all(x["review_delay"]==0 for x in cases));self.assertTrue(all(x["queue_size_after_review"]==0 for x in cp))
 def test_origin_label_and_episode_lead(self):
  cases,cp=simulate_course([event(2,d,label=int(d==14),episode=20) for d in DAYS],Capacity("fixed",1,"f"),"open_case","highest_score","f1_focused");self.assertEqual(cases[0]["actual_label"],1);s=summarize_scenario(cases,cp,[event(2,d,label=int(d==14),episode=20) for d in DAYS]);self.assertEqual(s["episodes_reviewed_before_start"],1);self.assertGreater(s["minimum_useful_lead_time"],0)
 def test_late_review_has_no_lead_and_no_episode_separate(self):
  ev=[event(2,d,episode=10) for d in DAYS]+[event(4,d,episode=None) for d in DAYS];cases,cp=simulate_course(ev,Capacity("unlimited",None,"u"),"open_case","highest_score","f1_focused");s=summarize_scenario(cases,cp,ev);self.assertIsNone(s["median_useful_lead_time"]);self.assertEqual(s["no_episode_reviewed_cases"],1)
 def test_model_and_capacity_misses(self):
  ev=[event(1,d,warning=0,label=1) for d in DAYS]+[event(2,d,warning=1,label=1) for d in DAYS];cases,cp=simulate_course(ev,Capacity("fixed",0,"z"),"every_alert","highest_score","f1_focused");s=summarize_scenario(cases,cp,ev);self.assertEqual(s["model_misses"],4);self.assertEqual(s["capacity_misses"],4)
 def test_independent_courses(self):
  a=self.full();b=[dict(x,module="B") for x in self.full()];audit=run_simulation(a+b,[Capacity("fixed",1,"f")],["open_case"],["highest_score"],["f1_focused"],{},2,42);self.assertEqual({(x["module"],x["presentation"]) for x in audit["checkpoints"]},{("A","P"),("B","P")})
 def test_demographics_not_in_rank_and_support(self):
  self.assertTrue(set(AUDIT_ATTRIBUTES).isdisjoint({"most_recent_score","first_alert_day","student_id","current_inactivity_gap"}));cases,cp=simulate_course(self.full(),Capacity("unlimited",None,"u"),"every_alert","highest_score","f1_focused");d={("A","P",1):{"gender":"F"}};r=subgroup_results(cases,d,50,25,10);self.assertTrue(any(x["group"]==UNKNOWN for x in r));self.assertTrue(all(not x["supported"] for x in r))
 def test_bootstrap_accounting_reproducible(self):
  cases,cp=simulate_course(self.full(),Capacity("unlimited",None,"unlimited"),"open_case","highest_score","f1_focused");s=[summarize_scenario(cases,cp,self.full())];a=bootstrap_intervals(cases,s,self.full(),5,42);b=bootstrap_intervals(cases,s,self.full(),5,42);self.assertEqual(a,b);self.assertTrue(all(x["valid_samples"]+x["skipped_samples"]==5 for x in a))
 def test_signatures_source_unchanged(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/"x";p.write_bytes(b"abc");a=file_signature(p);file_signature(p);self.assertEqual(a,file_signature(p))

if __name__=="__main__":unittest.main()
