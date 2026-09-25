"""Miniature fixtures for the read-only subgroup fairness audit."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.fairness_audit import (
    ATTRIBUTES,
    INTERSECTIONS,
    POLICY_COLUMNS,
    UNKNOWN,
    AuditConfig,
    bootstrap_values,
    credit_band,
    disparity_rows,
    group_metrics,
    load_joined_predictions,
    previous_attempt_band,
    reference_group,
    run_audit,
    safe_rate,
    save_audit,
    sha256,
    validate_saved_thresholds,
)


def row(student, label, prediction, score, gender="F", module="AAA", presentation="2013J", day=28):
    return {"module":module,"presentation":presentation,"student_id":student,
        "observation_day":day,"actual_label":label,"calibrated_output":score,
        "outer_holdout_id":f"{module}_{presentation}","student_overlap_after_removal":0,
        **{column:prediction for column in POLICY_COLUMNS.values()},
        "gender":gender,"region":"North","highest_education":"A Level","imd_band":UNKNOWN,
        "age_band":"0-35","disability":"N","num_of_prev_attempts":"0",
        "studied_credits":"credits <= 60","studied_credits_continuous":60.0}


def fixture_files(root: Path):
    predictions=root/"pred.parquet"; info=root/"studentInfo.csv"; thresholds=root/"thresholds.csv"
    with duckdb.connect() as db:
        db.execute("COPY (SELECT 'AAA' module,'2013J' presentation,1 student_id,28 observation_day,1 actual_label,0.8 calibrated_output,'sigmoid' calibration_method,'AAA_2013J' outer_holdout_id,0 student_overlap_after_removal,1 f1_focused_prediction,1 fixed_prediction,1 recall_focused_prediction,1 alert_capacity_prediction) TO ? (FORMAT PARQUET)",[str(predictions)])
    with info.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["code_module","code_presentation","id_student",*ATTRIBUTES]);writer.writeheader()
        writer.writerow({"code_module":"AAA","code_presentation":"2013J","id_student":1,
            "gender":"F","region":"North","highest_education":"A Level","imd_band":"?",
            "age_band":"0-35","disability":"N","num_of_prev_attempts":"0","studied_credits":"60"})
    with thresholds.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["observation_day","outer_holdout_id","calibration_method","threshold_policy","selected_threshold"]);writer.writeheader()
        for policy in ("f1_focused","fixed","recall_focused","alert_capacity"):
            writer.writerow({"observation_day":28,"outer_holdout_id":"AAA_2013J","calibration_method":"sigmoid","threshold_policy":policy,"selected_threshold":.5})
    return predictions,info,thresholds


class FairnessAuditTests(unittest.TestCase):
    def setUp(self):
        self.config=AuditConfig(bootstrap_samples=20,minimum_rows=1,minimum_students=1,
            minimum_positives=1,minimum_negatives=1,minimum_predicted_positives=1,
            intersection_rows=1,intersection_students=1,intersection_positives=1,intersection_negatives=1)

    def test_complete_three_key_join_and_missing_label(self):
        with tempfile.TemporaryDirectory() as folder:
            predictions,info,_=fixture_files(Path(folder))
            rows,validation=load_joined_predictions(predictions,info,self.config)
            self.assertEqual(validation["prediction_rows"],validation["joined_rows"])
            self.assertEqual(rows[0]["gender"],"F");self.assertEqual(rows[0]["imd_band"],UNKNOWN)
            self.assertEqual(rows[0]["num_of_prev_attempts"],"0")
            self.assertEqual(rows[0]["actual_label"],1)

    def test_duplicate_demographic_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            predictions,info,_=fixture_files(Path(folder))
            with info.open("a",encoding="utf-8") as handle:
                handle.write("AAA,2013J,1,F,North,A Level,?,0-35,N,0,60\n")
            with self.assertRaisesRegex(ValueError,"Duplicate studentInfo"):
                load_joined_predictions(predictions,info,self.config)

    def test_unmatched_demographic_row_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            predictions,info,_=fixture_files(Path(folder))
            with info.open("w",encoding="utf-8") as handle: handle.write("code_module,code_presentation,id_student,"+",".join(ATTRIBUTES)+"\n")
            rows,validation=load_joined_predictions(predictions,info,self.config)
            self.assertEqual(validation["unmatched_demographic_rows"],1)
            self.assertTrue(all(rows[0][name]==UNKNOWN for name in ATTRIBUTES))

    def test_saved_thresholds_and_predictions_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            predictions,info,thresholds=fixture_files(Path(folder))
            rows,_=load_joined_predictions(predictions,info,self.config)
            self.assertEqual(validate_saved_thresholds(rows,thresholds,POLICY_COLUMNS)["prediction_policy_checks"],4)
            rows[0]["f1_focused_prediction"]=0
            with self.assertRaisesRegex(ValueError,"disagrees"):
                validate_saved_thresholds(rows,thresholds,POLICY_COLUMNS)

    def test_demographics_are_audit_only(self):
        self.assertTrue(set(ATTRIBUTES).isdisjoint(PERSONAL_CHANGE_FEATURE_COLUMNS))

    def test_derived_groups_are_deterministic(self):
        self.assertEqual(previous_attempt_band("2"),"2 or more")
        self.assertEqual(previous_attempt_band("?"),UNKNOWN)
        self.assertEqual(credit_band(60,[60,120]),"credits <= 60")
        self.assertEqual(credit_band(90,[60,120]),"60 < credits <= 120")

    def test_confusion_and_binary_metric_formulas(self):
        rows=[row(1,1,1,.8),row(2,1,0,.3),row(3,0,1,.6),row(4,0,0,.1)]
        m=group_metrics(rows,"f1_focused",self.config)
        self.assertEqual((m["true_positives"],m["false_positives"],m["false_negatives"],m["true_negatives"]),(1,1,1,1))
        for name in ("recall","false_negative_rate","specificity","false_positive_rate","precision","negative_predictive_value","balanced_accuracy"):
            self.assertEqual(m[name],.5)
        self.assertAlmostEqual(m["brier_score"],(.04+.49+.36+.01)/4)
        self.assertAlmostEqual(m["expected_calibration_error"],(.1+.7+.6+.2)/4)

    def test_zero_denominator_is_null(self):
        self.assertEqual(safe_rate(0,0),(None,"zero_denominator"))
        m=group_metrics([row(1,0,0,.1)],"f1_focused",self.config)
        self.assertIsNone(m["recall"]);self.assertIn("zero_denominator",m["suppression_reasons"]["recall"])
        self.assertIsNone(m["precision"])

    def test_support_suppresses_but_preserves_group(self):
        config=AuditConfig(bootstrap_samples=2)
        m=group_metrics([row(1,1,1,.8)],"f1_focused",config)
        self.assertFalse(m["supported"]);self.assertIsNone(m["recall"])
        self.assertIn("rows<50",m["suppression_reasons"]["recall"])

    def test_reference_ranges_ratios_and_equalized_odds(self):
        metrics=[]
        for group,recall,fpr,n in (("F",.8,.1,100),("M",.5,.3,50)):
            metrics.append({"observation_day":28,"threshold_policy":"f1_focused","attribute":"gender","group":group,
                "rows":n,"supported":True,"suppression_reasons":{},"warning_rate":.2,"recall":recall,
                "false_negative_rate":1-recall,"false_positive_rate":fpr,"precision":.6,
                "brier_score":.1,"expected_calibration_error":.05})
        self.assertEqual(reference_group(metrics),"F")
        disparities=disparity_rows(metrics)
        recall=next(x for x in disparities if x["analysis_type"]=="group_minus_reference" and x["group"]=="M" and x["metric"]=="recall")
        self.assertAlmostEqual(recall["value"],-.3)
        odds=next(x for x in disparities if x["analysis_type"]=="equalized_odds_diagnostic" and x["group"]=="M")
        self.assertAlmostEqual(odds["value"],.3)
        span=next(x for x in disparities if x["analysis_type"]=="pairwise_range" and x["metric"]=="false_positive_rate")
        self.assertAlmostEqual(span["value"],.2);self.assertAlmostEqual(span["max_min_ratio"],3)

    def test_student_bootstrap_preserves_multiplicity_and_is_reproducible(self):
        rows=[row(1,1,1,.8,day=28),row(1,0,0,.2,day=56),row(2,0,0,.1,day=28)]
        first=bootstrap_values(rows,"f1_focused",100,42)
        second=bootstrap_values(rows,"f1_focused",100,42)
        self.assertTrue((first["warning_rate"]==second["warning_rate"]).all())
        self.assertTrue(any(x>0 for x in first["warning_rate"]))
        self.assertTrue(any(x<.5 for x in first["warning_rate"]))

    def test_course_adjustment_and_intersections(self):
        rows=[]
        for module,presentation in (("AAA","2013J"),("BBB","2014J")):
            for i in range(10):
                rows.append(row(i,1 if i%2 else 0,1 if i%3 else 0,.7 if i%3 else .2,
                                gender="F" if i<5 else "M",module=module,presentation=presentation))
        audit=run_audit(rows,self.config)
        self.assertEqual(set(x["intersection"] for x in audit["intersections"]),
                         {f"{a} x {b}" for a,b in INTERSECTIONS})
        self.assertTrue(any(x["contributing_presentations"]==2 for x in audit["course_adjusted"]))
        self.assertTrue(audit["policy_sensitivity"])
        self.assertTrue(audit["missingness"])

    def test_source_files_and_atomic_failure_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); predictions,info,thresholds=fixture_files(root)
            rows,join=load_joined_predictions(predictions,info,self.config)
            audit=run_audit(rows,self.config)
            sources={"predictions":predictions,"student_info":info,"threshold_results":thresholds}
            before={p:sha256(p) for p in sources.values()}
            targets={name:root/name for name in ("results","group_metrics","disparities","intersections","bootstrap_intervals","joined_predictions")}
            targets["figures_dir"]=root/"figures"
            save_audit(audit,rows,join,self.config,sources,targets)
            snapshots={p:p.read_bytes() for name,p in targets.items() if name!="figures_dir"}
            with patch("learnpulse.fairness_audit._publish_atomic",side_effect=RuntimeError("forced")):
                with self.assertRaisesRegex(RuntimeError,"forced"):
                    save_audit(audit,rows,join,self.config,sources,targets)
            self.assertEqual(snapshots,{p:p.read_bytes() for p in snapshots})
            self.assertEqual(before,{p:sha256(p) for p in sources.values()})


if __name__=="__main__": unittest.main()
