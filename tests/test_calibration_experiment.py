"""Tests for leakage-safe calibration and threshold selection."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.calibration_experiment import (
    MODEL_B_FEATURE_COLUMNS,
    THRESHOLD_CHART_METRICS,
    _logits,
    _plot_thresholds,
    apply_calibrator,
    calibration_bins,
    calibration_intercept_slope,
    calibration_metrics,
    create_inner_folds,
    fit_calibrators,
    inner_oof_scores,
    refresh_saved_reporting,
    run_calibration_experiment,
    save_calibration_outputs,
    select_thresholds,
    threshold_candidates,
    threshold_tradeoff_curves,
    validate_calibration_schema,
    worst_presentations,
)
from learnpulse.crosscourse_experiment import (
    FORBIDDEN_COLUMNS,
    create_course_holdout_splits,
)


def fixture_rows():
    rows=[]
    for unit,(module,presentation) in enumerate([("AAA","2013J"),("AAA","2014J"),("BBB","2013J"),("BBB","2014J")]):
        students=list(range(1,5))+list(range(100+unit*30,120+unit*30))
        for student_position,student in enumerate(students):
            row={"module":module,"presentation":presentation,"student_id":student,"observation_day":28,"future_inactivity":int((student_position+unit)%5==0)}
            for index,name in enumerate(PERSONAL_CHANGE_FEATURE_COLUMNS): row[name]=None if name=="click_z_score" and student_position%9==0 else float(student_position+index)
            rows.append(row)
    return rows


class CalibrationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.rows=fixture_rows()

    def test_exact_schema_and_forbidden_columns_excluded(self):
        validate_calibration_schema(); self.assertEqual(MODEL_B_FEATURE_COLUMNS,PERSONAL_CHANGE_FEATURE_COLUMNS)
        self.assertTrue(set(MODEL_B_FEATURE_COLUMNS).isdisjoint(FORBIDDEN_COLUMNS))

    def test_strict_outer_holdout_excludes_presentation_and_overlapping_students(self):
        split=create_course_holdout_splits(self.rows,28,"presentation",["student-disjoint"],"AAA","2013J")[0]
        self.assertFalse(any(self.rows[i]["module"]=="AAA" and self.rows[i]["presentation"]=="2013J" for i in split.train_indices))
        self.assertFalse({self.rows[i]["student_id"] for i in split.train_indices}&{self.rows[i]["student_id"] for i in split.test_indices})
        self.assertGreater(split.training_rows_removed,0)

    def test_inner_folds_are_inside_outer_training_and_group_students(self):
        split=create_course_holdout_splits(self.rows,28,"presentation",["student-disjoint"],"AAA","2013J")[0]
        folds,_=create_inner_folds(self.rows,split.train_indices,3,42)
        outer=set(split.train_indices)
        for fold in folds:
            self.assertTrue(set(fold.train_indices+fold.validation_indices)<=outer)
            self.assertFalse(fold.train_students&fold.validation_students)

    def test_every_outer_training_row_receives_inner_oof_score(self):
        split=create_course_holdout_splits(self.rows,28,"presentation",["student-disjoint"],"AAA","2013J")[0]
        folds,_=create_inner_folds(self.rows,split.train_indices,3,42)
        scores,labels,count,fitted=inner_oof_scores(self.rows,split.train_indices,folds,42)
        self.assertEqual(len(scores),len(split.train_indices)); self.assertEqual(len(labels),len(scores)); self.assertEqual(count,len(fitted))
        for pipeline in fitted: self.assertEqual(len(pipeline.named_steps["imputer"].statistics_),len(MODEL_B_FEATURE_COLUMNS))

    def test_calibrators_fit_oof_scores_and_outputs_are_bounded(self):
        scores=[.1,.2,.3,.6,.8,.9]; labels=[0,0,1,0,1,1]
        calibrators,_=fit_calibrators(scores,labels,42)
        for method in ("sigmoid","isotonic"):
            values=apply_calibrator(method,calibrators[method],[-1,0,.5,1,2])
            self.assertTrue(all(0<=value<=1 for value in values))
        self.assertEqual(apply_calibrator("uncalibrated",None,scores),scores)

    def test_isotonic_out_of_range_is_clipped(self):
        model=IsotonicRegression(out_of_bounds="clip").fit([.2,.4,.6,.8],[0,0,1,1])
        self.assertEqual(list(model.predict([-.1,1.1])),[0.0,1.0])

    def test_calibration_metrics_brier_logloss_and_ece(self):
        labels=[0,0,1,1]; scores=[.1,.2,.8,.9]
        metrics=calibration_metrics(labels,scores)
        self.assertAlmostEqual(metrics["brier_score"],.025)
        self.assertGreater(metrics["log_loss"],0); self.assertAlmostEqual(metrics["expected_calibration_error"],.15)

    def test_one_enters_last_bin_and_empty_bins_remain(self):
        bins=calibration_bins([0,1],[0.0,1.0])
        self.assertEqual(len(bins),10); self.assertEqual(bins[-1]["row_count"],1)
        self.assertTrue(any(row["row_count"]==0 for row in bins))

    def test_calibration_diagnostic_and_safe_clipping(self):
        intercept,slope,warning=calibration_intercept_slope([0,0,1,1],[0,.2,.8,1])
        self.assertIsNone(warning); self.assertIsNotNone(intercept); self.assertIsNotNone(slope)
        self.assertTrue(np.isfinite(_logits([0,1])).all())
        self.assertIsNotNone(calibration_intercept_slope([0,0],[.1,.2])[2])

    def test_threshold_boundaries_and_fixed_policy(self):
        candidates=threshold_candidates([.2,.8],[0,1])
        self.assertGreater(candidates[0]["threshold"],1); self.assertEqual(candidates[-1]["threshold"],0)
        selected=select_thresholds([.1,.4,.6,.9],[0,1,0,1],fixed=.5)
        self.assertEqual(selected["fixed"]["threshold"],.5)
        self.assertTrue(all(row["selection_dataset"]=="inner_oof_training" for row in selected.values()))

    def test_recall_f1_capacity_and_cost_policies(self):
        chosen=select_thresholds([.1,.2,.3,.4,.8,.9],[0,0,1,0,1,1],recall_target=.8,alert_capacity=.34)
        self.assertGreaterEqual(chosen["recall_focused"]["recall"],.8)
        self.assertLessEqual(chosen["alert_capacity"]["alert_rate"],.34)
        self.assertIn("f1_focused",chosen); self.assertIn("cost_sensitive",chosen)

    def test_threshold_selection_is_deterministic(self):
        args=([.1,.2,.3,.8,.9],[0,0,1,1,1])
        self.assertEqual(select_thresholds(*args),select_thresholds(*args))

    def test_nested_experiment_predictions_and_confusions(self):
        result=run_calibration_experiment(self.rows,[28],["uncalibrated","sigmoid","isotonic"],3,.5,.8,.1,5,1,42,"AAA","2013J")
        self.assertEqual(result.report["outer_model_fits"],1); self.assertEqual(result.report["inner_model_fits"],3)
        self.assertEqual(len(result.predictions),72)
        self.assertTrue(all(row["student_overlap_after_removal"]==0 for row in result.predictions))
        for row in result.holdout_results: self.assertEqual(sum(sum(x) for x in row["confusion_matrix"]),row["test_rows"])

    def test_outer_test_labels_do_not_change_training_selected_thresholds(self):
        first=run_calibration_experiment(self.rows,[28],["sigmoid"],3,.5,.8,.1,5,1,42,"AAA","2013J")
        changed=[dict(row) for row in self.rows]
        for row in changed:
            if row["module"]=="AAA" and row["presentation"]=="2013J": row["future_inactivity"]=1-int(row["future_inactivity"])
        second=run_calibration_experiment(changed,[28],["sigmoid"],3,.5,.8,.1,5,1,42,"AAA","2013J")
        self.assertEqual([r["selected_threshold"] for r in first.threshold_results],[r["selected_threshold"] for r in second.threshold_results])

    def test_repeated_experiment_is_deterministic(self):
        args=(self.rows,[28],["sigmoid"],3,.5,.8,.1,5,1,42,"AAA","2013J")
        self.assertEqual(run_calibration_experiment(*args).threshold_results,run_calibration_experiment(*args).threshold_results)

    def test_atomic_output_source_integrity_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/"source.parquet"
            with duckdb.connect() as connection: connection.execute("COPY (SELECT 1 x) TO ? (FORMAT PARQUET)",[str(source)])
            before=(source.stat().st_size,source.stat().st_mtime_ns)
            result=run_calibration_experiment(self.rows,[28],["uncalibrated","sigmoid","isotonic"],3,.5,.8,.1,5,1,42,"AAA","2013J")
            targets=[root/"results.json",root/"hold.csv",root/"pred.parquet",root/"curves.csv",root/"threshold.csv",root/"policies.json"]
            config={"observation_days":[28],"calibration_methods":["uncalibrated","sigmoid","isotonic"]}
            save_calibration_outputs(result,source,*targets,root/"figures",config)
            self.assertEqual(before,(source.stat().st_size,source.stat().st_mtime_ns)); self.assertTrue(all(path.exists() for path in targets))
            snapshots={path:path.read_bytes() for path in targets}
            with patch("learnpulse.calibration_experiment._publish",side_effect=RuntimeError("forced")):
                with self.assertRaises(RuntimeError): save_calibration_outputs(result,source,*targets,root/"figures",config)
            self.assertTrue(all(path.read_bytes()==snapshots[path] for path in targets))

    def test_worst_rates_use_denominators_not_larger_counts(self):
        def row(unit,fp,tn,fn,tp):
            return {"observation_day":28,"calibration_method":"sigmoid","threshold_policy":"fixed",
                    "outer_holdout_id":unit,"outer_test_recall":tp/(fn+tp),
                    "outer_test_false_positives":fp,"outer_test_true_negatives":tn,
                    "outer_test_false_negatives":fn,"outer_test_true_positives":tp}
        smaller=row("small",4,6,3,2)  # FPR .4, FNR .6
        larger=row("large",20,80,10,90)  # FPR .2, FNR .1
        results=worst_presentations([], [smaller,larger])
        by_measure={item["measure"]:item for item in results}
        self.assertEqual(by_measure["highest_false_positive_rate"]["holdout"],"small")
        self.assertAlmostEqual(by_measure["highest_false_positive_rate"]["value"],4/(4+6))
        self.assertEqual(by_measure["highest_false_negative_rate"]["holdout"],"small")
        self.assertAlmostEqual(by_measure["highest_false_negative_rate"]["value"],3/(3+2))
        self.assertEqual(by_measure["highest_false_positive_count"]["holdout"],"large")
        self.assertEqual(by_measure["highest_false_negative_count"]["holdout"],"large")

    def test_zero_rate_denominators_are_null_with_warnings(self):
        row={"observation_day":28,"calibration_method":"sigmoid","threshold_policy":"fixed",
             "outer_holdout_id":"empty","outer_test_recall":0,
             "outer_test_false_positives":0,"outer_test_true_negatives":0,
             "outer_test_false_negatives":0,"outer_test_true_positives":0}
        rates=[item for item in worst_presentations([], [row]) if item["measure"] in
               {"highest_false_positive_rate","highest_false_negative_rate"}]
        self.assertTrue(all(item["value"] is None and item.get("warning") for item in rates))

    def test_threshold_chart_has_four_metric_panels(self):
        self.assertEqual(THRESHOLD_CHART_METRICS,("precision","recall","f1","alert_rate"))
        predictions=[{"observation_day":28,"calibration_method":"sigmoid","actual_label":label,
                      "calibrated_output":score} for label,score in [(0,.1),(1,.8)]]
        curves=threshold_tradeoff_curves(predictions,[28],["sigmoid"])
        self.assertTrue(all(set(THRESHOLD_CHART_METRICS)<=set(row) for row in curves))
        with tempfile.TemporaryDirectory() as folder:
            with patch("learnpulse.calibration_experiment.plt.close") as close:
                _plot_thresholds(curves,28,Path(folder)/"chart.png")
                figure=close.call_args.args[0]
                self.assertEqual([axis.get_ylabel() for axis in figure.axes],
                                 ["Precision","Recall","F1","Alert rate"])

    def test_reporting_refresh_preserves_predictions_and_thresholds_without_refit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/"source.parquet"
            with duckdb.connect() as connection:
                connection.execute("COPY (SELECT 1 x) TO ? (FORMAT PARQUET)",[str(source)])
            result=run_calibration_experiment(self.rows,[28],["sigmoid"],3,.5,.8,.1,5,1,42,"AAA","2013J")
            paths=[root/name for name in ("results.json","holdouts.csv","predictions.parquet","curves.csv","thresholds.csv","policies.json")]
            save_calibration_outputs(result,source,*paths,root/"figures",{"observation_days":[28],"calibration_methods":["sigmoid"]})
            protected=(paths[1],paths[2],paths[4],source)
            before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
            with patch("learnpulse.calibration_experiment.build_logistic_pipeline",side_effect=AssertionError("refit attempted")):
                refreshed=refresh_saved_reporting(paths[0],paths[1],paths[4],paths[2],root/"figures")
            self.assertEqual(before,{path:hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})
            self.assertEqual(refreshed["reporting_correction"]["chart_metrics"],list(THRESHOLD_CHART_METRICS))


if __name__=="__main__": unittest.main()
