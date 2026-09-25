"""Read-only subgroup audit of strictly held-out sigmoid calibration predictions.

Demographics are audit-only. This module never fits models, calibrators, or thresholds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learnpulse.calibration_experiment import _publish as _publish_atomic
from learnpulse.calibration_experiment import calibration_bins

UNKNOWN = "Unknown / Not recorded"
ATTRIBUTES = ("gender", "region", "highest_education", "imd_band", "age_band",
              "disability", "num_of_prev_attempts", "studied_credits")
INTERSECTIONS = (("gender", "age_band"), ("gender", "disability"),
                 ("age_band", "disability"), ("highest_education", "age_band"))
POLICY_COLUMNS = {"f1_focused": "f1_focused_prediction", "fixed_0_50": "fixed_prediction",
                  "recall_focused": "recall_focused_prediction", "alert_capacity": "alert_capacity_prediction"}
POLICY_SOURCE_NAMES = {"fixed_0_50":"fixed", "f1_focused":"f1_focused",
                       "recall_focused":"recall_focused", "alert_capacity":"alert_capacity"}
AUDIT_ONLY_COLUMNS = tuple(ATTRIBUTES) + ("studied_credits_continuous", "module", "presentation",
    "student_id", "observation_day", "actual_label", "calibrated_output", *POLICY_COLUMNS.values())


@dataclass(frozen=True)
class AuditConfig:
    days: tuple[int, ...] = (28, 56)
    primary_policy: str = "f1_focused"
    sensitivity_policies: tuple[str, ...] = ("fixed_0_50", "recall_focused", "alert_capacity")
    bootstrap_samples: int = 1000
    minimum_rows: int = 50
    minimum_students: int = 25
    minimum_positives: int = 10
    minimum_negatives: int = 10
    minimum_predicted_positives: int = 10
    intersection_rows: int = 100
    intersection_students: int = 50
    intersection_positives: int = 15
    intersection_negatives: int = 15
    random_seed: int = 42
    only_attribute: str | None = None
    only_day: int | None = None
    only_module: str | None = None

    def validate(self) -> None:
        if not self.days or len(set(self.days)) != len(self.days) or any(day not in (28, 56) for day in self.days):
            raise ValueError("Observation days must be unique and selected from 28 and 56")
        if self.only_day is not None and self.only_day not in self.days:
            raise ValueError("only-observation-day must be among requested days")
        if self.only_attribute is not None and self.only_attribute not in ATTRIBUTES:
            raise ValueError("Unknown audit attribute")
        policies = (self.primary_policy, *self.sensitivity_policies)
        if self.primary_policy != "f1_focused" or len(set(policies)) != len(policies) or any(p not in POLICY_COLUMNS for p in policies):
            raise ValueError("Use F1-focused primary policy and distinct supported sensitivity policies")
        if any(value < 1 for value in (self.bootstrap_samples, self.minimum_rows, self.minimum_students,
                                      self.minimum_positives, self.minimum_negatives,
                                      self.minimum_predicted_positives, self.intersection_rows,
                                      self.intersection_students, self.intersection_positives,
                                      self.intersection_negatives)):
            raise ValueError("Bootstrap and support counts must be positive")
        if self.only_module is not None and (not self.only_module.isalpha() or len(self.only_module) != 3):
            raise ValueError("only-module must be a three-letter code")

    @property
    def filtered(self) -> bool:
        return self.only_attribute is not None or self.only_day is not None or self.only_module is not None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal(value: object) -> str:
    if value is None or str(value).strip() in ("", "?"):
        return UNKNOWN
    return str(value).strip()


def credit_boundaries(student_info: Path) -> list[float]:
    """Label-independent quartiles, deduplicated so ties never split a credit value."""
    with student_info.open(newline="", encoding="utf-8-sig") as handle:
        values = [float(row["studied_credits"]) for row in csv.DictReader(handle)
                  if _normal(row["studied_credits"]) != UNKNOWN]
    if not values:
        return []
    return sorted(set(float(x) for x in np.quantile(values, [.25, .5, .75])))


def credit_band(value: object, boundaries: Sequence[float]) -> str:
    if _normal(value) == UNKNOWN:
        return UNKNOWN
    number = float(value)
    lower = None
    for bound in boundaries:
        if number <= bound:
            return f"{_format_bound(lower)} < credits <= {_format_bound(bound)}" if lower is not None else f"credits <= {_format_bound(bound)}"
        lower = bound
    return f"credits > {_format_bound(lower)}" if lower is not None else "All recorded credits"


def _format_bound(value: float | None) -> str:
    return f"{value:g}" if value is not None else ""


def previous_attempt_band(value: object) -> str:
    if _normal(value) == UNKNOWN:
        return UNKNOWN
    number = int(value)
    if number < 0:
        raise ValueError("Negative previous-attempt count")
    return "0" if number == 0 else "1" if number == 1 else "2 or more"


def load_joined_predictions(predictions: Path, student_info: Path, config: AuditConfig) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate unique three-key demographic join without changing test predictions."""
    config.validate()
    if not predictions.is_file() or not student_info.is_file():
        raise FileNotFoundError("Calibration predictions and studentInfo.csv are required")
    boundaries = credit_boundaries(student_info)
    days = (config.only_day,) if config.only_day is not None else config.days
    with duckdb.connect() as connection:
        source_columns = {x[0] for x in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(predictions)]).fetchall()}
        required = {"module", "presentation", "student_id", "observation_day", "actual_label", "calibrated_output", "calibration_method", "outer_holdout_id", "student_overlap_after_removal", *POLICY_COLUMNS.values()}
        if not required <= source_columns:
            raise ValueError(f"Missing saved prediction columns: {sorted(required-source_columns)}")
        select_columns = ["module", "presentation", "student_id", "observation_day", "actual_label", "calibrated_output", "outer_holdout_id", "student_overlap_after_removal", *POLICY_COLUMNS.values()]
        placeholders = ",".join("?" for _ in days)
        where = f"calibration_method = 'sigmoid' AND observation_day IN ({placeholders})"
        parameters: list[object] = [str(predictions), *days]
        if config.only_module:
            where += " AND module = ?"; parameters.append(config.only_module)
        query = f"SELECT {','.join(select_columns)} FROM read_parquet(?) WHERE {where} ORDER BY observation_day,module,presentation,student_id"
        cursor = connection.execute(query, parameters)
        prediction_rows = [dict(zip(select_columns, row)) for row in cursor.fetchall()]
    if not prediction_rows:
        raise ValueError("No matching sigmoid predictions")
    prediction_keys = [(r["module"],r["presentation"],int(r["student_id"]),int(r["observation_day"])) for r in prediction_rows]
    if len(set(prediction_keys)) != len(prediction_keys):
        raise ValueError("Duplicate prediction key")
    demographics: dict[tuple[str,str,int], dict[str,str]] = {}
    with student_info.open(newline="", encoding="utf-8-sig") as handle:
        reader=csv.DictReader(handle)
        required_info={"code_module","code_presentation","id_student",*ATTRIBUTES}
        if not required_info <= set(reader.fieldnames or []):
            raise ValueError(f"Missing studentInfo columns: {sorted(required_info-set(reader.fieldnames or []))}")
        for raw in reader:
            key=(raw["code_module"],raw["code_presentation"],int(raw["id_student"]))
            if key in demographics:
                raise ValueError(f"Duplicate studentInfo key: {key}")
            demographics[key]=raw
    unmatched=0
    for row in prediction_rows:
        key=(str(row["module"]),str(row["presentation"]),int(row["student_id"]))
        info=demographics.get(key)
        unmatched += info is None
        for attribute in ATTRIBUTES:
            value=info.get(attribute) if info else None
            row[attribute] = (previous_attempt_band(value) if attribute=="num_of_prev_attempts" else
                              credit_band(value,boundaries) if attribute=="studied_credits" else _normal(value))
        row["studied_credits_continuous"] = float(info["studied_credits"]) if info and _normal(info["studied_credits"]) != UNKNOWN else None
        if int(row["student_overlap_after_removal"]) != 0:
            raise ValueError("Prediction is not student-disjoint")
        if row["outer_holdout_id"] != f"{row['module']}_{row['presentation']}":
            raise ValueError("Prediction is not from its held-out presentation")
        if int(row["actual_label"]) not in (0,1) or not 0 <= float(row["calibrated_output"]) <= 1:
            raise ValueError("Invalid saved label or score")
        if any(int(row[column]) not in (0,1) for column in POLICY_COLUMNS.values()):
            raise ValueError("Invalid saved policy prediction")
    if len(prediction_rows) != len(prediction_keys):
        raise AssertionError("Left join changed prediction row count")
    return prediction_rows, {"prediction_rows":len(prediction_rows),"joined_rows":len(prediction_rows),
        "unique_prediction_keys":len(set(prediction_keys)),"unmatched_demographic_rows":unmatched,
        "credit_quantile_boundaries":boundaries,"student_info_keys":len(demographics)}


def safe_rate(numerator: int, denominator: int) -> tuple[float | None,str | None]:
    return (numerator/denominator,None) if denominator else (None,"zero_denominator")


def validate_saved_thresholds(rows: Sequence[dict[str,object]], threshold_csv: Path,
                              policies: Sequence[str]) -> dict[str,object]:
    """Check saved policy labels against saved holdout thresholds; never select anew."""
    with threshold_csv.open(newline="",encoding="utf-8") as handle:
        records=list(csv.DictReader(handle))
    index={}
    for record in records:
        key=(int(record["observation_day"]),record["outer_holdout_id"],record["calibration_method"],record["threshold_policy"])
        if key in index: raise ValueError("Duplicate saved threshold key")
        index[key]=float(record["selected_threshold"])
    checked=0
    for row in rows:
        for policy in policies:
            source=POLICY_SOURCE_NAMES[policy]
            key=(int(row["observation_day"]),str(row["outer_holdout_id"]),"sigmoid",source)
            if key not in index: raise ValueError(f"Missing saved threshold: {key}")
            expected=int(float(row["calibrated_output"])>=index[key])
            if int(row[POLICY_COLUMNS[policy]])!=expected:
                raise ValueError(f"Saved prediction disagrees with saved threshold: {key}")
            checked+=1
    return {"threshold_keys":len(index),"prediction_policy_checks":checked,"all_matched":True}


def group_metrics(rows: Sequence[dict[str, object]], policy: str,
                  config: AuditConfig, intersection: bool = False) -> dict[str, object]:
    """Calculate metrics with null—not zero—for undefined or unsupported rates."""
    if policy not in POLICY_COLUMNS:
        raise ValueError("Unknown saved threshold policy")
    n=len(rows); students=len({int(row["student_id"]) for row in rows})
    labels=np.asarray([int(row["actual_label"]) for row in rows],dtype=int)
    predicted=np.asarray([int(row[POLICY_COLUMNS[policy]]) for row in rows],dtype=int)
    scores=np.asarray([float(row["calibrated_output"]) for row in rows],dtype=float)
    tp=int(np.sum((labels==1)&(predicted==1))); fp=int(np.sum((labels==0)&(predicted==1)))
    fn=int(np.sum((labels==1)&(predicted==0))); tn=int(np.sum((labels==0)&(predicted==0)))
    positives=tp+fn; negatives=fp+tn; warned=tp+fp
    raw: dict[str,float | None]={}
    reasons: dict[str,str] = {}
    fractions={"outcome_prevalence":(positives,n),"warning_rate":(warned,n),
        "recall":(tp,positives),"false_negative_rate":(fn,positives),
        "specificity":(tn,negatives),"false_positive_rate":(fp,negatives),
        "precision":(tp,warned),"negative_predictive_value":(tn,tn+fn)}
    for name,(num,den) in fractions.items():
        raw[name],reason=safe_rate(num,den)
        if reason: reasons[name]=reason
    raw["f1"] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    raw["balanced_accuracy"] = (raw["recall"]+raw["specificity"])/2 if raw["recall"] is not None and raw["specificity"] is not None else None
    raw["mean_calibrated_output"] = float(np.mean(scores)) if n else None
    raw["brier_score"] = float(np.mean((scores-labels)**2)) if n else None
    bins=calibration_bins(labels.tolist(),scores.tolist()) if n else []
    raw["expected_calibration_error"] = sum(int(b["row_count"])/n*float(b["absolute_calibration_gap"]) for b in bins if b["row_count"]) if n else None
    min_rows=config.intersection_rows if intersection else config.minimum_rows
    min_students=config.intersection_students if intersection else config.minimum_students
    min_pos=config.intersection_positives if intersection else config.minimum_positives
    min_neg=config.intersection_negatives if intersection else config.minimum_negatives
    general=[]
    if n<min_rows: general.append(f"rows<{min_rows}")
    if students<min_students: general.append(f"students<{min_students}")
    for name in raw:
        cause=list(general)
        if name in ("recall","false_negative_rate") and positives<min_pos: cause.append(f"positives<{min_pos}")
        if name in ("specificity","false_positive_rate") and negatives<min_neg: cause.append(f"negatives<{min_neg}")
        if name=="precision" and warned<config.minimum_predicted_positives: cause.append(f"predicted_positives<{config.minimum_predicted_positives}")
        if name in ("brier_score","expected_calibration_error") and (positives==0 or negatives==0): cause.append("calibration_requires_both_classes")
        if name=="balanced_accuracy" and (positives<min_pos or negatives<min_neg): cause.append("class_support_insufficient")
        if raw[name] is None: cause.append(reasons.get(name,"undefined"))
        if cause:
            reasons[name]=";".join(dict.fromkeys(cause)); raw[name]=None
    return {"rows":n,"unique_students":students,"positive_cases":positives,"negative_cases":negatives,
        "warning_count":warned,"true_positives":tp,"false_positives":fp,"false_negatives":fn,
        "true_negatives":tn,**raw,"suppression_reasons":reasons,
        "supported":not general,"calibration_bins":bins}


DISPARITY_METRICS = ("warning_rate","recall","false_negative_rate","false_positive_rate",
                     "precision","brier_score","expected_calibration_error")
BOOTSTRAP_METRICS = ("outcome_prevalence","warning_rate","recall","false_positive_rate",
                     "precision","brier_score")


def _group_values(rows: Sequence[dict[str, object]], attribute: str) -> dict[str,list[dict[str, object]]]:
    grouped: dict[str,list[dict[str,object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[attribute])].append(row)
    return grouped


def reference_group(metrics: Sequence[dict[str, object]]) -> str | None:
    """Largest generally supported group; lexical tie-break, no normative status."""
    supported=[row for row in metrics if row["supported"]]
    return sorted(supported,key=lambda r:(-int(r["rows"]),str(r["group"]))) [0]["group"] if supported else None


def disparity_rows(metrics: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Group-minus-reference and max-minus-min ranges, preserving nulls."""
    output=[]
    contexts=sorted({(r["observation_day"],r["threshold_policy"],r["attribute"]) for r in metrics})
    for day,policy,attribute in contexts:
        selected=[r for r in metrics if (r["observation_day"],r["threshold_policy"],r["attribute"])==(day,policy,attribute)]
        reference=reference_group(selected)
        index={r["group"]:r for r in selected}
        for metric in DISPARITY_METRICS:
            values=[(r["group"],float(r[metric])) for r in selected if r.get(metric) is not None]
            span=max((v for _,v in values),default=None)
            low=min((v for _,v in values),default=None)
            positive=[v for _,v in values if v>0]
            output.append({"analysis_type":"pairwise_range","observation_day":day,"threshold_policy":policy,
                "attribute":attribute,"group":None,"reference_group":reference,"metric":metric,
                "value":span-low if span is not None and low is not None and len(values)>1 else None,
                "max_min_ratio":span/min(positive) if span is not None and len(positive)==len(values) and len(values)>1 else None,
                "supported_groups":len(values)})
            for row in selected:
                ref=index.get(reference) if reference else None
                own=row.get(metric); base=ref.get(metric) if ref else None
                diff=float(own)-float(base) if own is not None and base is not None else None
                output.append({"analysis_type":"group_minus_reference","observation_day":day,
                    "threshold_policy":policy,"attribute":attribute,"group":row["group"],
                    "reference_group":reference,"metric":metric,"value":diff,
                    "group_value":own,"reference_value":base,
                    "max_min_ratio":float(own)/float(base) if own is not None and base is not None and float(own)>0 and float(base)>0 else None,
                    "suppression_reason":row["suppression_reasons"].get(metric) if own is None else None})
        for row in selected:
            ref=index.get(reference) if reference else None
            recall=None if ref is None or row["recall"] is None or ref["recall"] is None else float(row["recall"])-float(ref["recall"])
            fpr=None if ref is None or row["false_positive_rate"] is None or ref["false_positive_rate"] is None else float(row["false_positive_rate"])-float(ref["false_positive_rate"])
            output.append({"analysis_type":"equalized_odds_diagnostic","observation_day":day,
                "threshold_policy":policy,"attribute":attribute,"group":row["group"],
                "reference_group":reference,"metric":"max_abs_recall_or_fpr_difference",
                "value":max(abs(recall),abs(fpr)) if recall is not None and fpr is not None else None})
    return output


def _student_arrays(rows: Sequence[dict[str, object]], policy: str) -> np.ndarray:
    """One summary per student; all their rows travel together in a bootstrap draw."""
    groups: dict[int,list[dict[str,object]]] = defaultdict(list)
    for row in rows: groups[int(row["student_id"])].append(row)
    summaries=[]
    for student in sorted(groups):
        group=groups[student]
        y=np.asarray([int(x["actual_label"]) for x in group]); pred=np.asarray([int(x[POLICY_COLUMNS[policy]]) for x in group]); score=np.asarray([float(x["calibrated_output"]) for x in group])
        summaries.append([len(group),int(y.sum()),int(pred.sum()),int(((y==1)&(pred==1)).sum()),
            int(((y==0)&(pred==1)).sum()),int(((y==1)&(pred==0)).sum()),int(((y==0)&(pred==0)).sum()),float(((score-y)**2).sum())])
    return np.asarray(summaries,dtype=float).reshape(-1,8)


def bootstrap_values(rows: Sequence[dict[str, object]], policy: str, samples: int,
                     seed: int) -> dict[str,np.ndarray]:
    """Student-cluster resampling with replacement and preserved multiplicity."""
    data=_student_arrays(rows,policy)
    if len(data)==0:
        return {name:np.full(samples,np.nan) for name in BOOTSTRAP_METRICS}
    generator=np.random.default_rng(seed)
    output={name:np.empty(samples,dtype=float) for name in BOOTSTRAP_METRICS}
    for start in range(0,samples,100):
        size=min(100,samples-start)
        draw=generator.integers(0,len(data),size=(size,len(data)))
        counts=data[draw].sum(axis=1)
        n,pos,warn,tp,fp,fn,tn,brier=(counts[:,i] for i in range(8))
        vectors={"outcome_prevalence":(pos,n),"warning_rate":(warn,n),
            "recall":(tp,pos),"false_positive_rate":(fp,fp+tn),
            "precision":(tp,tp+fp),"brier_score":(brier,n)}
        for name,(numerator,denominator) in vectors.items():
            with np.errstate(divide="ignore",invalid="ignore"):
                output[name][start:start+size]=np.where(denominator>0,numerator/denominator,np.nan)
    return output


def _seed(seed: int, *values: object) -> int:
    payload="|".join(map(str,(seed,*values))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"big")


def _interval(values: np.ndarray) -> dict[str,object]:
    valid=values[np.isfinite(values)]
    return {"lower_95":float(np.quantile(valid,.025)) if len(valid) else None,
        "upper_95":float(np.quantile(valid,.975)) if len(valid) else None,
        "valid_samples":len(valid),"skipped_samples":int(len(values)-len(valid))}


def bootstrap_intervals(rows: Sequence[dict[str, object]], metrics: Sequence[dict[str, object]],
                        config: AuditConfig) -> list[dict[str,object]]:
    """Intervals for supported groups and F1-policy group-reference differences."""
    output=[]; cache={}
    member_index: dict[tuple[int,str,str],list[dict[str,object]]] = defaultdict(list)
    attributes=(config.only_attribute,) if config.only_attribute else ATTRIBUTES
    for row in rows:
        for attribute in attributes:
            member_index[(int(row["observation_day"]),attribute,str(row[attribute]))].append(row)
    for metric_row in metrics:
        if not metric_row["supported"] or metric_row["threshold_policy"]!=config.primary_policy: continue
        day=metric_row["observation_day"]; policy=metric_row["threshold_policy"]
        attr=metric_row["attribute"]; group=metric_row["group"]
        members=member_index[(day,attr,group)]
        values=bootstrap_values(members,policy,config.bootstrap_samples,_seed(config.random_seed,day,policy,attr,group))
        cache[(day,policy,attr,group)]=values
        for name in BOOTSTRAP_METRICS:
            if metric_row.get(name) is None: continue
            output.append({"interval_type":"group","observation_day":day,"threshold_policy":policy,
                "attribute":attr,"group":group,"reference_group":None,"metric":name,
                **_interval(values[name])})
    contexts=sorted({(r["observation_day"],r["attribute"]) for r in metrics})
    for day,attr in contexts:
        selected=[r for r in metrics if r["observation_day"]==day and r["attribute"]==attr and r["threshold_policy"]==config.primary_policy]
        reference=reference_group(selected)
        if reference is None: continue
        base=cache.get((day,config.primary_policy,attr,reference))
        if base is None: continue
        for row in selected:
            group=row["group"]
            if group==reference: continue
            current=cache.get((day,config.primary_policy,attr,group))
            if current is None: continue
            for name in ("recall","false_positive_rate","warning_rate","brier_score"):
                if row.get(name) is None or next(r for r in selected if r["group"]==reference).get(name) is None: continue
                output.append({"interval_type":"group_minus_reference","observation_day":day,
                    "threshold_policy":config.primary_policy,"attribute":attr,"group":group,
                    "reference_group":reference,"metric":name,**_interval(current[name]-base[name])})
    return output


def course_adjusted(rows: Sequence[dict[str, object]], metrics: Sequence[dict[str, object]],
                    config: AuditConfig) -> list[dict[str,object]]:
    """Aggregate supported within-presentation group-reference differences."""
    output=[]
    by_context: dict[tuple[int,str,str,str],list[dict[str,object]]] = defaultdict(list)
    for row in rows:
        if config.only_attribute:
            attributes=(config.only_attribute,)
        else:
            attributes=ATTRIBUTES
        for attr in attributes:
            by_context[(int(row["observation_day"]),str(row["module"]),str(row["presentation"]),attr)].append(row)
    references={}
    for metric in metrics:
        key=(metric["observation_day"],metric["attribute"],metric["threshold_policy"])
        references.setdefault(key,[]).append(metric)
    for (day,module,presentation,attribute),members in sorted(by_context.items()):
        for policy in (config.primary_policy,):
            reference=reference_group(references.get((day,attribute,policy),[]))
            if reference is None: continue
            grouped=_group_values(members,attribute)
            ref_rows=grouped.get(reference,[])
            if not ref_rows: continue
            reference_metrics=group_metrics(ref_rows,policy,config)
            for group,group_rows in grouped.items():
                if group==reference: continue
                current=group_metrics(group_rows,policy,config)
                for name in ("warning_rate","recall","false_positive_rate","brier_score"):
                    a=current[name]; b=reference_metrics[name]
                    if a is None or b is None: continue
                    output.append({"observation_day":day,"attribute":attribute,"group":group,
                        "reference_group":reference,"module":module,"presentation":presentation,
                        "metric":name,"difference":float(a)-float(b),"weight":len(group_rows)+len(ref_rows)})
    summary=[]
    keys=sorted({(r["observation_day"],r["attribute"],r["group"],r["metric"]) for r in output})
    for day,attr,group,name in keys:
        selected=[r for r in output if (r["observation_day"],r["attribute"],r["group"],r["metric"])==(day,attr,group,name)]
        values=[r["difference"] for r in selected]; weights=[r["weight"] for r in selected]
        summary.append({"observation_day":day,"attribute":attr,"group":group,"metric":name,
            "reference_group":selected[0]["reference_group"],"contributing_presentations":len(selected),
            "equal_presentation_difference":statistics.fmean(values),
            "row_weighted_difference":sum(v*w for v,w in zip(values,weights))/sum(weights)})
    return summary


def intersection_metrics(rows: Sequence[dict[str,object]], config: AuditConfig) -> list[dict[str,object]]:
    if config.only_attribute: return []
    output=[]
    for day in sorted(set(int(row["observation_day"]) for row in rows)):
        day_rows=[row for row in rows if row["observation_day"]==day]
        for left,right in INTERSECTIONS:
            groups: dict[tuple[str,str],list[dict[str,object]]] = defaultdict(list)
            for row in day_rows: groups[(str(row[left]),str(row[right]))].append(row)
            for (a,b),members in sorted(groups.items()):
                metrics=group_metrics(members,config.primary_policy,config,intersection=True)
                metrics.pop("calibration_bins")
                output.append({"observation_day":day,"intersection":f"{left} x {right}",
                    "group":f"{a} | {b}","left_group":a,"right_group":b,
                    "threshold_policy":config.primary_policy,**metrics})
    return output


def missingness(rows: Sequence[dict[str,object]], config: AuditConfig) -> list[dict[str,object]]:
    output=[]
    attributes=(config.only_attribute,) if config.only_attribute else ATTRIBUTES
    for day in sorted(set(int(row["observation_day"]) for row in rows)):
        day_rows=[row for row in rows if row["observation_day"]==day]
        for attr in attributes:
            missing=[row for row in day_rows if row[attr]==UNKNOWN]
            recorded=[row for row in day_rows if row[attr]!=UNKNOWN]
            for status,members in (("missing",missing),("recorded",recorded)):
                m=group_metrics(members,config.primary_policy,config)
                output.append({"observation_day":day,"attribute":attr,"recording_status":status,
                    "missing_count":len(missing),"missing_rate":len(missing)/len(day_rows),
                    **{name:m[name] for name in ("rows","unique_students","outcome_prevalence","warning_rate","recall","false_positive_rate")},
                    "suppression_reasons":m["suppression_reasons"]})
    return output


def policy_sensitivity(disparities: Sequence[dict[str,object]], config: AuditConfig) -> list[dict[str,object]]:
    primary={(r["observation_day"],r["attribute"],r["group"],r["metric"]):r for r in disparities
             if r["analysis_type"]=="group_minus_reference" and r["threshold_policy"]==config.primary_policy}
    output=[]
    for row in disparities:
        if row["analysis_type"]!="group_minus_reference" or row["threshold_policy"]==config.primary_policy: continue
        key=(row["observation_day"],row["attribute"],row["group"],row["metric"])
        earlier=primary.get(key)
        if earlier is None: continue
        a=earlier["value"]; b=row["value"]
        status="unsupported" if a is None or b is None else "reverses" if a*b<0 else "same_direction" if a*b>0 else "zero_or_boundary"
        if status=="same_direction" and abs(b)>2*abs(a): status="same_direction_magnitude_more_than_doubled"
        output.append({"observation_day":row["observation_day"],"attribute":row["attribute"],
            "group":row["group"],"metric":row["metric"],"sensitivity_policy":row["threshold_policy"],
            "primary_difference":a,"sensitivity_difference":b,"direction_status":status})
    return output


def run_audit(rows: Sequence[dict[str,object]], config: AuditConfig) -> dict[str,object]:
    config.validate()
    attributes=(config.only_attribute,) if config.only_attribute else ATTRIBUTES
    policies=(config.primary_policy,*config.sensitivity_policies)
    indexed: dict[tuple[int,str],list[dict[str,object]]] = defaultdict(list)
    for row in rows: indexed[(int(row["observation_day"]),str(row["module"]))].append(row)
    group_rows=[]; calibration_rows=[]
    for day in sorted(set(int(row["observation_day"]) for row in rows)):
        day_rows=[row for (d,_),members in indexed.items() if d==day for row in members]
        for policy in policies:
            for attr in attributes:
                for group,members in sorted(_group_values(day_rows,attr).items()):
                    result=group_metrics(members,policy,config)
                    bins=result.pop("calibration_bins")
                    group_rows.append({"observation_day":day,"threshold_policy":policy,
                        "attribute":attr,"group":group,**result})
                    if policy==config.primary_policy and result["expected_calibration_error"] is not None:
                        for bin_ in bins:
                            calibration_rows.append({"observation_day":day,"attribute":attr,
                                "group":group,**bin_})
    disparities=disparity_rows(group_rows)
    intervals=bootstrap_intervals(rows,group_rows,config)
    adjusted=course_adjusted(rows,group_rows,config)
    intersections=intersection_metrics(rows,config)
    missing=missingness(rows,config)
    sensitivity=policy_sensitivity(disparities,config)
    return {"group_metrics":group_rows,"disparities":disparities,"bootstrap_intervals":intervals,
        "course_adjusted":adjusted,"intersections":intersections,"missingness":missing,
        "policy_sensitivity":sensitivity,"calibration_bins":calibration_rows}


def _csv_value(value: object) -> object:
    return json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value


def _write_csv(path: Path, rows: Sequence[dict[str,object]]) -> None:
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:_csv_value(row.get(key)) for key in columns})


def _write_joined_parquet(path: Path, rows: Sequence[dict[str,object]]) -> None:
    temporary=path.with_suffix(".csv")
    _write_csv(temporary,rows)
    destination=path.resolve().as_posix().replace("'","''")
    with duckdb.connect() as connection:
        connection.execute(f"COPY (SELECT * FROM read_csv_auto(?,header=true)) TO '{destination}' (FORMAT PARQUET,COMPRESSION ZSTD)",
                           [str(temporary)])
    temporary.unlink()


def _interval_index(intervals: Sequence[dict[str,object]]) -> dict[tuple[int,str,str,str],dict[str,object]]:
    return {(r["observation_day"],r["attribute"],r["group"],r["metric"]):r for r in intervals
            if r["interval_type"]=="group"}


def _plot_metric(groups: Sequence[dict[str,object]], intervals: Sequence[dict[str,object]],
                 day: int, metric: str, path: Path, attributes: Sequence[str]) -> None:
    figure,axes=plt.subplots(math.ceil(len(attributes)/2),2,figsize=(16,max(5,4*math.ceil(len(attributes)/2))))
    flat=np.asarray(axes).flatten(); lookup=_interval_index(intervals)
    for axis,attribute in zip(flat,attributes):
        selected=sorted((r for r in groups if r["observation_day"]==day and r["threshold_policy"]=="f1_focused"
                         and r["attribute"]==attribute and r.get(metric) is not None),key=lambda r:str(r["group"]))
        names=[str(r["group"]) for r in selected]; values=[float(r[metric]) for r in selected]
        y=np.arange(len(names))
        axis.barh(y,values,color="#376996")
        axis.set_yticks(y,names,fontsize=8); axis.invert_yaxis(); axis.set_xlim(0,1)
        axis.set_title(attribute.replace("_"," ")); axis.set_xlabel(metric.replace("_"," "))
        for pos,row in enumerate(selected):
            ci=lookup.get((day,attribute,row["group"],metric))
            if ci and ci["lower_95"] is not None:
                axis.errorbar([values[pos]],[pos],xerr=[[max(0,values[pos]-ci["lower_95"])],[max(0,ci["upper_95"]-values[pos])]],fmt="none",ecolor="#dc8a37",capsize=3)
        if not selected: axis.text(.5,.5,"No supported groups",ha="center",va="center",transform=axis.transAxes)
    for axis in flat[len(attributes):]: axis.axis("off")
    figure.suptitle(f"Day {day} · sigmoid calibration · F1-focused warning · {metric.replace('_',' ')}\n95% student-bootstrap intervals; suppressed groups omitted; descriptive only")
    figure.tight_layout(rect=[0,0,1,.96]); figure.savefig(path,dpi=140); plt.close(figure)


def _plot_calibration_bins(bins: Sequence[dict[str,object]], path: Path) -> None:
    figure,axes=plt.subplots(1,2,figsize=(13,5))
    for axis,day in zip(axes,(28,56)):
        axis.plot([0,1],[0,1],"--",color="#555555",label="perfect calibration")
        for attribute in ("gender","disability"):
            groups=sorted({r["group"] for r in bins if r["observation_day"]==day and r["attribute"]==attribute})
            for group in groups:
                selected=sorted((r for r in bins if r["observation_day"]==day and r["attribute"]==attribute
                    and r["group"]==group and r["row_count"]),key=lambda r:r["bin_lower"])
                if selected:
                    axis.plot([r["mean_predicted_value"] for r in selected],
                              [r["observed_positive_rate"] for r in selected],marker="o",markersize=3,
                              label=f"{attribute}: {group}")
        axis.set(xlim=(0,1),ylim=(0,1),xlabel="Sigmoid-calibrated output",
                 ylabel="Observed future-inactivity frequency",title=f"Day {day} · supported groups only")
        axis.legend(fontsize=7)
    figure.suptitle("Subgroup calibration · fixed-width bins · descriptive, not causal")
    figure.tight_layout(); figure.savefig(path,dpi=150); plt.close(figure)


def save_audit(audit: dict[str,object], rows: Sequence[dict[str,object]], join_validation: dict[str,object],
               config: AuditConfig, sources: dict[str,Path], targets: dict[str,Path]) -> dict[str,object]:
    """Stage and publish only new audit outputs, restoring old outputs on failure."""
    required=("results","group_metrics","disparities","intersections","bootstrap_intervals","joined_predictions","figures_dir")
    if not set(required)<=set(targets): raise ValueError("Missing audit output path")
    figures=[targets["figures_dir"]/f"day{day}_subgroup_{metric}.png" for day in (28,56)
             for metric in ("recall","false_positive_rate")]
    figures.append(targets["figures_dir"]/"subgroup_calibration.png")
    outputs=[targets[name] for name in required if name!="figures_dir"]+figures
    if len(set(path.resolve() for path in outputs))!=len(outputs): raise ValueError("Audit output paths must be distinct")
    source_hashes={name:sha256(path) for name,path in sources.items()}
    started=time.perf_counter()
    summary={"configuration":{**config.__dict__,"filtered_run":config.filtered},"source_sha256":source_hashes,
        "join_validation":join_validation,"prediction_rows":len(rows),
        "group_rows":len(audit["group_metrics"]),"supported_groups":sum(r["supported"] for r in audit["group_metrics"] if r["threshold_policy"]==config.primary_policy),
        "suppressed_groups":sum(not r["supported"] for r in audit["group_metrics"] if r["threshold_policy"]==config.primary_policy),
        "attributes_examined":1 if config.only_attribute else len(ATTRIBUTES),
        "intersections_examined":len(INTERSECTIONS) if not config.only_attribute else 0,
        "suppressed_intersections":sum(not r["supported"] for r in audit["intersections"]),
        "course_adjusted":audit["course_adjusted"],"missingness":audit["missingness"],
        "policy_sensitivity":audit["policy_sensitivity"],"calibration_bins":audit["calibration_bins"],
        "warnings":["Descriptive disparities are not evidence of cause or inherent learner risk.",
                    "Many comparisons and small groups require cautious interpretation."]}
    targets["results"].parent.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix=".fairness_audit_",dir=targets["results"].parent))
    try:
        staged=[root/f"{i}_{path.name}" for i,path in enumerate(outputs)]
        staged_map=dict(zip(required[:-1],staged[:6]))
        _write_csv(staged_map["group_metrics"],audit["group_metrics"])
        _write_csv(staged_map["disparities"],audit["disparities"])
        _write_csv(staged_map["intersections"],audit["intersections"])
        _write_csv(staged_map["bootstrap_intervals"],audit["bootstrap_intervals"])
        _write_joined_parquet(staged_map["joined_predictions"],rows)
        attributes=(config.only_attribute,) if config.only_attribute else ATTRIBUTES
        for index,day in enumerate((28,56)):
            for offset,metric in enumerate(("recall","false_positive_rate")):
                _plot_metric(audit["group_metrics"],audit["bootstrap_intervals"],day,metric,staged[6+index*2+offset],attributes)
        _plot_calibration_bins(audit["calibration_bins"],staged[10])
        summary["output_build_seconds"]=time.perf_counter()-started
        summary["source_integrity"]={name:sha256(path)==source_hashes[name] for name,path in sources.items()}
        if not all(summary["source_integrity"].values()): raise RuntimeError("Source changed during audit")
        staged_map["results"].write_text(json.dumps(summary,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
        _publish_atomic(list(zip(staged,outputs)))
    finally:
        if root.exists(): shutil.rmtree(root)
    return summary
