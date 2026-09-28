# LearnPulse Final Technical Report

## 1. Executive summary

LearnPulse is a reproducible research prototype for predicting a narrow behavioral outcome: no recorded OULAD virtual-learning-environment activity during the 14 course days after an observation. It evaluates whether changes from a learner's own prior activity add signal, while explicitly avoiding claims about motivation, psychological disengagement, causation or intervention benefit.

## 2–4. Problem, objectives, dataset and governance

The project processes 10,655,280 raw activity records into 1,808,119 daily summaries. The final multi-course table contains 107,147 eligible observations from 24,738 student identifiers across seven modules and 22 module-presentations. Observation checkpoints are Days 14, 28, 42 and 56. Raw OULAD is not redistributed; identifiers and demographics are excluded from scoring.

## 5. Scalable data pipeline

DuckDB joins activity and resource metadata, aggregates daily behavior, and writes module/presentation-partitioned Parquet. Processing is spill-controlled, deterministic at the logical level, and validated against the trusted Python implementation. Partition-by-partition modeling avoids loading 1.8M daily rows together in Python.

## 6–8. Features, personal baselines and outcome

Seven-day windows preserve exact distinct resources and activity types. Personal-change features compare a current complete window with earlier complete windows only. The target inspects Days t+1 through t+14 and stays separate from features. Course-end and pre-observation withdrawal rules define prediction eligibility.

## 9. Experimental design

Rows are grouped by student for within-course validation. Generalization experiments hold out entire presentations and modules, with a stricter variant removing training students appearing in the test course. Preprocessing is fitted only on training partitions. Metrics prioritize PR-AUC, recall, precision, F1 and Brier score rather than accuracy alone.

## 10–12. Baselines, ablation and timing

Dummy, rule and balanced-logistic baselines established minimum comparisons. Ablation compared absolute behavior, personal change and their combination using identical folds. Personal-change features improved several priority metrics. Day 28 was the earliest reasonably useful checkpoint in the original timing study; Day 56 was stronger on some metrics but leaves less support time. These findings remain uncertain because positive cases are comparatively sparse.

## 13. Cross-course generalization

The complete study executed strict unseen-presentation and unseen-module evaluations. Model B generalized better than absolute-only behavior on average, but performance varied materially by course. Macro, pooled and worst-group summaries prevent an average result from being presented as universal transfer.

## 14. Calibration

Nested calibration compared uncalibrated, sigmoid and isotonic outputs. Calibrators and thresholds used inner grouped OOF training scores only. Sigmoid was the preferred research method; outputs are still not trustworthy individual risk percentages. The F1-focused policy is a descriptive compromise, not an institutional policy.

## 15. Explainability

Fold-specific standardized coefficients, validation-only permutation importance and exact logistic contributions provide transparent associations. Correlated features can exchange apparent importance, and no coefficient is interpreted causally.

## 16. Fairness audit

Saved held-out sigmoid predictions were joined to audit-only learner attributes using the complete course/student key. Metrics use minimum-support suppression, student bootstrap intervals, course-adjusted sensitivity and limited intersections. Measured disparities do not establish discrimination or intrinsic group differences.

## 17. Temporal stability

Four checkpoint-specific models and future windows were analyzed without treating scores as one invariant longitudinal scale. Warning persistence, reversals, confirmation delays and inactivity-episode lead time were reported. Requiring confirmation can reduce false positives while also losing true positives and available support time.

## 18. Workload simulation

A retrospective queue simulation evaluated checkpoint review capacity, backlog, deduplication, cooldowns and label-free priority rules. Review selection does not imply help, prevention or intervention effect. Outcomes are evaluation-only and demographics never affect queue order.

## 19–20. Reference architecture and monitoring

A separate `research_reference_model` is trained on all eligible historical rows solely for reproducible inference demonstrations. Five student-grouped folds generate OOF scores for sigmoid calibration; the final base pipeline fits all rows. Hashes, a logical fingerprint and a local registry protect integrity. FastAPI and atomic batch scoring reject identifiers, demographics and future fields. Monitoring separates unlabeled schema/feature/score drift from aligned delayed-label performance.

## 21. Privacy and security

The local service logs no payloads, identifiers, opaque references or individual scores. It applies request limits and controlled errors. Raw data and restricted outputs are excluded from Docker and Git policy. Authentication and authorization are prerequisites for any network exposure.

## 22. Limitations

VLE inactivity is an incomplete behavioral proxy; OULAD is historical and may not represent present institutions; missingness and course composition matter; positive outcomes are uneven; scores are not causal; no intervention occurred; and Docker runtime validation was unavailable locally at release preparation.

## 23–25. Future research, reproducibility and conclusion

Future work should use participatory outcome design, a predeclared prospective shadow study, institutional governance, staff-workflow evidence and safeguarding review. Reproduction uses bounded dependencies, deterministic seeds, source signatures, atomic publication, miniature CI fixtures and documented commands. LearnPulse demonstrates a rigorous research and ML engineering workflow; the held-out studies—not the full-data reference artifact—remain the evidence about model behavior.
