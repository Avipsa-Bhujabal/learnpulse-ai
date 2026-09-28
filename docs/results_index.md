# Results index

| Question | Method | Input | Output | Main metric/finding | Key limitation | Documentation |
|---|---|---|---|---|---|---|
| Do personal changes add signal? | Grouped ablation | Modeling table | `experiments/ablation_results.json` | Model B improved several rare-outcome metrics | One dataset | [Within-person study](../research/within_person_experiment.md) |
| How early is signal visible? | Day-specific grouped validation | Modeling table | `experiments/early_warning_results.json` | Day 28 was the earliest reasonably useful checkpoint | Few positives | [Early warning](../research/early_warning_experiment.md) |
| Is timing comparison cohort-sensitive? | Four-day common cohort | Modeling table | `experiments/common_cohort_results.json` | Same-student sensitivity analysis | Reduced cohort | [Common cohort](../research/common_cohort_sensitivity.md) |
| What influences scores? | OOF coefficients and permutation | Common cohort | `experiments/explainability_results.json` | Transparent associations, not causes | Correlated features | [Explainability](../research/explainability_methodology.md) |
| Does performance transfer? | Held-out presentations/modules | Multi-course table | `experiments/crosscourse_results.json` | Personal-change features generalized better on average | Large course variation | [Generalization](../research/crosscourse_generalization.md) |
| Are scores calibrated? | Nested held-out calibration | Multi-course table | `experiments/calibration_results.json` | Sigmoid was preferred for research reporting | Not deployment calibration | [Calibration](../research/calibration_and_thresholds.md) |
| Which thresholds trade recall for alerts? | Training-selected policies | Calibration OOF scores | `experiments/threshold_results.csv` | F1 policy was a descriptive compromise | No institutional costs | [Calibration](../research/calibration_and_thresholds.md) |
| Do subgroup errors differ? | Held-out subgroup audit | Saved predictions | `experiments/fairness_results.json` | Differences require support and uncertainty context | Observational audit | [Fairness](../research/fairness_and_subgroup_audit.md) |
| Are warnings stable? | Four-checkpoint trajectories | Saved predictions | `experiments/temporal_stability_results.json` | Warnings can persist, reverse or recur | Separate day models | [Temporal stability](../research/temporal_alert_stability.md) |
| What does review capacity change? | Retrospective queue simulation | Temporal artifacts | `experiments/workload_simulation_results.json` | Capacity and confirmation trade capture for workload | No intervention effects | [Workload](../research/operational_workload_simulation.md) |
| Can inference be reproduced? | Versioned full-data reference artifact | Multi-course table | `artifacts/reference_model/model_manifest.json` | Hash-validated local scoring architecture | No new external evidence | [Architecture](architecture.md) |
