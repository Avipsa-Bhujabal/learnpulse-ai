# Interview guide

## 60 seconds

LearnPulse is a research-only OULAD pipeline testing whether deviations from a student's own earlier activity predict no recorded VLE use in the next 14 course days. I built scalable DuckDB/Parquet preparation, leakage-safe features, student/course holdouts, calibration and audits, then packaged a distinct full-history model for local API demonstrations. The scientific evidence remains the held-out studies.

## Three minutes

Explain the 10.7M-to-1.8M aggregation, exact seven-day distinct counts, complete-window personal baselines, target boundary, grouped folds, unseen presentations/modules, nested sigmoid calibration, subgroup/temporal audits and workload simulation. Close with why the full-data inference artifact cannot create a fresh performance claim.

## Ten-minute walkthrough and likely questions

- **Architecture:** Why Parquet? Partition pruning avoids scanning unrelated presentations.
- **Data engineering:** How are failures handled? Staged outputs, validation and atomic replacement.
- **Leakage:** Why earlier complete windows only? The current/future row cannot enter its baseline.
- **Modeling:** Why logistic regression? Transparent, stable baseline appropriate to the research question.
- **Calibration:** Why inner OOF scores? Calibrators cannot see scores from rows used to fit their base model.
- **Fairness:** Were demographics used? Never as inputs; only post-hoc audit attributes with suppression.
- **Monitoring:** Does drift prove failure? No; multiple transparent indicators prompt investigation.
- **Failure modes:** Course heterogeneity, positive scarcity, missingness and alert burden.
- **Institutional data:** Add governance, access controls, prospective shadow evaluation and workflow studies.
- **Educational-technology question:** Does inactivity mean disengagement? No; it is an incomplete behavioral proxy.
