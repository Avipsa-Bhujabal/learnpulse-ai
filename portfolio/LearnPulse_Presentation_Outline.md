# LearnPulse 12-slide portfolio presentation

1. **Problem** — next-14-day recorded inactivity; behavioral proxy; research-only. Visual: outcome timeline. Notes: avoid psychological claims. Source: `research/outcome_definition.md`.
2. **Research gap** — absolute versus within-person behavior; course transfer; human review. Visual: baseline comparison. Source: `research/within_person_experiment.md`.
3. **Dataset** — 10.7M raw, 1.8M daily, 22 presentations. Visual: scale funnel. Source: `experiments/scalable_pipeline_report.json`.
4. **Pipeline** — DuckDB, Parquet partitions, feature/outcome separation. Visual: architecture. Source: `docs/architecture.md`.
5. **Personal baseline** — earlier complete windows only; ten Model B fields. Visual: seven-day windows. Source: artifact manifest.
6. **Evaluation** — grouped students, held-out presentations/modules, strict overlap removal. Visual: holdout diagram. Source: cross-course report.
7. **Main results** — personal-change model improved several priority metrics but varied by course. Visual: cross-course summary. Source: cross-course results.
8. **Generalization and calibration** — sigmoid research calibration; no probability guarantee. Visual: calibration curve. Source: calibration results.
9. **Fairness and temporal stability** — support suppression; warnings can reverse. Visual: aggregate audit charts. Sources: fairness and temporal reports.
10. **Workload simulation** — capacity changes review access; no intervention effect. Visual: capacity trade-off. Source: workload report.
11. **Engineering** — reference registry, FastAPI, monitoring, CI, 340 tests. Visual: service architecture. Sources: manifest and test report.
12. **Limitations and next research** — proxy outcome, heterogeneity, governance and prospective shadow study. Visual: research roadmap. Source: methodology documents.
