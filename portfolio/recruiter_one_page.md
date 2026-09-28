# LearnPulse — two-minute project summary

LearnPulse is a research-only learning-analytics and ML engineering project that predicts a precise behavioral outcome: no recorded OULAD VLE activity during the next 14 course days. It does not diagnose disengagement or claim intervention effects.

The pipeline aggregates 10,655,280 raw activity records into 1,808,119 daily records, then produces 107,147 eligible observations across 22 module-presentations and 24,738 students. DuckDB and partitioned Parquet keep processing scalable. Seven-day behavior is compared with each learner's earlier complete windows, with explicit temporal leakage controls.

Three logistic feature sets were evaluated with student grouping and held-out courses. Personal-change features generalized better than absolute-only behavior on several priority metrics, though performance varied substantially across courses. Nested sigmoid calibration, threshold trade-offs, transparent coefficients, subgroup auditing, temporal stability and a retrospective staffing simulation expose limitations rather than hiding them.

Engineering deliverables include 340 automated tests at this release checkpoint, a versioned and hash-validated reference artifact, strict FastAPI schemas, atomic batch scoring, drift monitoring, Streamlit aggregate views, Docker/CI definitions and privacy-aware logs. The reference model demonstrates inference architecture; it is not new external evidence or deployment approval.

Stack: Python, DuckDB, Parquet, scikit-learn, FastAPI, Streamlit, Joblib, Matplotlib, unittest, Docker and GitHub Actions.
