"""Aggregate-only local dashboard for the Step 16 research simulation."""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="LearnPulse workload simulation",layout="wide")
st.warning("Research simulation only. This dashboard does not estimate intervention effects and must not be used for automatic learner decisions.")
root=Path(__file__).resolve().parents[1]/"experiments"
scenario_path=root/"workload_scenario_results.csv";checkpoint_path=root/"workload_checkpoint_results.csv";course_path=root/"workload_course_results.csv";subgroup_path=root/"workload_subgroup_results.csv"
required=[scenario_path,checkpoint_path,course_path,subgroup_path]
if not all(x.exists() for x in required):st.error("Run the workload simulation first.");st.stop()
scenarios=pd.read_csv(scenario_path);checkpoints=pd.read_csv(checkpoint_path);courses=pd.read_csv(course_path);subgroups=pd.read_csv(subgroup_path)
st.sidebar.header("Aggregate filters")
day=st.sidebar.selectbox("Observation day",sorted(checkpoints.observation_day.unique()))
course=st.sidebar.selectbox("Module-presentation",["All",*sorted((checkpoints.module+" "+checkpoints.presentation).unique())])
threshold=st.sidebar.selectbox("Threshold policy",sorted(checkpoints.threshold_policy.unique()))
capacity=st.sidebar.selectbox("Capacity",sorted(checkpoints.capacity_scenario.unique()))
dedup=st.sidebar.selectbox("Deduplication rule",sorted(checkpoints.deduplication_scenario.unique()))
priority=st.sidebar.selectbox("Prioritization policy",sorted(checkpoints.priority_policy.unique()))
f=checkpoints[(checkpoints.observation_day==day)&(checkpoints.threshold_policy==threshold)&(checkpoints.capacity_scenario==capacity)&(checkpoints.deduplication_scenario==dedup)&(checkpoints.priority_policy==priority)]
if course!="All":m,p=course.split(" ",1);f=f[(f.module==m)&(f.presentation==p)]
cols=st.columns(8);values=[("Alerts",f.raw_alerts_generated.sum()),("Reviews",f.reviews_completed.sum()),("Capacity utilization",f.capacity_utilization.mean()),("Remaining backlog",f.queue_size_after_review.sum()),("Median delay",f.median_review_delay.median()),("Future-inactive capture","See scenario table"),("Capacity misses","See scenario table"),("Episode-before-start","See scenario table")]
for col,(name,value) in zip(cols,values):col.metric(name,value if isinstance(value,str) else f"{value:.3g}")
st.subheader("Capacity versus capture");s=scenarios[(scenarios.threshold_policy==threshold)&(scenarios.deduplication_scenario==dedup)&(scenarios.priority_policy==priority)];st.line_chart(s.set_index("capacity_scenario")[["fraction_all_future_inactive_reviewed"]])
st.subheader("Backlog by checkpoint");st.line_chart(f.groupby("observation_day").queue_size_after_review.sum())
st.subheader("Course comparison");st.dataframe(f.groupby(["module","presentation"])[["raw_alerts_generated","reviews_completed","queue_size_after_review","median_review_delay"]].sum(),use_container_width=True)
st.subheader("Supported subgroup access");sg=subgroups[(subgroups.scenario_id.isin(s.scenario_id))&(subgroups.supported==True)];st.dataframe(sg[["attribute","group","rows","proportion_alerts_reviewed","median_review_delay","unresolved_at_horizon_rate"]],use_container_width=True)
st.caption("Only aggregate CSV artifacts are loaded. Case-level Parquet and student identifiers are not read by this dashboard.")
