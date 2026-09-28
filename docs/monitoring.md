# Monitoring

Monitoring compares a batch with packaged training summaries using null-rate changes, standardized mean differences, median shifts, out-of-range rates and PSI. It separately summarizes score drift. No single statistic proves harmful drift. Delayed-label PR-AUC, ROC-AUC, Brier, calibration error and confusion metrics are computed only when labels are explicitly supplied with exact outcome-window alignment. Informational, warning and critical cutoffs are illustrative and trigger manual investigation only.
