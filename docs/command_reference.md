# Command reference

| Task | PowerShell / cross-platform command |
|---|---|
| Tests | `python -m unittest discover -s tests -v` |
| Compile | `python -m compileall learnpulse` |
| Release audit | `python -m learnpulse.release_audit_cli --repository .` |
| Build reference | `python -m learnpulse.reference_model_cli --modeling-table data/processed/multicourse_modeling_table.parquet --artifact-dir artifacts/reference_model --folds 5 --random-seed 42 --model-version 1.0.0` |
| Validate artifact | `python -m learnpulse.task_cli validate-artifact` |
| Start API | `python -m uvicorn learnpulse.scoring_api:app --host 127.0.0.1 --port 8000` |
| Dashboard | `streamlit run learnpulse/workload_dashboard.py` |
| Docker | `docker compose up --build` |

On Unix-like shells, the commands are identical; activate the environment with `source .venv/bin/activate` rather than PowerShell's `.venv\Scripts\Activate.ps1`.
