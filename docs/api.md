# Scoring API

Run `python -m uvicorn learnpulse.scoring_api:app --host 127.0.0.1 --port 8000`. Endpoints are `/health`, `/ready`, `/version`, `/v1/schema`, `/v1/model-card`, `/v1/validate`, `/v1/score`, `/v1/score/batch`, `/v1/monitoring/summary`, and `/metrics`.

Supply exactly the ten numeric personal-change fields shown by `/v1/schema`; documented nullable statistics may be JSON `null`. Unknown, identifier, demographic, target and future fields are rejected. Scoring defaults to no decision. An explicit `threshold` or packaged `decision_policy` is required for a binary research decision. Errors are structured and omit payloads. The service is local-only by default; remote use requires authentication and authorization not included here.
