# Contributing

Use Python 3.10 or 3.11, create a virtual environment, install `requirements.txt`, and run `python -m unittest discover -s tests -v`. Pull requests should include focused tests, compile successfully and pass the release audit.

Never commit raw OULAD data, demographic records, student-level predictions, case outputs, logs containing payloads, or real learner examples. Tests and API examples must use synthetic miniature data. Changes to outcomes, features or evaluation protocols require explicit methodological documentation and must not rewrite historical artifacts.
