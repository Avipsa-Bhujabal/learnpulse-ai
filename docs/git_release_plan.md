# Git release plan

This directory was not detected as a Git worktree during release preparation. Do not initialize or publish it automatically.

Suggested manual sequence after review:

1. Initialize Git and commit core pipeline code: `feat: add reproducible OULAD research pipeline`.
2. Commit evaluation and methodology documents: `docs: add held-out research methodology and results index`.
3. Commit reference service and monitoring: `feat: add research reference scoring service`.
4. Commit release and portfolio materials: `docs: prepare v1.0.0 public release candidate`.
5. Verify ignored files, tests, links, secrets and license; then optionally tag `v1.0.0`.

Raw OULAD, processed Parquet/DuckDB data, model binaries, individual predictions, examples, logs and temporary outputs must remain untracked. Before pushing, inspect every staged path and repeat the release audit. No Git mutation was performed by this plan.

The exact proposed tracked-file inventory produced by the correction pass is
`experiments/release/proposed_tracked_files.txt`. It matches the public-candidate
ZIP and must be reviewed before any future `git add` operation.
