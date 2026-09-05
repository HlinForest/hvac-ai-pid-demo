# outputs/ — regenerable pipeline scratch (dedup P1)

This directory is **not tracked** (see `.gitignore`). It is the default
`python main.py --output outputs` target — rerun the pipeline to recreate it.

- Sealed history: `outputs_review_v3/` (canonical; includes the Modelica
  reference CSV, OpenModelica cross-check, `submission_package/` and
  `documents/` cherry-picked here in dedup P1).
- New runs: `artifacts/runs/<run_id>/` (see `experiments/manifests/v4.yaml`).
- Removed batches and their new locations: `experiments/manifests/archive_map.csv`.
- Full pre-dedup history: git tag `pre-dedup-a3d8fba`.
