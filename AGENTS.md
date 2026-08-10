# Repository instructions

## Communication and reasoning

- Default to Chinese when communicating with the repository owner. Keep code,
  identifiers, and public documentation consistent with the surrounding files.
- Before acting, identify missing information, unsupported assumptions, concept
  confusion, and contradictions. Do not treat insufficient evidence as proof that
  a claim is false, and do not present an inference as a verified fact.
- For numerical results, dates, hardware, citations, and experiment claims, verify
  the primary local source when possible. State uncertainty and provenance gaps.
- Do not agree automatically. Explain disagreements with evidence, risks, and a
  practical alternative.

## Experiment artifacts

- Runtime outputs belong under `Standalone/KPConvX/results/` and remain local.
- Versioned summaries belong under `experiment_reports/<dataset>/<run-id>/`.
- Every versioned run must contain `README.md` and `metrics.csv`. Optional small,
  sanitized JSON/YAML/CSV files may hold configuration, class metrics, or an
  inventory of checkpoint file names.
- Never commit model weights, checkpoints, datasets, raw logs, point-cloud
  predictions, profiler dumps, process state, lock files, or notebook checkpoints.
- A report may name a checkpoint and its internal epoch, but must not include the
  checkpoint binary.
- Remove credentials, user names, PIDs, host-specific absolute paths, and private
  endpoints. Use placeholders such as `<DATASET_PATH>` in commands.
- Record dataset and split, model/configuration, seed, training and evaluation
  protocol, metric definitions, selection rule, hardware, software environment,
  resource use, completion status, and known limitations.
- Record the exact Git commit only when it was captured during the run. Otherwise
  write `not recorded`; never substitute the current commit after the fact.
- Disclose single-seed uncertainty, checkpoint-selection bias, interrupted/resumed
  training, adaptive batch-size changes, failed attempts, and mismatched training
  and evaluation hardware when applicable.

## Validation and Git hygiene

- Run `python3 tools/validate_experiment_reports.py` before committing reports.
- Run focused tests for changed code plus `git diff --check`.
- Stage reviewed paths explicitly. Do not use `git add -A` in this repository,
  because local result trees can contain many gigabytes of generated artifacts.
- Push personal work to the `fork` remote. Treat `origin` as the upstream Apple
  repository unless the repository owner explicitly says otherwise.
- Server shutdown must be an explicit opt-in. Do not stop or restart `cc-switch`
  or its local proxy while working in this repository.
- AutoDL-specific safety: never execute `shutdown`, `poweroff`, or `halt` (including
  `--help`, `--show`, `-c`, or dry-run forms), and never create an automatic
  poweroff watcher or put poweroff logic in an `EXIT` trap. Inspect such files
  only with read-only commands; instance power operations are console-only.
