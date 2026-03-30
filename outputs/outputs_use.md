# Outputs Folder Quick Guide

This folder stores generated artifacts from curve generation and validation runs.

## Main Folders

- `curves/`
  - Active curve CSV files (`*_curves.csv`).
  - Usually the latest generated datasets used for plotting and post-processing.

- `curves_archive/`
  - Archived curve batches (for example, zip snapshots created by archive scripts).
  - Keeps historical curve outputs after they are rotated out of `curves/`.

- `validation/`
  - Validation report outputs.
  - Contains aggregated and per-run validation artifacts.

- `validation_modes/`
  - Mode-based validation outputs organized by source group.
  - Includes subfolders such as `weg_w21/`, `weg_w22/`, and `weg_w40/`.

- `validation_perf_after/`
  - Validation results focused on performance/runtime-oriented runs.
  - Used to compare runtime behavior after optimization changes.

## Note

These folders contain generated outputs and can grow quickly over time.