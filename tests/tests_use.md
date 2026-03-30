# Tests Folder Quick Guide

This folder contains regression tests for the Guimaraes validation pipeline.

## Python Files

- `guimaraes/test_deterministic_pipeline.py`
	- Validates deterministic pipeline correctness.
	- Checks grouping constraints (frequency and pole-pair consistency).
	- Checks physical guards (auto-correct and reject paths).
	- Checks deterministic block consistency.
	- Runs a small end-to-end validation and verifies key report sections.

- `guimaraes/test_runtime_optimizations.py`
	- Validates runtime/performance controls added for faster execution.
	- Checks iterative vs benchmark legacy runtime profile behavior.
	- Checks partition manifest reuse (skip rebuild when unchanged).
	- Checks timing metadata presence in validation/comparison reports.
	- Checks parser defaults for practical fast runs (iterative profile and quiet logging).

## Note

`__pycache__/` is generated automatically by Python and is not test source code.
