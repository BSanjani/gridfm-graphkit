# GraphKit Agent Handoff Prompt

You are in the `gridfm-graphkit` repository.

I copied a prepared dataset folder from GridFM DataKit. Your job is to consume it and run a first end-to-end training/evaluation pass.

## Dataset Location
Use this directory as `data_dir`:
- `graphkit_data/case24_ieee_rts_s10_droop_mprange_mqrange`

Expected files:
- `raw/pf_node.csv`
- `raw/pf_edge.csv`
- `scenario_split.json`
- `train_scenarios.csv`
- `val_scenarios.csv`
- `test_scenarios.csv`
- `dataset_manifest.json`

## What This Dataset Is
- Source grid: IEEE RTS 24-bus (`case24_ieee_rts`)
- Number of scenarios: 10
- PF with droop control enabled
- Droop randomized by scenario:
  - `mp_range = [0.03, 0.05]`
  - `mq_range = [0.02, 0.04]`
- Bus voltage limits were set to `Vmax = 1.2` in the source case.

## Schema
### Node file: `raw/pf_node.csv`
Columns:
- `scenario`
- `bus`
- `Pd`, `Qd`, `Pg`, `Qg`, `Vm`, `Va`
- `PQ`, `PV`, `REF`

### Edge file: `raw/pf_edge.csv`
Columns:
- `scenario`
- `index1`, `index2`
- `G`, `B`

## Fixed Scenario Split (must use)
- train: scenarios `[0,1,2,3,4,5,6,7]`
- val: scenario `[8]`
- test: scenario `[9]`

Read from `scenario_split.json` (do not resample split).

## Required Tasks
1. Verify dataset loads correctly with current GraphKit datamodule.
2. Create a training config for a first baseline run (feature reconstruction task).
3. Run training.
4. Run evaluation on test split.
5. Export predictions for test split.
6. Save artifacts (config used, logs, metrics, predictions path).

## Practical Constraints
- Do not modify dataset files.
- Keep run deterministic (set seed).
- If any expected GraphKit config field names differ, adapt to current GraphKit API and document the exact mapping.

## Deliverables
Return:
1. The exact config file used.
2. Exact commands run.
3. Final metrics summary (at least loss + key voltage metrics if available).
4. Paths to model checkpoint and predictions.
5. Any assumptions or schema adaptations made.
