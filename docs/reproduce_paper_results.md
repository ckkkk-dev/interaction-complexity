# Reproducing Paper Results

This repository separates metric computation from planner evaluation.

## Step 1: Compute IC Scores

```bash
python scripts/run_ic_batch.py \
  --scenario-list scenario_lists/sind_left_turn_1075.txt \
  --config configs/ic_v5_alpha1.json \
  --fusion-config configs/normalized_fusion.yaml \
  --output-dir outputs/ic_v5_full \
  --workers 24
```

This produces per-scenario `scene_complexity.pkl` files and a normalized fusion
table under:

```text
outputs/ic_v5_full/normalized_fusion/scores_and_labels.csv
```

## Step 2: Add Planner Labels

Prepare a CSV with:

```text
scenario_id,rl_fail,cr_fail,cs_fail
```

`FRL`, `CRP`, and `CRS` correspond to the three planner-failure labels used in
the paper. The planner implementations and outputs are not included in this
repository.

## Step 3: Reproduce Table I/II Statistics

```bash
python scripts/reproduce_table1_table2.py \
  --scores outputs/ic_v5_full/normalized_fusion/scores_and_labels.csv \
  --labels path/to/planner_labels.csv \
  --output-dir outputs/table_reproduction
```

Outputs include:

- `table1_effectiveness.csv`: AUC, standardized logistic coefficient,
  point-biserial correlation, and p-value;
- `table2_discrimination.csv`: decile trend and top-decile McNemar statistics;
- `decile_failure_rates.csv`: appendix-style decile failure-rate table.

## Notes on Randomness

Planner labels are treated as fixed deterministic evaluation results. Random
seeds affect candidate sampling and DP search in IC computation. Use the
`--seed` argument of `scripts/run_ic_single.py` if you need seed-controlled
single-scenario runs.
