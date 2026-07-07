# Data Preparation

The IC runner accepts CommonRoad XML scenario files. The repository includes
three example XML files in `examples/`.

For your own data, either pass an explicit XML path:

```bash
python scripts/run_ic_single.py \
  --scenario /path/to/scenario.xml \
  --config configs/ic_v5_alpha1.json \
  --output-dir outputs/my_scenario \
  --no-vis
```

or create a local data directory:

```text
data/sind_left_turn/
  CHN_SIND-...xml
```

and list scenario ids in `scenario_lists/*.txt`.

## Scenario Id Convention

For full-horizon evaluation, the runner infers the horizon from SIND-style
scenario ids:

```text
CHN_SIND-<recording>_<frame_in>_LEFT_T-<frame_out>
```

The number of computation steps is:

```text
n_step = frame_out - frame_in
```

If your scenario id does not follow this convention, pass `--n-step` explicitly
to `scripts/run_ic_single.py`.

## Planner Labels

Evaluation-table reproduction requires planner failure labels in CSV form:

```text
scenario_id,rl_fail,cr_fail,cs_fail
```

These labels are not needed to compute IC. They are only needed to reproduce
Table I/II-style statistical evaluation.
