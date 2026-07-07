# Interaction Complexity

This repository contains the public implementation of the interaction
complexity (IC) metric used in the paper:

> Interaction Complexity in Autonomous Driving: A Subjective-Objective
> Quantification Framework

The released code focuses on the IC computation pipeline only.

Given a CommonRoad XML scenario and an IC configuration, the code computes:

- `IC-Area`: reachable-area restriction score;
- `IC-Action`: least-action GP/DP trajectory-evaluation score;
- `IC-Combined`: normalized fusion of area and action components.

## Repository Layout

```text
interaction-complexity/
├── configs/                  # IC-V5 and normalized-fusion configs
├── docs/                     # method, data, and reproduction notes
├── examples/                 # three small CommonRoad XML examples
├── scripts/                  # command-line entry points
├── src/interaction_complexity/
│   ├── engine.py             # public single-scenario IC API
│   ├── evaluation/           # normalized fusion and paper-table utilities
│   └── legacy/               # IC full-horizon reachable-set + least-action core
├── tests/
├── README.md
├── LICENSE
├── pyproject.toml
└── environment.yml
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate interaction-complexity
pip install -e .
```

The implementation depends on the CommonRoad ecosystem, especially
`commonroad-io`, `commonroad-reach`, and `commonroad-drivability-checker`.
If your platform needs custom CommonRoad installation steps, install those
packages following the official CommonRoad instructions, then run `pip install -e .`.

## Quick Start

Run IC on one example scenario:

```bash
python scripts/run_ic_single.py \
  --scenario examples/CHN_SIND-834_10834_LEFT_T-10935.xml \
  --config configs/ic_v5_alpha1.json \
  --fusion-config configs/normalized_fusion.yaml \
  --output-dir outputs/ic_single/CHN_SIND-834_10834_LEFT_T-10935 \
  --no-vis
```

The output directory contains:

```text
scene_complexity.pkl
scene_complexity.json
action_field_diagnostics.json
path_conflict_diagnostics.json
metadata.json
```

The concise JSON summary reports the values most users need:

```json
{
  "ic_area": ...,
  "ic_action": ...,
  "ic_combined_raw_0p5": ...,
  "ic_combined_normalized_w_area_0p7": ...,
  "action_field_component_scaled": true
}
```

`ic_combined_normalized_w_area_0p7` is the paper-style normalized IC score for
the area-dominant sensitivity setting (`w_area=0.7`, `w_action=0.3`).

## Batch Usage

Create a scenario-list file with one XML path or scenario id per line. For the
included examples:

```bash
python scripts/run_ic_batch.py \
  --scenario-list scenario_lists/example_scenarios.txt \
  --config configs/ic_v5_alpha1.json \
  --fusion-config configs/normalized_fusion.yaml \
  --output-dir outputs/ic_example_batch \
  --workers 3
```

If a scenario id rather than a path is used, the script resolves it under
`data/sind_left_turn/<scenario_id>.xml`. For public use, explicit XML paths are
recommended.

## Reproducing Paper Tables

The paper evaluates IC against fixed planner-failure labels. This repository
does not ship planner outputs, but it provides the scripts used to reproduce
the table statistics once scores and labels are available.

1. Run `scripts/run_ic_batch.py` on your scenario set.
2. Prepare a labels CSV with columns:

```text
scenario_id,rl_fail,cr_fail,cs_fail
```

3. Reproduce Table I/II-style statistics:

```bash
python scripts/reproduce_table1_table2.py \
  --scores outputs/ic_example_batch/normalized_fusion/scores_and_labels.csv \
  --labels path/to/planner_labels.csv \
  --output-dir outputs/table_reproduction
```

See [docs/reproduce_paper_results.md](docs/reproduce_paper_results.md) for
details.

## Method Summary

The released configuration `configs/ic_v5_alpha1.json` corresponds to the
paper revision setting:

- full-horizon evaluation (`n_step = frame_out - frame_in`);
- component-scaled least-action field;
- reference-path path-conflict potential;
- incremental GP target;
- incremental physical DP;
- normalized area/action fusion.

See [docs/method_overview.md](docs/method_overview.md) for the implementation
details and how the outputs map to paper notation.

## Code Availability Statement

This repository is intended to support paper reproducibility and future
research on interaction complexity. The released code computes IC from
CommonRoad XML scenarios and exposes the configuration used in the paper. Large
datasets and planner outputs are not included; users should provide their own
CommonRoad scenarios and planner labels when reproducing evaluation tables.

## Citation

If you use this code, please cite the associated paper. A BibTeX entry will be
added after publication.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
