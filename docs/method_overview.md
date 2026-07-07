# Method Overview

This repository implements the revised interaction complexity (IC) pipeline
used in the paper.

## Components

For each CommonRoad scenario, the code computes two scenario-level components.

### IC-Area

`IC-Area` measures how dynamic traffic constraints reduce the reachable driving
corridor of the ego vehicle. It is computed by comparing the reachable area in
the actual multi-agent scene with the ideal no-dynamic-obstacle scene over the
full scenario horizon.

### IC-Action

`IC-Action` is derived from a least-action trajectory-evaluation field:

1. generate candidate ego trajectories;
2. compute per-step action components;
3. reconstruct the least-action field with Gaussian processes;
4. search the low-action path with dynamic programming;
5. compare the ideal path evaluated in the actual field with the actual
   lowest-action path.

The released configuration uses the component-scaled field:

```text
ActionField = normalized physical action
            + alpha * normalized path-conflict action
```

where `alpha = 1.0` in `configs/ic_v5_alpha1.json`.

### Path-Conflict Potential

The path-conflict potential detects when surrounding agents intrude into the
ego reference-path tube and forms a per-step temporal conflict term. This term
is part of the least-action interaction potential; it is not a standalone final
metric in the paper.

## Normalized Fusion

The final paper score uses robust normalization:

```text
z_area   = robust_z(IC-Area)
z_action = robust_z(IC-Action)
IC-Combined(w_area) = w_area * z_area + (1 - w_area) * z_action
```

`configs/normalized_fusion.yaml` stores the training-split median/MAD
statistics and the weight grid. The paper reports the area-dominant sensitivity
setting `w_area = 0.7`, `w_action = 0.3`.

## Output Mapping

`scene_complexity.json` contains:

- `ic_area`: scenario-level IC-Area;
- `ic_action`: scenario-level IC-Action;
- `ic_combined_raw_0p5`: legacy raw 0.5/0.5 frame-level score;
- `action_field_component_scaled`: whether the V5 component-scaled action field
  is active;
- `path_conflict_event_count`: number of detected reference-path conflict
  events.

For paper-style normalized IC, use the batch output
`normalized_fusion/scores_and_labels.csv` and the column corresponding to the
desired weight, e.g. `IC-Normalized-w_area=0.7`.
