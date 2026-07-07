import pandas as pd
from interaction_complexity.evaluation.normalized_fusion import add_normalized_fusion


def test_normalized_fusion_outputs_grid(tmp_path):
    df = pd.DataFrame({
        "scenario_id": [f"s_{i}_LEFT_T-{i+20}" for i in range(10)],
        "IC-Area": list(range(10)),
        "IC-Action": list(reversed(range(10))),
        "rl_fail": [0,0,0,0,0,1,1,1,1,1],
        "cr_fail": [0,0,0,1,1,0,1,1,1,1],
        "cs_fail": [0,1,0,1,0,1,0,1,0,1],
    })
    cfg = {"normalization": {"clip": 5.0}, "fusion": {"sensitivity_grid": [0.0, 0.5, 1.0]}}
    add_normalized_fusion(df, cfg, tmp_path)
    assert (tmp_path / "normalized_fusion_grid.csv").exists()
    assert (tmp_path / "scores_and_labels.csv").exists()
