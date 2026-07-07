from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


PLANNERS = {"FRL": "rl_fail", "CRP": "cr_fail", "CRS": "cs_fail"}
COMPARISONS = {"FRL-CRP": ("rl_fail", "cr_fail"), "FRL-CRS": ("rl_fail", "cs_fail")}


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def mean_finite(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.nanmean(arr)) if arr.size else np.nan


def scenario_dirs(scene_root: Path) -> List[Path]:
    return sorted([p for p in scene_root.iterdir() if p.is_dir() and not p.name.endswith("_MCTS")])


def load_ic_scores(scene_root: Path) -> pd.DataFrame:
    rows = []
    for scen_dir in scenario_dirs(scene_root):
        pkl_path = scen_dir / "scene_complexity.pkl"
        if not pkl_path.exists():
            continue
        res = load_pickle(pkl_path)
        area = np.asarray(res.get("area_penalty", []), dtype=float)
        action = np.asarray(res.get("action_penalty", []), dtype=float)
        complexity = np.asarray(res.get("complexity", []), dtype=float)
        row = {
            "scenario_id": scen_dir.name,
            "IC-Combined": float(res.get("overall_score", np.nan)),
            "IC-Area": mean_finite(area),
            "IC-Action": mean_finite(action),
            "IC-FrameMean": mean_finite(complexity),
            "ic_n_frames": int(np.isfinite(complexity).sum()),
            "n_step": int(res.get("n_step", -1)),
        }
        for key in [
            "actual_path_conflict_cost",
            "ideal_path_conflict_cost_on_actual_field",
            "path_conflict_offset",
            "actual_dp_min_clearance",
            "ideal_path_min_clearance",
            "actual_path_conflict_event_count",
            "actual_small_actor_event_count",
        ]:
            row[key] = res.get(key, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_fail_labels_from_csv(labels_csv: Path, scenario_ids: Sequence[str]) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    required = {"scenario_id"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels CSV is missing required columns: {sorted(missing)}")
    keep = ["scenario_id"] + [c for c in ["rl_fail", "cr_fail", "cs_fail"] if c in labels.columns]
    labels = labels[keep].copy()
    return pd.DataFrame({"scenario_id": list(scenario_ids)}).merge(labels, on="scenario_id", how="left")


def build_fail_labels_from_pickles(least_action_root: Path, commonroad_search_root: Path, scenario_ids: Sequence[str]) -> pd.DataFrame:
    res_rl = load_pickle(least_action_root / "output/data/scenarios/res_rl.pkl")
    res_cr = load_pickle(least_action_root / "output/data/scenarios/res_cr.pkl")
    res_cs = load_pickle(commonroad_search_root / "res_cs.pkl")

    def fail_set(res: dict) -> set:
        return set(map(str, res.get("collided_scenes", []))) | set(map(str, res.get("no_result_scenes", [])))

    fail = {"rl_fail": fail_set(res_rl), "cr_fail": fail_set(res_cr), "cs_fail": fail_set(res_cs)}
    return pd.DataFrame([{ "scenario_id": sid, **{col: int(sid in vals) for col, vals in fail.items()} } for sid in scenario_ids])


def fit_logit_coef(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return np.nan
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(x.reshape(-1, 1), y.astype(int))
    return float(lr.coef_[0][0])


def single_stats(df: pd.DataFrame, score_col: str, fail_col: str) -> dict:
    d = df[[score_col, fail_col]].rename(columns={score_col: "score", fail_col: "fail"}).replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return {"n": 0, "fail_count": 0, "fail_rate": np.nan, "auc": np.nan, "logit_coef": np.nan, "pointbiserial_r": np.nan, "pointbiserial_p": np.nan}
    y = d["fail"].astype(int).to_numpy()
    x = d["score"].astype(float).to_numpy()
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        auc = np.nan
        r_pb, p_pb = np.nan, np.nan
    else:
        auc = float(roc_auc_score(y, x))
        r_pb, p_pb = pointbiserialr(y, x)
    return {
        "n": int(len(d)),
        "fail_count": int(y.sum()),
        "fail_rate": float(y.mean()),
        "auc": auc,
        "logit_coef": fit_logit_coef(x, y),
        "pointbiserial_r": float(r_pb),
        "pointbiserial_p": float(p_pb),
    }


def table1(df: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for planner, fail_col in PLANNERS.items():
            rows.append({"method": method, "planner": planner, **single_stats(df, method, fail_col)})
    return pd.DataFrame(rows)


def qcut_decile(values: pd.Series, n_bins: int = 10) -> pd.Series:
    return pd.qcut(values, q=n_bins, labels=False, duplicates="drop")


def table2(df: pd.DataFrame, methods: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    decile_rows = []
    for method in methods:
        d = df[[method, "rl_fail", "cr_fail", "cs_fail"]].rename(columns={method: "score"}).replace([np.inf, -np.inf], np.nan).dropna().copy()
        if d.empty:
            continue
        d["decile"] = qcut_decile(d["score"])
        if d["decile"].dropna().empty:
            continue
        rate = d.groupby("decile")[["rl_fail", "cr_fail", "cs_fail"]].mean()
        counts = d.groupby("decile").size().rename("n").reset_index()
        dec = rate.reset_index().merge(counts, on="decile", how="left")
        dec.insert(0, "method", method)
        decile_rows.append(dec)
        for label, (a, b) in COMPARISONS.items():
            diff = (rate[a] - rate[b]).dropna()
            rho, rho_p = spearmanr(diff.index.to_numpy(dtype=float), diff.to_numpy(dtype=float)) if len(diff) >= 2 else (np.nan, np.nan)
            best_decile = d["decile"].dropna().max()
            high = d[d["decile"] == best_decile]
            b_count = int(((high[a] == 1) & (high[b] == 0)).sum())
            c_count = int(((high[a] == 0) & (high[b] == 1)).sum())
            n_discordant = b_count + c_count
            p_val = float(binomtest(min(b_count, c_count), n=n_discordant, p=0.5).pvalue) if n_discordant else 1.0
            rows.append(
                {
                    "method": method,
                    "comparison": label,
                    "n": int(len(d)),
                    "best_decile": int(best_decile),
                    "spearman_rho": float(rho),
                    "spearman_p": float(rho_p),
                    "mcnemar_b": b_count,
                    "mcnemar_c": c_count,
                    "mcnemar_n": n_discordant,
                    "mcnemar_p": p_val,
                    "top_decile_fail_rate_a": float(high[a].mean()) if len(high) else np.nan,
                    "top_decile_fail_rate_b": float(high[b].mean()) if len(high) else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()


def add_weight_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    methods = []
    for w in np.linspace(0.0, 1.0, 11):
        col = f"IC-w_area={w:.1f}"
        out[col] = float(w) * out["IC-Area"] + (1.0 - float(w)) * out["IC-Action"]
        methods.append(col)
    return out, methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize social-force IC outputs against planner failure labels.")
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional CSV with scenario_id, rl_fail, cr_fail, cs_fail columns.")
    parser.add_argument("--least-action-root", type=Path, default=None, help="Optional legacy result root containing output/data/scenarios/res_rl.pkl and res_cr.pkl.")
    parser.add_argument("--commonroad-search-root", type=Path, default=None, help="Optional legacy CommonRoad Search root containing res_cs.pkl.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ic = load_ic_scores(args.scene_root)
    scenario_ids = sorted(ic["scenario_id"].astype(str).tolist())
    if args.labels_csv is not None:
        labels = build_fail_labels_from_csv(args.labels_csv, scenario_ids)
    elif args.least_action_root is not None and args.commonroad_search_root is not None:
        labels = build_fail_labels_from_pickles(args.least_action_root, args.commonroad_search_root, scenario_ids)
    else:
        labels = pd.DataFrame({"scenario_id": scenario_ids})
    scores = labels.merge(ic, on="scenario_id", how="left")
    scores, weight_methods = add_weight_grid(scores)
    main_methods = ["IC-Combined", "IC-Area", "IC-Action"]
    scores.to_csv(args.output_dir / "social_force_scores_and_labels.csv", index=False)
    diag_cols = [
        "scenario_id",
        "actual_path_conflict_cost",
        "ideal_path_conflict_cost_on_actual_field",
        "path_conflict_offset",
        "actual_dp_min_clearance",
        "ideal_path_min_clearance",
        "actual_path_conflict_event_count",
        "actual_small_actor_event_count",
        "rl_fail",
        "cr_fail",
        "cs_fail",
    ]
    scores[[c for c in diag_cols if c in scores.columns]].to_csv(args.output_dir / "path_conflict_component_summary.csv", index=False)
    if set(PLANNERS.values()).issubset(scores.columns):
        table1(scores, main_methods).to_csv(args.output_dir / "table1_effectiveness.csv", index=False)
        disc, deciles = table2(scores, main_methods)
        disc.to_csv(args.output_dir / "table2_discrimination.csv", index=False)
        deciles.to_csv(args.output_dir / "decile_failure_rates.csv", index=False)
        table1(scores, weight_methods).to_csv(args.output_dir / "weight_sensitivity_effectiveness.csv", index=False)
        wdisc, _ = table2(scores, weight_methods)
        wdisc.to_csv(args.output_dir / "weight_sensitivity_discrimination.csv", index=False)
    else:
        (args.output_dir / "README_no_labels.txt").write_text(
            "Planner label columns were not provided, so Table I/II statistics were skipped. "
            "Provide --labels-csv with scenario_id, rl_fail, cr_fail, cs_fail columns to reproduce paper tables.\n",
            encoding="utf-8",
        )
    summary = {"scene_count": int(len(ic)), "output_dir": str(args.output_dir), "scene_root": str(args.scene_root)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
