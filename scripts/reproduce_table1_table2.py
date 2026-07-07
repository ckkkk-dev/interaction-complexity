#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PLANNERS = {"FRL": "rl_fail", "CRP": "cr_fail", "CRS": "cs_fail"}
COMPARISONS = {"FRL-vs-CRP": ("rl_fail", "cr_fail"), "FRL-vs-CRS": ("rl_fail", "cs_fail"), "CRP-vs-CRS": ("cr_fail", "cs_fail")}


def _standardize(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < 1e-9:
        std = 1.0
    return (x - med) / std


def _logit_coef_std(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return np.nan
    z = _standardize(x)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(z.reshape(-1, 1), y.astype(int))
    return float(clf.coef_[0][0])


def table1(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        for planner, label in PLANNERS.items():
            d = df[[method, label]].replace([np.inf, -np.inf], np.nan).dropna()
            y = d[label].astype(int).to_numpy()
            x = d[method].astype(float).to_numpy()
            if len(d) == 0 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
                auc = beta = r_pb = p_pb = np.nan
            else:
                auc = float(roc_auc_score(y, x))
                beta = _logit_coef_std(x, y)
                r_pb, p_pb = pointbiserialr(y, x)
            rows.append({
                "method": method,
                "planner": planner,
                "n": int(len(d)),
                "fail_count": int(y.sum()) if len(d) else 0,
                "auc": auc,
                "std_beta1": beta,
                "pointbiserial_r": float(r_pb),
                "pointbiserial_p": float(p_pb),
            })
    return pd.DataFrame(rows)


def _decile(values: pd.Series) -> pd.Series:
    return pd.qcut(values, q=10, labels=False, duplicates="drop")


def table2(df: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    decile_rows = []
    for method in methods:
        d = df[[method, *PLANNERS.values()]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if d.empty:
            continue
        d["decile"] = _decile(d[method])
        rate = d.groupby("decile")[list(PLANNERS.values())].mean()
        dec = rate.reset_index()
        dec.insert(0, "method", method)
        decile_rows.append(dec)
        for pair, (a, b) in COMPARISONS.items():
            diff = (rate[a] - rate[b]).dropna()
            rho, p = spearmanr(diff.index.to_numpy(float), diff.to_numpy(float)) if len(diff) >= 2 else (np.nan, np.nan)
            top = d[d["decile"] == d["decile"].max()]
            b_count = int(((top[a] == 1) & (top[b] == 0)).sum())
            c_count = int(((top[a] == 0) & (top[b] == 1)).sum())
            discordant = b_count + c_count
            mcnemar_p = float(binomtest(min(b_count, c_count), n=discordant, p=0.5).pvalue) if discordant else 1.0
            rows.append({
                "method": method,
                "comparison": pair,
                "spearman_rho": float(rho),
                "spearman_p": float(p),
                "top_decile_fail_rate_a": float(top[a].mean()) if len(top) else np.nan,
                "top_decile_fail_rate_b": float(top[b].mean()) if len(top) else np.nan,
                "mcnemar_b": b_count,
                "mcnemar_c": c_count,
                "mcnemar_p": mcnemar_p,
            })
    return pd.DataFrame(rows), pd.concat(decile_rows, ignore_index=True) if decile_rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Table I/II-style IC statistics from scores and planner labels.")
    parser.add_argument("--scores", required=True, type=Path, help="CSV containing scenario_id and IC score columns.")
    parser.add_argument("--labels", required=True, type=Path, help="CSV containing scenario_id, rl_fail, cr_fail, cs_fail.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ic-column", default="IC-Normalized-w_area=0.7")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    labels = pd.read_csv(args.labels)
    df = scores.merge(labels[[c for c in ["scenario_id", *PLANNERS.values()] if c in labels.columns]], on="scenario_id", how="inner")
    if args.ic_column not in df.columns:
        raise ValueError(f"IC column not found: {args.ic_column}")
    df = df.rename(columns={args.ic_column: "IC-final"})
    methods = ["IC-final"]
    for col in ["IC-Area", "IC-Action"]:
        if col in df.columns:
            methods.append(col)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table1(df, methods).to_csv(args.output_dir / "table1_effectiveness.csv", index=False)
    disc, deciles = table2(df, methods)
    disc.to_csv(args.output_dir / "table2_discrimination.csv", index=False)
    deciles.to_csv(args.output_dir / "decile_failure_rates.csv", index=False)
    df.to_csv(args.output_dir / "merged_scores_and_labels.csv", index=False)


if __name__ == "__main__":
    main()
