from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def legacy_root() -> Path:
    return Path(os.environ.get("IC_LEGACY_ROOT", str(repo_root()))).resolve()


def scenario_id_from_path(value: str | Path) -> str:
    name = Path(str(value)).name
    return name[:-4] if name.endswith(".xml") else name


def frame_range_from_scenario_id(scenario_id: str) -> tuple[int, int]:
    m = re.search(r"_([0-9]+)_LEFT_T-([0-9]+)$", scenario_id_from_path(scenario_id))
    if not m:
        return 0, 30
    return int(m.group(1)), int(m.group(2))


def dynamic_n_step(scenario_id: str, cap_step: int = 0) -> int:
    frame_in, frame_out = frame_range_from_scenario_id(scenario_id)
    span = max(int(frame_out - frame_in), 1)
    return span if int(cap_step) <= 0 else min(span, int(cap_step))


def stable_split(scenario_id: str, seed: int = 2026) -> str:
    key = f"{seed}:{scenario_id_from_path(scenario_id)}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def load_scenario_list(path: str | Path) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_scenario_for_legacy(scenario: str | Path) -> str:
    raw = Path(str(scenario))
    root = repo_root()
    old = legacy_root()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((root / raw).resolve())
    sid = scenario_id_from_path(raw)
    candidates.append(old / "data/scenarios/left_turn_scenarios" / f"{sid}.xml")
    for cand in candidates:
        if cand.exists():
            real = cand.resolve()
            # Explicit XML paths should stay absolute. Some CommonRoad-Reach
            # installations otherwise reinterpret relative paths under their
            # own default scenario root.
            if raw.suffix == ".xml" or "/" in str(raw):
                return real.with_suffix("").as_posix() if real.suffix == ".xml" else real.as_posix()
            try:
                rel = real.relative_to(old)
                if rel.suffix == ".xml":
                    rel = rel.with_suffix("")
                return rel.as_posix()
            except ValueError:
                return real.with_suffix("").as_posix() if real.suffix == ".xml" else real.as_posix()
    return str(scenario)


def robust_z(series: pd.Series, median: float | None = None, mad: float | None = None, clip: float = 5.0) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    med = float(np.nanmedian(x)) if median is None else float(median)
    scale = float(np.nanmedian(np.abs(x - med))) if mad is None else float(mad)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return ((x - med) / scale).clip(-clip, clip)
