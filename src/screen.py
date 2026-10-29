"""Screen sensors for association with fails, then check whether each association survives
once time is held fixed.

Why the second step: yield rose from about 77% in July to about 99% by mid-October, so any sensor
that drifts over time looks "related to fails" even if it has nothing to do with them.

Steps
  1. drop sensors that are constant or missing in 50% or more of runs;
  2. overall test: Mann-Whitney U (fail vs pass), AUC = U / (n_fail * n_pass), Benjamini-Hochberg q;
  3. time check: split runs into equal-size blocks in time order and compute the AUC inside each block
     (pooled over blocks, weighted by fail-pass pairs); p-value by permuting labels within blocks.

Usage: python -m src.screen           # writes reports/screen.csv
"""
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from scipy.stats import false_discovery_control

from .load import DB, ROOT

MAX_MISSING = 0.5
N_BLOCKS = 8
N_PERM = 2000
TOP_FOR_PERM = 40


def load_wide(db=DB):
    con = sqlite3.connect(db)
    runs = pd.read_sql("SELECT run_id, ts, fail FROM runs ORDER BY run_id", con)
    m = pd.read_sql("SELECT run_id, sensor, value FROM measurements", con)
    con.close()
    wide = m.pivot(index="run_id", columns="sensor", values="value").sort_index()
    return runs, wide


def usable_sensors(wide):
    missing = wide.isna().mean()
    nunique = wide.nunique(dropna=True)
    keep = wide.columns[(missing < MAX_MISSING) & (nunique > 1)]
    return keep, missing


def auc(x, y):
    """P(sensor value of a random fail > that of a random pass), ties counted half. NaNs dropped."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return np.nan, 0
    r = rankdata(x)
    u = r[y == 1].sum() - n1 * (n1 + 1) / 2
    return u / (n1 * n0), n1 * n0


def within_block_auc(x, y, blocks):
    """AUC computed inside each time block, pooled with weights = number of fail-pass pairs in the block."""
    num = den = 0.0
    for b in np.unique(blocks):
        sel = blocks == b
        a, w = auc(x[sel], y[sel])
        if w:
            num += a * w
            den += w
    return num / den if den else np.nan


def permute_within(y, blocks, rng):
    y = y.copy()
    for b in np.unique(blocks):
        idx = np.flatnonzero(blocks == b)
        y[idx] = rng.permutation(y[idx])
    return y


def screen(runs, wide, n_blocks=N_BLOCKS, n_perm=N_PERM, top_for_perm=TOP_FOR_PERM, seed=0):
    keep, missing = usable_sensors(wide)
    y = runs["fail"].to_numpy()
    order = np.arange(len(y))
    blocks = order * n_blocks // len(y)
    rows = []
    for s in keep:
        x = wide[s].to_numpy()
        ok = ~np.isnan(x)
        p = mannwhitneyu(x[ok & (y == 1)], x[ok & (y == 0)], alternative="two-sided").pvalue
        a, _ = auc(x, y)
        rho = spearmanr(order[ok], x[ok]).statistic
        rows.append({"sensor": int(s), "missing_share": missing[s], "auc": a, "p": p, "rho_time": rho,
                     "auc_within_time": within_block_auc(x, y, blocks)})
    df = pd.DataFrame(rows)
    df["q"] = false_discovery_control(df["p"].to_numpy(), method="bh")
    df["effect"] = (df["auc"] - 0.5).abs()
    df = df.sort_values("p").reset_index(drop=True)

    # permutation test inside time blocks, only for the strongest overall candidates (it is the slow part)
    rng = np.random.default_rng(seed)
    perms = [permute_within(y, blocks, rng) for _ in range(n_perm)]
    df["p_within_time"] = np.nan
    for i in df.index[:top_for_perm]:
        x = wide[df.at[i, "sensor"]].to_numpy()
        obs = abs(df.at[i, "auc_within_time"] - 0.5)
        null = np.array([abs(within_block_auc(x, yp, blocks) - 0.5) for yp in perms])
        df.at[i, "p_within_time"] = (1 + np.sum(null >= obs)) / (1 + n_perm)
    tested = df["p_within_time"].notna()
    df.loc[tested, "q_within_time"] = false_discovery_control(df.loc[tested, "p_within_time"].to_numpy(), method="bh")
    return df, {"sensors_total": wide.shape[1], "sensors_screened": len(keep), "blocks": n_blocks,
                "fails_per_block": [int(y[blocks == b].sum()) for b in range(n_blocks)]}


if __name__ == "__main__":
    runs, wide = load_wide()
    df, info = screen(runs, wide)
    out = ROOT / "reports" / "screen.csv"
    df.to_csv(out, index=False, float_format="%.5g")
    print(info, file=sys.stderr)
    cols = ["sensor", "missing_share", "auc", "q", "rho_time", "auc_within_time", "p_within_time", "q_within_time"]
    print(df[cols].head(25).to_string(float_format="%.3f"))
