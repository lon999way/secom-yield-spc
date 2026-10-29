"""Run every analysis step and write tables, charts and reports/summary.json.

Usage: python -m src.report      (after python -m src.load and python -m src.screen)
"""
import json
import sqlite3
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from . import model, spc
from .load import DB, ROOT
from .screen import load_wide

REPORTS = ROOT / "reports"
SQL = ROOT / "sql"
BASELINE_N = 300        # first 300 readings of each sensor form the provisional Phase I baseline
GROUP_RHO = 0.9         # sensors this correlated are treated as one signal
N_CHARTS = 5
EXTREME_Z = 10          # robust z (median / MAD) beyond which a single reading counts as extreme
EVENT_MIN_SENSORS = 30  # a run with this many extreme sensors at once is treated as a multi-sensor event

BLUE, RED, GRAY, INK, MUTED = "#2a78d6", "#e34948", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK})


def run_sql():
    con = sqlite3.connect(DB)
    out = {}
    (REPORTS / "sql").mkdir(parents=True, exist_ok=True)
    for f in sorted(SQL.glob("*.sql")):
        df = pd.read_sql(f.read_text(), con)
        df.to_csv(REPORTS / "sql" / f"{f.stem}.csv", index=False)
        out[f.stem] = df
    con.close()
    return out


def group_signals(survivors, wide):
    """Greedy grouping: walk sensors from strongest to weakest, attach each to the first group
    whose representative it correlates with at |Spearman rho| >= GROUP_RHO."""
    corr = wide[survivors].corr(method="spearman").abs()
    groups = []
    for s in survivors:
        for g in groups:
            if corr.at[g[0], s] >= GROUP_RHO:
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


def chart_sensor(x, fail, sensor):
    """I-MR limits from the first BASELINE_N readings, then Western Electric rules over all readings."""
    ok = ~np.isnan(x)
    idx = np.flatnonzero(ok)
    vals = x[ok]
    lim = spc.imr_limits(vals[:BASELINE_N])
    z = (vals - lim["center"]) / lim["sigma"]
    rules = spc.western_electric(z)
    flagged = np.array([bool(r) for r in rules])
    f = fail[idx]
    after = np.arange(len(vals)) >= BASELINE_N
    row = {"sensor": int(sensor), "center": lim["center"], "sigma": lim["sigma"],
           "readings": int(len(vals)), "flagged_points": int(flagged.sum()),
           "flagged_by_rule": {r: int(sum(r in s for s in rules)) for r in (1, 2, 3, 4)},
           "fail_rate_flagged": float(f[flagged].mean()) if flagged.any() else None,
           "fail_rate_not_flagged": float(f[~flagged].mean()),
           "fails_caught": int(f[flagged].sum()), "fails_total": int(f.sum()),
           "monitoring_flagged": int((flagged & after).sum()),
           "monitoring_fails_flagged": int(f[flagged & after].sum()),
           "monitoring_fail_rate_flagged": float(f[flagged & after].mean()) if (flagged & after).any() else None,
           "monitoring_fail_rate_not_flagged": float(f[~flagged & after].mean())}
    table = [[int(f[flagged & after].sum()), int((1 - f[flagged & after]).sum())],
             [int(f[~flagged & after].sum()), int((1 - f[~flagged & after]).sum())]]
    row["monitoring_fisher_p"] = float(fisher_exact(table).pvalue)
    return row, idx, vals, lim, flagged


def plot_imr(ax, idx, vals, lim, flagged, fail, sensor):
    c, s = lim["center"], lim["sigma"]
    lo = min(c - 3.5 * s, np.percentile(vals, 0.5))
    hi = max(c + 3.5 * s, np.percentile(vals, 99.5))
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    shown = np.clip(vals, lo, hi)
    ax.plot(idx, shown, color=BLUE, lw=0.8, zorder=2)
    for k, ls in ((3, "-"), (2, ":"), (1, ":")):
        for sign in (1, -1):
            ax.axhline(c + sign * k * s, color=GRAY, lw=0.8 if k == 3 else 0.5, ls=ls, zorder=1)
    ax.axhline(c, color=MUTED, lw=0.8, zorder=1)
    ax.axvspan(idx[0], idx[min(BASELINE_N, len(idx)) - 1], color=GRAY, alpha=0.08, lw=0)
    inside = (vals >= lo) & (vals <= hi)
    ax.scatter(idx[flagged & inside], vals[flagged & inside], s=14, color=RED, zorder=3,
               label="Western Electric signal", edgecolor="white", linewidth=0.6)
    off = ~inside
    ax.scatter(idx[off], np.where(vals[off] > hi, hi, lo), marker="^", s=22, color=RED, zorder=3,
               label="off scale (signal)", edgecolor="white", linewidth=0.6)
    strip = lo - 0.08 * (hi - lo)
    fails = fail[idx] == 1
    ax.scatter(idx[fails], np.full(fails.sum(), strip), marker="|", s=40, color=INK, label="failed unit", zorder=3)
    ax.set_ylim(strip - 0.04 * (hi - lo), hi + 0.02 * (hi - lo))
    ax.set_title(f"Sensor {sensor}: individuals chart ({off.sum()} readings off scale)", loc="left")
    ax.set_xlabel("run (time order)")


def plot_pchart(ax, daily, pc):
    days = pd.to_datetime(daily["day"])
    ax.step(days, pc["ucl"], where="mid", color=GRAY, lw=0.8, label="UCL (varies with daily volume)")
    ax.axhline(pc["center"], color=MUTED, lw=0.8)
    ax.plot(days, pc["p"], color=BLUE, lw=0.8, marker="o", ms=3)
    out = pc["p"] > pc["ucl"]
    ax.scatter(days[out], pc["p"][out], s=30, color=RED, zorder=3, label="above UCL", edgecolor="white")
    ax.set_title("Daily fail fraction: p-chart", loc="left")
    ax.set_ylabel("fail fraction")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(axis="x", rotation=30)


def main():
    sql = run_sql()
    runs, wide = load_wide()
    fail = runs["fail"].to_numpy()
    screen = pd.read_csv(REPORTS / "screen.csv")

    # yield: p-chart on daily volumes, plus Western Electric rules on the standardized values
    daily = sql["03_daily_yield"]
    pc = spc.p_chart(daily["fails"], daily["runs"])
    pc_rules = spc.western_electric(pc["z"])
    pchart = {"days": len(daily), "center_fail_rate": round(pc["center"], 4),
              "days_above_ucl": daily["day"][pc["p"] > pc["ucl"]].tolist(),
              "days_rule4_below_center": daily["day"][[4 in r and z < 0 for r, z in zip(pc_rules, pc["z"])]].tolist()}

    # sensors: survivors of the time-adjusted test, grouped into distinct signals
    surv = screen[screen["q_within_time"] < 0.05].sort_values("p_within_time")
    groups = group_signals(surv["sensor"].tolist(), wide)
    reps = [g[0] for g in groups][:N_CHARTS]
    charts = []
    fig, axes = plt.subplots(len(reps), 1, figsize=(9, 2.3 * len(reps)), constrained_layout=True)
    for ax, s in zip(axes, reps):
        row, idx, vals, lim, flagged = chart_sensor(wide[s].to_numpy(), fail, s)
        charts.append(row)
        plot_imr(ax, idx, vals, lim, flagged, fail, s)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False, fontsize=8,
               title="Shaded = baseline (first 300 readings); solid gray = 3-sigma limits, dotted = 1 and 2 sigma",
               title_fontsize=8)
    fig.savefig(REPORTS / "control_charts.png", dpi=150)
    plt.close(fig)

    # multi-sensor events: runs where many sensors are extreme at the same time
    X = wide.to_numpy()
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0) * 1.4826
    use = mad > 0
    extreme = (np.abs((X[:, use] - med[use]) / mad[use]) > EXTREME_Z).sum(axis=1)
    ev = np.flatnonzero(extreme >= EVENT_MIN_SENSORS)
    events = {"sensors_checked": int(use.sum()), "median_extreme_per_run": float(np.median(extreme)),
              "p99_extreme_per_run": float(np.percentile(extreme, 99)),
              "runs": [{"run_id": int(r), "ts": runs["ts"].iloc[r], "extreme_sensors": int(extreme[r]),
                        "fail": int(fail[r])} for r in ev]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = [model.evaluate(wide.to_numpy(), fail, k=k) for k in (20, 40, 80)]

    # one-page dashboard
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    w = sql["04_weekly_yield"]
    ax[0, 0].plot(pd.to_datetime(w["week_start"]), w["yield"] * 100, color=BLUE, lw=2, marker="o", ms=4)
    ax[0, 0].set_title("Weekly yield (%)", loc="left")
    ax[0, 0].tick_params(axis="x", rotation=30)
    plot_pchart(ax[0, 1], daily, pc)
    top = screen.dropna(subset=["auc_within_time"]).head(20)
    ax[1, 0].scatter(top["auc"], top["auc_within_time"], s=24, color=BLUE, edgecolor="white", zorder=3)
    drop = (top["auc"] - 0.5).abs() - (top["auc_within_time"] - 0.5).abs()
    label = set(reps[:3]) | set(top.loc[drop.nlargest(2).index, "sensor"])
    sel = top[top["sensor"].isin(label)]
    for (a, aw), g in sel.groupby([sel["auc"].round(2), sel["auc_within_time"].round(2)]):
        ax[1, 0].annotate("/".join(str(int(v)) for v in g["sensor"]), (g["auc"].mean(), g["auc_within_time"].mean()), fontsize=7, color=MUTED,
                          xytext=(3, 2), textcoords="offset points")
    lo, hi = 0.35, 0.72
    ax[1, 0].plot([lo, hi], [lo, hi], color=GRAY, lw=0.8)
    ax[1, 0].axhline(0.5, color=GRAY, lw=0.5, ls=":")
    ax[1, 0].set_xlabel("AUC, all runs pooled")
    ax[1, 0].set_ylabel("AUC within time blocks")
    ax[1, 0].set_title("Top 20 sensors: association before vs. after holding time fixed", loc="left")
    row, idx, vals, lim, flagged = chart_sensor(wide[reps[0]].to_numpy(), fail, reps[0])
    plot_imr(ax[1, 1], idx, vals, lim, flagged, fail, reps[0])
    fig.savefig(REPORTS / "dashboard.png", dpi=150)
    plt.close(fig)

    summary = {
        "runs": int(len(runs)), "fails": int(fail.sum()), "first_run": runs["ts"].iloc[0], "last_run": runs["ts"].iloc[-1],
        "sensors": int(wide.shape[1]),
        "missing_buckets": sql["01_missing_by_sensor"].to_dict("records"),
        "constant_sensors": int(sql["02_constant_sensors"].iloc[0, 0]),
        "weekly_yield": w[["week_start", "yield"]].to_dict("records"),
        "pchart": pchart,
        "screen": {"screened": int(len(screen)), "q_overall_lt_05": int((screen["q"] < 0.05).sum()),
                   "tested_within_time": int(screen["p_within_time"].notna().sum()),
                   "q_within_time_lt_05": int(len(surv)),
                   "dropped_after_time_check": screen[(screen["q"] < 0.05) & ~(screen["q_within_time"] < 0.05)]["sensor"].tolist()},
        "multi_sensor_events": events,
        "signal_groups": groups, "charted_sensors": reps, "control_charts": charts, "models": models,
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: s[k] for k in ("pchart", "screen", "signal_groups", "multi_sensor_events", "models")}, indent=1, default=float))
    for c in s["control_charts"]:
        print(c)
