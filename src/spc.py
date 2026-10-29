"""Control-chart calculations: individuals (I-MR) limits, p-chart with varying sample size,
and the four Western Electric run rules.

All functions take plain numpy arrays and return plain data, so they can be tested without plots.
"""
import numpy as np

D2_N2 = 1.128  # d2 constant for moving ranges of two consecutive points


def imr_limits(baseline):
    """Center line and sigma for an individuals chart, from a baseline (Phase I) sample.

    sigma is estimated from the average moving range (MR-bar / d2), not the sample standard deviation,
    so a shift inside the baseline inflates sigma less.
    """
    x = np.asarray(baseline, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        raise ValueError("need at least 2 baseline points")
    mr_bar = np.mean(np.abs(np.diff(x)))
    return {"center": float(np.mean(x)), "sigma": float(mr_bar / D2_N2), "mr_bar": float(mr_bar)}


def p_chart(defectives, sample_sizes):
    """p-chart with limits that vary with each subgroup's size. Returns p, center, per-point limits and z."""
    d = np.asarray(defectives, dtype=float)
    n = np.asarray(sample_sizes, dtype=float)
    if np.any(n <= 0):
        raise ValueError("sample sizes must be positive")
    p = d / n
    pbar = d.sum() / n.sum()
    se = np.sqrt(pbar * (1 - pbar) / n)
    return {"p": p, "center": pbar, "ucl": pbar + 3 * se, "lcl": np.clip(pbar - 3 * se, 0, None),
            "z": (p - pbar) / se}


def western_electric(z):
    """Apply the four Western Electric rules to standardized values z = (x - center) / sigma.

    Returns a list of sets: rules triggered at each point (the point that completes the pattern).
      1: one point beyond 3 sigma
      2: two of three consecutive points beyond 2 sigma, same side
      3: four of five consecutive points beyond 1 sigma, same side
      4: eight consecutive points on the same side of the center line
    """
    z = np.asarray(z, dtype=float)
    out = [set() for _ in z]
    for i in range(len(z)):
        if abs(z[i]) > 3:
            out[i].add(1)
        for side in (1, -1):
            s = side * z
            if i >= 2 and s[i] > 2 and np.sum(s[i - 2:i + 1] > 2) >= 2:
                out[i].add(2)
            if i >= 4 and s[i] > 1 and np.sum(s[i - 4:i + 1] > 1) >= 4:
                out[i].add(3)
            if i >= 7 and np.all(s[i - 7:i + 1] > 0):
                out[i].add(4)
    return out
