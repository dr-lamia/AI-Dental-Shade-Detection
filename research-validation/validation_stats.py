"""Descriptive agreement statistics for the shade-validation app."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(b - a)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((b - a) ** 2)))


def concordance_correlation_coefficient(y_true, y_pred) -> float:
    """Lin's CCC for paired continuous measurements."""
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float((2 * cov) / denom)


def bland_altman(y_true, y_pred) -> dict:
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    means = (x + y) / 2.0
    diffs = y - x
    bias = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else float("nan")
    return {
        "means": means,
        "diffs": diffs,
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
    }
