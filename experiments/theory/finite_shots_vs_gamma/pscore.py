"""Independent shot sampler and read-out scoring.  No repo code.

Sampler: dyadic binomial splitting of n shots down the 5 bits of the outcome
(the repo uses rng.multinomial).  Read-out: SVD ridge (the repo solves normal
equations).  Same split and same selection rule as the study:
fit on rows [D, 480), pick one ridge by summed capacity on [480, 600),
refit on [D, 600), score on [600, 1000), with D = largest delay.
"""
from __future__ import annotations

import numpy as np

import psim

RIDGES = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, np.inf)
SINGLES = [psim.bit(i) for i in range(psim.N)]
PAIRS = [psim.bit(i) | psim.bit(j) for i in range(psim.N) for j in range(i + 1, psim.N)]
ORDER45 = [(a, m) for a in range(3) for m in SINGLES] + [(a, m) for a in range(3) for m in PAIRS]
ORDER93 = [(a, m) for a in range(3) for m in range(1, 32)]


def tree_counts(p, n, rng):
    """p: (T, 32) probabilities, n shots each row.  Returns counts (T, 32).

    Top-down: split n on the highest bit, then each part on the next bit, ...
    The sums of p over dyadic blocks give the conditional probabilities."""
    levels = [p]
    while levels[-1].shape[1] > 1:
        q = levels[-1]
        levels.append(q[:, ::2] + q[:, 1::2])
    counts = np.full((p.shape[0], 1), n, dtype=np.int64)
    for q in reversed(levels[:-1]):
        parent = q[:, ::2] + q[:, 1::2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(parent > 0, q[:, ::2] / parent, 0.5)
        left = rng.binomial(counts, np.clip(ratio, 0.0, 1.0))
        next_counts = np.empty((p.shape[0], counts.shape[1] * 2), dtype=np.int64)
        next_counts[:, ::2] = left
        next_counts[:, 1::2] = counts - left
        counts = next_counts
    return counts


def features_from_probs(P, order=ORDER45):
    """P (T, 3, 32) -> Pauli features in the given order."""
    W = P @ psim.HAD            # (T, 3, 32), column m = <P^m>
    return np.column_stack([W[:, a, m] for a, m in order])


def sampled_features(P, n, rng, order=ORDER45):
    F = np.empty(P.shape)
    for a in range(3):
        F[:, a, :] = tree_counts(P[:, a, :], n, rng) / n
    return features_from_probs(F, order)


def svd_fit(X, Y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    return U.T @ Y, s, Vt


def weights(fit, ridge, X=None, Y=None):
    UY, s, Vt = fit
    if ridge == np.inf:
        W = Vt.T @ (s[:, None] * UY)
        norm = np.linalg.norm(W, axis=0, keepdims=True)
        return W / np.where(norm > 1e-30, norm, 1.0)
    if ridge == 0.0:
        keep = s > max(X.shape) * np.finfo(float).eps * s[0]
        return Vt.T[:, keep] @ (UY[keep] / s[keep, None])
    return Vt.T @ ((s / (s ** 2 + ridge))[:, None] * UY)


def r2(y, f):
    y = y - y.mean(axis=0)
    f = f - f.mean(axis=0)
    den = np.sum(y ** 2, axis=0) * np.sum(f ** 2, axis=0)
    return np.divide(np.sum(y * f, axis=0) ** 2, den, out=np.zeros(y.shape[1]), where=den >= 1e-30)


def targets(post, delays):
    D = max(delays)
    return np.column_stack([post[D - d:len(post) - d] for d in delays])


def selected(X, post, delays, tr=480, va=120, te=400, ridges=RIDGES):
    """Returns (test capacities by delay, chosen ridge).  X has no bias column."""
    D = max(delays)
    Y = targets(post, delays)                       # rows D ...
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])[D:]
    a, b, c = tr - D, tr + va - D, tr + va + te - D
    fit = svd_fit(Xb[:a], Y[:a])
    best, pick = -1.0, None
    for ridge in ridges:
        w = weights(fit, ridge, Xb[:a], Y[:a])
        v = float(np.sum(r2(Y[a:b], Xb[a:b] @ w)))
        if v > best:
            best, pick = v, ridge
    fit = svd_fit(Xb[:b], Y[:b])
    w = weights(fit, pick, Xb[:b], Y[:b])
    return r2(Y[b:c], Xb[b:c] @ w), pick


def two_scores(X, post):
    c20, r20 = selected(X, post, tuple(range(1, 21)))
    c60, r60 = selected(X, post, tuple(range(1, 61)))
    return float(c20.sum()), float(c60.sum()), r20, r60


def noise_cov(P, order=ORDER45):
    """Time-averaged single-shot covariance of the features.  P (T, 3, 32)."""
    p = len(order)
    C = np.zeros((p, p))
    for a in range(3):
        idx = [i for i, (ax, _) in enumerate(order) if ax == a]
        masks = [order[i][1] for i in idx]
        E = psim.HAD[:, masks]                       # (32, 15)
        pa = P[:, a, :]
        second = (E.T * pa.mean(axis=0)) @ E
        mean = pa @ E
        C[np.ix_(idx, idx)] = second - mean.T @ mean / P.shape[0]
    return C
