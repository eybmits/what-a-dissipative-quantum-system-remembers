#!/usr/bin/env python3
"""Linear memory capacity over loss rate and input amplitude: is there a best amplitude? (Appendix E.4)

The five-spin reservoir of run_clock.py with the uniform input, for independent decay ("separate") at
gamma = 2^-8 ... 2^-2 and 1 and for the opposite-phase channel ("signed") at gamma = 1, 4 and 2^3 ... 2^8, at fifteen
amplitudes eta = 2^-7 ... 1 in steps of sqrt(2), on the first four reservoirs. Features are exact or sampled with
10^6, 10^7 and 10^8 preparations per setting and step from the state with the full history. The lifetime of a loss
rate is the single-input recall time at eta = 2^-8, where mixing is negligible (geometric mean over the reservoirs).

The analysis reports, for the loss rates with a lifetime of at least four steps:
  exact readout     the amplitude of the largest capacity, and how the largest amplitude that keeps 90% of it scales
                    with the lifetime
  finite sampling   the amplitude of the largest capacity (parabola in log2 eta through the maximum and its
                    neighbours) and its scaling with the lifetime; the capacity that one amplitude for all loss rates
                    recovers
and the same without the loss rates whose exact recall over delays 281-300 exceeds three times the floor 1/1600
(their memory reaches past the 300 delays summed).

Run with:  python experiments/run_amplitude_map.py [--workers 6]     (resumes; about two hours on six cores)
           python experiments/run_amplitude_map.py --analyse         (prints the numbers of the paper)
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "theory" / "finite_shots_vs_gamma"))

import numpy as np  # noqa: E402

OUT = HERE.parent / "results" / "amplitude_map"
GAMMAS = {"separate": [2.0 ** k for k in range(-8, -1)] + [1.0], "signed": [1.0, 4.0] + [2.0 ** k for k in range(3, 9)]}
ETAS = [2.0 ** (k / 2) for k in range(-14, 1)]
SHOTS = (10 ** 6, 10 ** 7, 10 ** 8)
FLOOR = 1 / 1600


def case(task):
    import run_clock as rc
    if task[0] == "lifetime":
        _, seed, name, gamma = task
        r = rc.clock_case((seed, name, gamma, 2.0 ** -8))
        return "lifetime", {"seed": seed, "design": name, "gamma": gamma, "forget": r["forget"]}
    _, seed, name, gamma, eta = task
    P, post = rc.simulate(seed, name, gamma, eta)
    out = rc.scores(P, post, seed, shots=SHOTS)
    return "map", {"seed": seed, "design": name, "gamma": gamma, "eta": eta, "exact": out["exact"],
                   "tail": float(np.mean(out["curve"][-20:])), "var_mean": out["var_mean"],
                   **{f"shots_{n}": out[f"shots_{n}"] for n in SHOTS}}


def run(workers):
    import run_final_scaling as rfs
    seeds = rfs.deterministic_seeds(8)[:4]
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    for kind in ("map", "lifetime"):
        path = OUT / f"{kind}.jsonl"
        if path.exists():
            done |= {(kind, r["seed"], r["design"], r["gamma"], r.get("eta")) for r in map(json.loads, path.open())}
    tasks = [("lifetime", s, n, g) for n in GAMMAS for g in GAMMAS[n] for s in seeds]
    tasks += [("map", s, n, g, e) for n in GAMMAS for g in GAMMAS[n] for e in ETAS for s in seeds]
    tasks = [t for t in tasks if (*t[:4], t[4] if t[0] == "map" else None) not in done]
    print(f"{len(tasks)} cases to run", flush=True)
    files = {kind: (OUT / f"{kind}.jsonl").open("a") for kind in ("map", "lifetime")}
    with get_context("spawn").Pool(workers) as pool:
        for i, (kind, record) in enumerate(pool.imap_unordered(case, tasks), 1):
            files[kind].write(json.dumps(record) + "\n")
            files[kind].flush()
            print(f"{i}/{len(tasks)}", flush=True)


# ------------------------------------------------------------------------------------------------- analysis
def load():
    recs = [json.loads(line) for line in (OUT / "map.jsonl").open()]
    life = defaultdict(list)
    for r in map(json.loads, (OUT / "lifetime.jsonl").open()):
        life[(r["design"], r["gamma"])].append(r["forget"])
    return recs, {k: float(np.exp(np.mean(np.log(v)))) for k, v in life.items()}


def table(recs, design, field):
    """Mean over the reservoirs of one field, as an array over (loss rate, amplitude)."""
    cell = defaultdict(list)
    for r in recs:
        if r["design"] == design:
            cell[(r["gamma"], r["eta"])].append(r[field])
    return np.array([[np.mean(cell[(g, e)]) for e in ETAS] for g in GAMMAS[design]])


def knee(c, frac=0.9):
    """Largest amplitude with at least `frac` of the best capacity, interpolated in log2 eta; None if eta = 1."""
    x, level = np.log2(ETAS), frac * c.max()
    i = int(np.max(np.nonzero(c >= level)[0]))
    if i == len(c) - 1:
        return None
    return float(2.0 ** (x[i] + (c[i] - level) / (c[i] - c[i + 1]) * (x[i + 1] - x[i])))


def peak(c):
    """Amplitude of the largest capacity, from a parabola through it and its neighbours; None at an edge."""
    x, i = np.log2(ETAS), int(np.argmax(c))
    if i in (0, len(c) - 1):
        return None
    y0, y1, y2 = c[i - 1], c[i], c[i + 1]
    return float(2.0 ** (x[i] + np.clip(0.5 * (y0 - y2) / (y0 - 2 * y1 + y2), -1, 1) * (x[i + 1] - x[i])))


def exponent(lifetimes, amplitudes):
    """Slope of log amplitude against log lifetime with its 95% interval."""
    from scipy import stats
    pairs = [(t, a) for t, a in zip(lifetimes, amplitudes) if a is not None]
    if len(pairs) < 3:
        return "too few points"
    x, y = np.log([p[0] for p in pairs]), np.log([p[1] for p in pairs])
    fit = stats.linregress(x, y)
    half = stats.t.ppf(0.975, len(x) - 2) * fit.stderr
    return f"{fit.slope:+.2f} [{fit.slope - half:+.2f}, {fit.slope + half:+.2f}] from {len(x)} loss rates"


def analyse():
    recs, life = load()
    for design in GAMMAS:
        gam = GAMMAS[design]
        exact, tail = table(recs, design, "exact"), table(recs, design, "tail")
        t = np.array([life[(design, g)] for g in gam])
        print(f"\n{design}")
        for i, g in enumerate(gam):
            print(f"  gamma {g:9.4g}  lifetime {t[i]:6.1f}  best exact capacity {exact[i].max():5.2f} at eta "
                  f"{ETAS[int(np.argmax(exact[i]))]:.4f}  at eta = 1: {exact[i, -1]:5.2f}  "
                  f"largest recall over delays 281-300: {tail[i].max() / FLOOR:5.1f} floors")
        weakest = sum(int(np.argmax(row)) == 0 for row in exact)
        print(f"  exact readout: best at the weakest amplitude at {weakest} of {len(gam)} loss rates")
        for label, keep in (("all loss rates with a lifetime of at least four steps", t >= 4),
                            ("without recall past 300 delays", (t >= 4) & (tail.max(axis=1) <= 3 * FLOOR))):
            print(f"  {label}: gamma = {[g for g, k in zip(gam, keep) if k]}")
            print(f"    amplitude keeping 90% of the best exact capacity: exponent {exponent(t[keep], [knee(c) for c in exact[keep]])}")
            for n in SHOTS:
                rows = table(recs, design, f"shots_{n}")[keep]
                tracked = rows.max(axis=1)
                j = int(np.argmax(rows.mean(axis=0)))
                share = rows[:, j] / tracked
                print(f"    optimum, {n:.0e} preparations: exponent {exponent(t[keep], [peak(c) for c in rows])}; "
                      f"eta = {ETAS[j]:.3f} at every loss rate keeps {share.min():.2f}-{share.max():.2f} "
                      f"(mean {share.mean():.3f}) of the tracked optimum; best capacity {tracked.max():.1f} tracked, "
                      f"{rows[:, j].max():.1f} with this amplitude")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args()
    analyse() if args.analyse else run(args.workers)
