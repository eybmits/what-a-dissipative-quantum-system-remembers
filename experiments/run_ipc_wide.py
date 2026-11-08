"""Information-processing capacity with a basis wide enough to close the budget.

For a fading-memory reservoir with linearly independent readout features, the
capacities over a complete orthogonal basis of input-history functions sum to
the number of independent features (Dambre et al., Sci. Rep. 2, 514 (2012)).
A truncated basis therefore reports a shortfall that measures what the basis
missed, not what the reservoir lacks.  An earlier version of this analysis cut
the delays at 12, and the shared channel is exactly the design whose linear
memory extends beyond that.

This driver answers the question directly.  It reports

* the numerical rank of the feature covariance, which is the ceiling;
* the degree-one capacity accumulated over growing delay windows, which shows
  whether the earlier shortfall was truncation;
* degree-resolved capacity over a basis with long delay windows, and the
  total as a fraction of the rank;
* the linear memory at every delay up to 60.

Targets are products of Legendre polynomials of inputs drawn uniformly on
[-1, 1], each evaluated from one thin QR factorization of the centred feature
matrix; capacities below a shuffled-surrogate noise floor are discarded.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import sys
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import _paths  # noqa: F401
import numpy as np
from numpy.polynomial import legendre

import run_final_scaling as rfs
from qrc import dissipators as dsp, readout, reservoirs as res
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir

SIZE = 5
DESIGNS = (("uniform local", "CD_paper"), ("collective", "B3_collective"))
DEGREE_ONE_WINDOWS = (12, 20, 30, 40, 60)

# (polynomial orders, delay window) per structural class of target.
# Single-factor terms reach far back; products are confined to shorter windows,
# where their capacity is concentrated.
CLASSES = (
    ((1,), 60),
    ((2,), 40), ((1, 1), 24),
    ((3,), 30), ((2, 1), 16), ((1, 1, 1), 16),
    ((4,), 24), ((3, 1), 12), ((2, 2), 12), ((2, 1, 1), 12), ((1, 1, 1, 1), 12),
)


def legendre_value(order, x):
    coefficients = np.zeros(order + 1)
    coefficients[order] = 1.0
    return legendre.legval(x, coefficients)


def targets():
    """Yield (degree, max delay used, {delay: order}) without duplicates."""
    seen = set()
    for orders, window in CLASSES:
        k = len(orders)
        for delays in itertools.permutations(range(1, window + 1), k):
            key = tuple(sorted(zip(delays, orders)))
            if key in seen:
                continue
            seen.add(key)
            yield sum(orders), max(delays), dict(zip(delays, orders))


def evaluate(features, raw, washout, rng):
    centred = features - features.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-9))
    Q, _ = np.linalg.qr(np.column_stack([centred, np.ones(len(centred))]))
    T = len(centred)

    def capacity(target):
        target = target - target.mean()
        return float(np.sum((Q.T @ target) ** 2) / float(target @ target))

    floor = float(np.percentile(
        [capacity(rng.permutation(raw[washout: washout + T])) for _ in range(300)], 99))

    by_degree = defaultdict(float)
    degree_one = defaultdict(float)
    by_delay = {}
    count = 0
    for degree, reach, factors in targets():
        product = np.ones(T)
        for delay, order in factors.items():
            product = product * legendre_value(order, raw[washout - delay: washout - delay + T])
        value = capacity(product)
        count += 1
        if degree == 1:
            by_delay[reach] = value
        if value <= floor:
            continue
        by_degree[degree] += value
        if degree == 1:
            for window in DEGREE_ONE_WINDOWS:
                if reach <= window:
                    degree_one[window] += value
    return {"rank": rank, "noise_floor": floor, "targets": count,
            "by_degree": {str(k): v for k, v in sorted(by_degree.items())},
            "degree_one_cumulative": {str(k): v for k, v in sorted(degree_one.items())},
            "degree_one_by_delay": [by_delay[k] for k in sorted(by_delay)],
            "total": float(sum(by_degree.values()))}


def run(method, seed, length, washout):
    weight = dsp.jump_strength(dsp.local_loss(SIZE, rfs.GAMMA))
    rng = np.random.default_rng(seed)
    J = res.random_couplings(SIZE, 1.0, rng)
    jumps = rfs.build_jumps(method, J, SIZE, weight, np.random.default_rng(seed + 1))
    H0, Hx = ising_xx_hamiltonian(J, 0.5, SIZE), transverse_drive(SIZE)
    reservoir = SparseLindbladReservoir.from_terms(SIZE, H0 + 0.5 * Hx, 0.5 * Hx, jumps, 0.5)
    raw = rng.uniform(-1.0, 1.0, length)
    features = reservoir.run((raw + 1.0) / 2.0,
                             readout.pauli_observables(SIZE, max_weight=2), washout=washout)
    return features, raw


def run_case(task):
    label, method, seed, length, washout = task
    features, raw = run(method, seed, length, washout)
    result = evaluate(features, raw, washout, np.random.default_rng(seed + 11))
    return {"label": label, "method": method, "seed": seed, **result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--length", type=int, default=11000)
    ap.add_argument("--washout", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/ipc_wide")
    a = ap.parse_args()

    todo = [(label, method, seed, a.length, a.washout)
            for label, method in DESIGNS for seed in rfs.deterministic_seeds(a.seeds)]
    path = os.path.join(a.out, "ipc_wide.json")
    records = json.load(open(path))["records"] if os.path.exists(path) else []     # runs are deterministic
    kept = {(r["label"], r["seed"]) for r in records}
    with Pool(a.workers) as pool:
        for result in pool.imap_unordered(run_case, [t for t in todo if (t[0], t[2]) not in kept]):
            records.append(result)
            print(f"  {result['label']} seed {result['seed']}: rank {result['rank']}, "
                  f"total {result['total']:.2f}", flush=True)
    records.sort(key=lambda r: (r["label"], r["seed"]))

    print(f"\n=== Capacity with a wide basis (N={SIZE}, {a.seeds} paired reservoirs) ===")
    print(f"  targets per reservoir: {records[0]['targets']}")
    for label, method in DESIGNS:
        mine = [r for r in records if r["method"] == method]
        rank = np.mean([r["rank"] for r in mine])
        total = np.mean([r["total"] for r in mine])
        print(f"\n  {label}: rank {rank:.1f}, total capacity {total:.2f} "
              f"({100 * total / rank:.1f}% of rank)")
        degrees = sorted({d for r in mine for d in r["by_degree"]}, key=int)
        print("    by degree: " + "  ".join(
            f"{d}:{np.mean([r['by_degree'].get(d, 0) for r in mine]):.2f}" for d in degrees))
        print("    degree-one capacity up to delay W: " + "  ".join(
            f"W={w}:{np.mean([r['degree_one_cumulative'].get(str(w), 0) for r in mine]):.2f}"
            for w in DEGREE_ONE_WINDOWS))

    os.makedirs(a.out, exist_ok=True)
    with open(path, "w") as h:
        json.dump({"protocol": {"n_qubits": SIZE, "length": a.length, "washout": a.washout,
                                "classes": [[list(o), w] for o, w in CLASSES]},
                   "records": records}, h, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
