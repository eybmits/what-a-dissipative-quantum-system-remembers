#!/usr/bin/env python3
"""Memory against the loss rate at three, four and six spins (the curves of Fig. 2c at other sizes).

Same Hamiltonian, input stream, washout, readout and seeds as the five-spin study, for independent decay
("separate"), the collective channel with equal phases ("shared") and the collective channel with opposite
phases ("signed"), whose last floor(N/2) spins couple with reversed sign: (1,1,-1) at three spins, as in the
three-spin limits, and (1,1,1,-1,-1) at five. Every channel has Tr K = N 2^(N-1) gamma. The features are the
3N + 3N(N-1)/2 expectation values of X_i, Y_i, Z_i and X_iX_j, Y_iY_j, Z_iZ_j; exact values, no sampling.

fixed          full input amplitude
gentle         amplitude min(1, gamma^(1/2)) for independent decay and min(1, gamma^(-1/2)) for opposite phases

The generator is built for any N from the single-site matrices of left and right multiplication in the Pauli
basis. `--check` compares N = 5 with the records of the five-spin study in results/dark_projection/memory.jsonl.

Run with:  python experiments/run_sizes.py [--sizes 3 4 6] [--seeds 8] [--workers 4]   (resumes)
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "theory" / "finite_shots_vs_gamma"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import pscore  # noqa: E402
import psim  # noqa: E402
import run_final_scaling as rfs  # noqa: E402

OUT = HERE.parent / "results" / "sizes"
H_FIELD, NODES = 0.5, 10
TRAIN, VALIDATION, TEST = 2000, 400, 1600
DELAYS = tuple(range(1, 301))
DESIGNS = ("separate", "shared", "signed")
GAMMAS = {3: tuple(float(2.0 ** k) for k in range(-5, 9)), 4: tuple(float(2.0 ** k) for k in range(-5, 9)),
          5: (0.25, 1.0, 16.0, 64.0), 6: tuple(float(4.0 ** k) for k in range(-2, 4))}

_S = (np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0]))
# matrices of rho -> sigma_a rho and rho -> rho sigma_a on the coordinates Tr(sigma_c rho) of one spin
LEFT = [sp.csr_matrix(np.array([[np.trace(c @ a @ b) / 2 for b in _S] for c in _S])) for a in _S]
RIGHT = [sp.csr_matrix(np.array([[np.trace(c @ b @ a) / 2 for b in _S] for c in _S])) for a in _S]


def product(table, n, letters):
    """Left or right multiplication by the Pauli string with the given letters {site: 1, 2 or 3}."""
    out = sp.identity(1, dtype=complex, format="csr")
    for i in range(n):
        out = sp.kron(out, table[letters.get(i, 0)], format="csr")
    return out


def commutator(n, letters):
    """rho -> -i [P, rho]."""
    return -1j * (product(LEFT, n, letters) - product(RIGHT, n, letters))


def lowering(coeffs):
    """Pauli expansion of L = sum_i c_i sigma^-_i with sigma^- = |0><1| = (X + iY)/2."""
    return [term for i, c in enumerate(coeffs) if c for term in (({i: 1}, 0.5 * c), ({i: 2}, 0.5j * c))]


def dissipator(n, terms):
    """rho -> L rho L^+ - {L^+ L, rho}/2 for L = sum_k l_k P_k."""
    total = sp.csr_matrix((4 ** n, 4 ** n), dtype=complex)
    left = [product(LEFT, n, letters) for letters, _ in terms]
    right = [product(RIGHT, n, letters) for letters, _ in terms]
    for k, (_, lk) in enumerate(terms):
        for m, (_, lm) in enumerate(terms):
            total = total + lk * np.conj(lm) * (left[k] @ right[m] - 0.5 * left[m] @ left[k]
                                                - 0.5 * right[k] @ right[m])
    return total


def phases(n):
    return [1.0] * (n - n // 2) + [-1.0] * (n // 2)


def channel(n, name):
    if name == "separate":
        return sum(dissipator(n, lowering([float(k == i) for k in range(n)])) for i in range(n))
    return dissipator(n, lowering([1.0] * n if name == "shared" else phases(n)))


def real(matrix):
    dense = matrix.toarray()
    assert np.abs(dense.imag).max() < 1e-12
    return np.ascontiguousarray(dense.real)


def index(n, letters):
    return sum(a * 4 ** (n - 1 - i) for i, a in letters.items())


def features(n):
    singles = [index(n, {i: a}) for a in (1, 2, 3) for i in range(n)]
    pairs = [index(n, {i: a, j: a}) for a in (1, 2, 3) for i in range(n) for j in range(i + 1, n)]
    return np.array(singles + pairs)


def amplitude(name, rule, gamma):
    if rule == "fixed" or name == "shared":
        return 1.0
    return float(min(1.0, gamma ** (0.5 if name == "separate" else -0.5)))


def case(task):
    n, seed, name, gamma, rule = task
    eta = amplitude(name, rule, gamma)
    wash = int(max(800, 60 * gamma))
    rng = np.random.default_rng(seed)                 # the couplings first, then the input stream, as in psim
    J = {(i, j): rng.uniform(-1.0, 1.0) for i in range(n) for j in range(i + 1, n)}
    inputs = rng.uniform(0.0, 1.0, size=wash + TRAIN + VALIDATION + TEST)
    field_x = sum(commutator(n, {i: 1}) for i in range(n))
    static = H_FIELD * sum(commutator(n, {i: 3}) for i in range(n)) + 1.5 * H_FIELD * field_x
    for (i, j), value in J.items():
        static = static + value * commutator(n, {i: 1, j: 1})
    G1 = real(eta * H_FIELD * field_x)                # the input enters as h [3/2 + eta (u - 1/2)] sum_i X_i
    p = psim.Propagator(real(static + gamma * channel(n, name)) - 0.5 * G1, G1, nodes=NODES)
    feat = features(n)
    r = np.zeros(4 ** n)                              # |0...0><0...0|: one on every string of I and Z
    r[[index(n, {i: 3 for i in range(n) if m >> i & 1}) for m in range(2 ** n)]] = 1.0
    rows = np.empty((len(inputs) - wash, len(feat)))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            rows[k - wash] = r[feat]
    caps, _ = pscore.selected(rows, inputs[wash:], DELAYS, tr=TRAIN, va=VALIDATION, te=TEST, ridges=(0.0,))
    return {"spins": n, "seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta,
            "features": len(feat), "capacity": float(caps.sum()), "tail": float(caps[-20:].sum()),
            "half": int(np.searchsorted(np.cumsum(caps), 0.5 * caps.sum())) + 1}


def jobs(sizes, seeds):
    out = []
    for n in sizes:
        for seed in seeds[:4] if n == 6 else seeds:
            for name in DESIGNS:
                for gamma in GAMMAS[n]:
                    out.append((n, seed, name, gamma, "fixed"))
                    if amplitude(name, "gentle", gamma) < 1.0:
                        out.append((n, seed, name, gamma, "gentle"))
    return out


def check():
    """N = 5 against the five-spin study: the same reservoirs must give the same capacities."""
    path = HERE.parent / "results" / "dark_projection" / "memory.jsonl"
    stored = {(r["seed"], r["design"], r["gamma"]): sum(r["memory"]) for r in map(json.loads, path.open())}
    seed = rfs.deterministic_seeds(1)[0]
    worst = 0.0
    for name in DESIGNS:
        for gamma in (1.0, 16.0):
            got = case((5, seed, name, gamma, "fixed"))["capacity"]
            worst = max(worst, abs(got - stored[(seed, name, gamma)]))
            print(name, gamma, round(got, 4), round(stored[(seed, name, gamma)], 4), flush=True)
    print("largest difference", worst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 6])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "memory.jsonl"
    key = lambda r: (r["spins"], r["seed"], r["design"], r["gamma"], r["rule"])
    done = {key(json.loads(line)) for line in path.open()} if path.exists() else set()
    todo = [t for t in jobs(args.sizes, rfs.deterministic_seeds(args.seeds)) if t not in done]
    print(f"{len(todo)} cases to run, {len(done)} done", flush=True)
    with get_context("spawn").Pool(args.workers) as pool, path.open("a") as handle:
        for i, record in enumerate(pool.imap_unordered(case, todo), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
