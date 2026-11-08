#!/usr/bin/env python3
"""Exact memory of the three-spin reservoir with one shared jump, without shot noise.

For each Hamiltonian, loss rate and input strength this computes, for the
physical observable X_1 and from stationary moments only:

  orders      share of the ideal memory carried by products of k input times,
              k = 1..K, from the order recursion G_{k+1} = (I - B)^{-1} D G_k D^T
  mean order  sum_k k C_k
  single      share of strictly past single inputs, a^T A G_1 A^T a / a^T S a
  size        signal variance a^T S a of X_1
  range       delays that hold 90% of the single-input recall

Model as in the Nov 2026 manuscript: H = sum_{i<j} J_ij X_i X_j + Z/2 + (3 + eta u) X / 4,
one jump sqrt(gamma) (s1 - s2 - s3), steps of 1/2, binary inputs u = +-1.

prefactor   share of single past inputs and signal size against the prefactor c of the gentle-writing amplitude
            eta = c / sqrt(gamma), at gamma = 4096

Run with:  python experiments/run_spin_orders.py [extra | prefactor]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import itertools
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres

OUT = Path(__file__).resolve().parent.parent / "results" / "spin_orders"
SEEDS = (8675309, 424242, 97531, 123457)
GAMMAS = tuple(float(2 ** k) for k in range(4, 17))
DRIVES = ("strong", "gentle")
PAULI = (np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))


def tensor(factors):
    out = np.ones((1, 1))
    for f in factors:
        out = np.kron(out, f)
    return out


class SpinModel:
    """Pauli-transfer generators of the three-spin reservoir (traceless block)."""

    def __init__(self, seed, n=3):
        self.n, self.d = n, 2 ** n
        labels = list(itertools.product(range(4), repeat=n))
        ops = [tensor(PAULI[a] for a in lab) for lab in labels]
        self.basis = np.column_stack([p.reshape(-1, order="F") for p in ops]) / np.sqrt(self.d)
        index = {lab: i for i, lab in enumerate(labels)}
        op = lambda axis, sites: ops[index[tuple(axis if i in sites else 0 for i in range(n))]]
        rng = np.random.default_rng(seed)
        couplings = [rng.uniform(-1, 1) for _ in itertools.combinations(range(n), 2)]
        xsum = sum(op(1, [i]) for i in range(n))
        H = (sum(j * op(1, pair) for j, pair in zip(couplings, itertools.combinations(range(n), 2)))
             + 0.5 * sum(op(3, [i]) for i in range(n)) + 0.75 * xsum)
        lowering = np.array([[0, 1], [0, 0]])
        jump = sum(c * tensor(lowering if k == i else PAULI[0] for k in range(n))
                   for i, c in enumerate((1, -1, -1)))
        eye = np.eye(self.d)
        K = jump.conj().T @ jump
        loss = np.kron(jump.conj(), jump) - (np.kron(eye, K) + np.kron(K.T, eye)) / 2
        comm = lambda h: -1j * (np.kron(eye, h) - np.kron(h.T, eye))
        self.static, self.drive, self.loss = (self.ptm(comm(H)), self.ptm(comm(0.25 * xsum)), self.ptm(loss))
        self.x1 = np.zeros(len(labels) - 1)
        self.x1[index[(1, 0, 0)] - 1] = 1.0          # coordinate of X_1 / sqrt(8)

    def ptm(self, generator):
        value = self.basis.conj().T @ generator @ self.basis
        assert np.max(np.abs(value.imag)) < 2e-12
        return value.real

    def maps(self, gamma, eta):
        out = []
        for u in (-1.0, 1.0):
            E = la.expm(0.5 * (self.static + gamma * self.loss + u * eta * self.drive))
            out.append((E[1:, 1:], E[1:, 0]))
        return out


class Stein:
    """Solve X - A X A^T = C through a cached real Schur form (bilinear transform)."""

    def __init__(self, A):
        eye = np.eye(len(A))
        inv = la.solve(A + eye, eye)
        self.tri, self.orth = la.schur((A - eye) @ inv, output="real")
        self.left = self.orth.T @ inv
        self.trsyl = la.get_lapack_funcs("trsyl", (self.tri,))

    def solve(self, C):
        x, scale, info = self.trsyl(self.tri, self.tri, -2 * self.left @ C @ self.left.T, trana="N", tranb="T")
        assert info == 0
        y = self.orth @ (x / scale) @ self.orth.T
        return (y + y.T) / 2


def solve_full(C, A, D, stein):
    shape = A.shape
    op = LinearOperator((A.size, A.size), matvec=lambda v: (v.reshape(shape) - A @ v.reshape(shape) @ A.T
                                                            - D @ v.reshape(shape) @ D.T).ravel())
    pre = LinearOperator(op.shape, matvec=lambda z: stein.solve(z.reshape(shape)).ravel())
    sol, info = gmres(op, C.ravel(), M=pre, rtol=1e-11, atol=0.0, restart=70, maxiter=200)
    if info != 0:       # ill-conditioned at the largest loss rates: restart longer, then accept 1e-9
        sol, info = gmres(op, C.ravel(), x0=sol, M=pre, rtol=1e-9, atol=0.0, restart=200, maxiter=1000)
    assert info == 0
    sol = sol.reshape(shape)
    return (sol + sol.T) / 2


def case(task):
    seed, drive, gamma = task
    model = SpinModel(seed)
    eta = 1.0 if drive == "strong" else 8.0 / np.sqrt(gamma)
    (Am, bm), (Ap, bp) = model.maps(gamma, eta)
    A, D = (Ap + Am) / 2, (Ap - Am) / 2
    mean = la.solve(np.eye(len(A)) - A, (bp + bm) / 2)
    v = D @ mean + (bp - bm) / 2
    stein = Stein(A)
    sigma = solve_full(np.outer(v, v), A, D, stein)
    a = model.x1
    size = float(a @ sigma @ a)
    G = stein.solve(np.outer(v, v))                     # order one, delay zero included
    past = A @ G @ A.T
    single = float(a @ past @ a) / size
    # order shares C_k = a^T G_k a / size until 1 - 1e-4 of the memory is accounted for
    shares, total, k = [], 0.0, 0
    while total < 1 - 1e-4 and k < 200000:
        c = float(a @ G @ a) / size
        shares.append(c)
        total += c
        k += 1
        G = stein.solve(D @ G @ D.T)
    shares = np.array(shares)
    mean_order = float(np.sum(np.arange(1, len(shares) + 1) * shares) / shares.sum())
    # 90% range of the single-input recall
    target, h, P = 0.1 * float(a @ past @ a), 0, A.copy()
    lo, hi = 0, 1
    tail = lambda t: float(a @ np.linalg.matrix_power(A, t) @ past @ np.linalg.matrix_power(A, t).T @ a)
    while tail(hi) > target:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if tail(mid) > target else (lo, mid)
    # keep the order shares compact: exact up to 64, then logarithmic bins
    edges = np.unique(np.r_[np.arange(1, 65), np.round(np.geomspace(65, len(shares) + 2, 80)).astype(int)])
    binned = [float(shares[e0 - 1:e1 - 1].sum()) for e0, e1 in zip(edges[:-1], edges[1:])]
    return {"seed": seed, "drive": drive, "gamma": gamma, "eta": eta, "size": size, "single": single,
            "mean_order": mean_order, "accounted": float(total), "orders_counted": len(shares),
            "range": hi, "edges": edges.tolist(), "binned": binned}


PREFACTORS = tuple(float(c) for c in np.geomspace(0.5, 64.0, 43))


def prefactor_case(task):
    seed, c = task
    gamma = 4096.0
    model = SpinModel(seed)
    (Am, bm), (Ap, bp) = model.maps(gamma, c / np.sqrt(gamma))
    A, D = (Ap + Am) / 2, (Ap - Am) / 2
    mean = la.solve(np.eye(len(A)) - A, (bp + bm) / 2)
    v = D @ mean + (bp - bm) / 2
    stein = Stein(A)
    sigma = solve_full(np.outer(v, v), A, D, stein)
    a = model.x1
    size = float(a @ sigma @ a)
    G = stein.solve(np.outer(v, v))
    return {"seed": seed, "gamma": gamma, "prefactor": c, "size": size, "single": float(a @ (A @ G @ A.T) @ a) / size}


def prefactor_main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(s, c) for s in SEEDS for c in PREFACTORS]
    with get_context("spawn").Pool(2) as pool, (OUT / "prefactor.jsonl").open("w") as fh:
        for rec in pool.imap_unordered(prefactor_case, tasks):
            fh.write(json.dumps(rec) + "\n")


def main(extra=False):
    """The grid, or with ``extra`` two further loss rates for the strong input without order shares."""
    OUT.mkdir(parents=True, exist_ok=True)
    if extra:
        path = OUT / "orders_extra.jsonl"
        done = set()
        if path.exists():
            done = {(r["seed"], r["drive"], r["gamma"]) for r in map(json.loads, path.open())}
        tasks = [(s, d, g) for d in DRIVES for g in (131072.0, 262144.0) for s in SEEDS
                 if (s, d, g) not in done]
    else:
        tasks = [(s, d, g) for d in DRIVES for g in GAMMAS for s in SEEDS]
        path = OUT / "orders.jsonl"
    with get_context("spawn").Pool(4) as pool, path.open("a" if extra else "w") as fh:
        for i, rec in enumerate(pool.imap_unordered(case, tasks), 1):
            if extra:
                rec.pop("edges")
                rec.pop("binned")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{i}/{len(tasks)} {rec['drive']} {rec['gamma']:.0f} order {rec['mean_order']:.1f}", flush=True)


if __name__ == "__main__":
    import sys
    if "prefactor" in sys.argv[1:]:
        prefactor_main()
    else:
        main(extra="extra" in sys.argv[1:])
