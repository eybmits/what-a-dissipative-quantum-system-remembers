#!/usr/bin/env python3
"""Order of the memory of the five-spin reservoir from exact moments (paper_v2, Sec. 2.4, Methods, Appendix A).

For local loss and opposite phases on a grid of loss rates (powers of four) and input amplitudes (powers of two, and the
gentle-writing amplitude), this computes for the readout of the 45 features of the paper:

  orders      share of the capacity carried by products of k inputs, from the order recursion of Methods,
              G_1 = (1-B)^{-1} V, G_{k+1} = (1-B)^{-1} K G_k, continued by its geometric tail once the ratio of
              successive orders has settled
  mean_order  mean order of the capacity; single: share of single past inputs (delays 1, 2, ...)
  memory300   linear recall summed over delays 1 to 300, from exact moments (no finite test set); memory: all delays
  tau90       the delay below which 90% of that recall lies
  t_mem, s    (full amplitude and gentle writing) the mixing law: t_mem = 1/(1 - lambda) from the slowest
              covariance mode lambda of the full step T X = E[A X A^T] (right eigenvector Q, left eigenvector l, two
              steps of inverse iteration), and the mixing fraction s = l(K Q) / l(Q), so that the law predicts the
              mean order s t_mem (Appendix A, physical channels)

Inputs u are uniform on [0, 1]; expectations over u use Gauss-Legendre quadrature, and every step map is an exact
matrix exponential at a quadrature node. The input amplitude is scaled as in run_clock.py.

Run with:  python experiments/run_five_spin_orders.py [--seeds 4] [--workers 2] [--threads 5]   (resumes)
"""
from __future__ import annotations

import os
import sys

_THREADS = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "1"   # BLAS threads per worker
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = _THREADS

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "theory" / "finite_shots_vs_gamma"))

import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
import scipy.linalg as la  # noqa: E402

import psim  # noqa: E402
import run_clock as rc  # noqa: E402
import run_dark_projection as rdp  # noqa: E402
import run_final_scaling as rfs  # noqa: E402

OUT = HERE.parent / "results" / "five_spin_orders"
NODES = 6                                  # Gauss-Legendre nodes in u; 5 and 12 agree to six digits
FEAT = psim.FEAT - 1                       # the 45 features among the traceless coordinates
DELAYS = 300
VAR_U = 1.0 / 12.0


def step_maps(seed, name, gamma, eta, nodes=NODES):
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = rdp.hamiltonian(J, rc.drive(eta))
    G = H0 + gamma * rdp.dissipator(name)
    x, w = np.polynomial.legendre.leggauss(nodes)
    u, w = 0.5 * (x + 1), 0.5 * w
    E = [la.expm(psim.DT * (G + ui * H1)) for ui in u]
    assert all(np.allclose(e[0], np.eye(len(e))[0], atol=1e-12) for e in E)     # the trace is kept
    return u, w, [e[1:, 1:] for e in E], [e[1:, 0] for e in E]


class Stein:
    """X - A X A^T = C (and X - A^T X A = C) through the eigenvectors of A, checked against the residual."""

    def __init__(self, A):
        self.A = A
        lam, P = np.linalg.eig(A)
        self.P, self.Pi = P, np.linalg.inv(P)
        self.R = 1.0 / (1.0 - np.outer(lam, lam))

    def solve(self, C):
        X = (self.P @ (self.R * (self.Pi @ C @ self.Pi.T)) @ self.P.T).real
        return (X + X.T) / 2

    def solve_adjoint(self, C):
        X = (self.Pi.T @ (self.R * (self.P.T @ C @ self.P)) @ self.Pi).real
        return (X + X.T) / 2

    def residual(self, C):
        X = self.solve(C)
        return float(np.linalg.norm(X - self.A @ X @ self.A.T - C) / np.linalg.norm(C))


def series(first, step, tol=1e-8, maxit=50000):
    """Sum of Y_0 = first, Y_{k+1} = step(Y_k), with the geometric tail once successive terms shrink at a settled
    ratio; returns the sum, the ratio and the number of terms."""
    Y, total, norms = first, first.copy(), [np.linalg.norm(first)]
    for k in range(1, maxit):
        Y = step(Y)
        total += Y
        norms.append(np.linalg.norm(Y))
        r = norms[-1] / norms[-2]
        if r < 1 and norms[-1] * r / (1 - r) < tol * np.linalg.norm(total):
            return total, r, k
        if k > 40 and abs(r - norms[-11] / norms[-12]) < 1e-3 * (1 - r):
            return total + Y * r / (1 - r), r, k
    raise RuntimeError("series did not settle")


def case(task):
    seed, name, gamma, eta, law = task
    t0 = time.time()
    u, w, A, b = step_maps(seed, name, gamma, eta)
    n = len(b[0])
    Abar, bbar = sum(wq * Aq for wq, Aq in zip(w, A)), sum(wq * bq for wq, bq in zip(w, b))
    m = la.solve(np.eye(n) - Abar, bbar)
    xi = [Aq @ m + bq - m for Aq, bq in zip(A, b)]
    V = sum(wq * np.outer(x, x) for wq, x in zip(w, xi))
    D = [Aq - Abar for Aq in A]
    stein = Stein(Abar)
    res = stein.residual(V)
    K = lambda X: sum(wq * Dq @ X @ Dq.T for wq, Dq in zip(w, D))
    Kadj = lambda X: sum(wq * Dq.T @ X @ Dq for wq, Dq in zip(w, D))

    # order recursion on the feature block, with the geometric tail once successive orders settle
    f = FEAT
    G = stein.solve(V)
    past1 = Abar @ G @ Abar.T
    full = G.copy()
    blocks, ratios, tail = [], [], None
    for k in range(1, 50001):
        blocks.append(G[np.ix_(f, f)].copy())
        G = stein.solve(K(G))
        full += G
        ratios.append(float(np.trace(G[np.ix_(f, f)]) / np.trace(blocks[-1])))
        r_k = ratios[-1]
        if r_k < 1 and np.trace(blocks[-1]) * r_k / (1 - r_k) < 1e-8 * sum(np.trace(B) for B in blocks):
            break                              # what is left is negligible
        if k > 40 and abs(r_k - ratios[-11]) < 1e-3 * (1 - r_k):
            tail = r_k                         # successive orders settled: sum the geometric tail
            full += G * r_k / (1 - r_k)
            break
    t1 = time.time()
    F = np.array(blocks)
    Sigma = F.sum(axis=0)
    rho = tail if tail is not None else 0.0
    if tail is not None:                     # orders beyond the last one computed: F_last rho^j
        Sigma = Sigma + F[-1] * rho / (1 - rho)
    Sinv = np.linalg.pinv(Sigma, rcond=1e-12, hermitian=True)
    shares = np.einsum("ij,kji->k", Sinv, F)
    rank = float(np.trace(Sinv @ Sigma))
    Kn = len(shares)
    ks = np.arange(1, Kn + 1)
    tail_mass = shares[-1] * rho / (1 - rho) if tail is not None else 0.0
    tail_first = shares[-1] * (Kn * rho / (1 - rho) + rho / (1 - rho) ** 2) if tail is not None else 0.0
    total = shares.sum() + tail_mass
    mean_order = float((ks @ shares + tail_first) / total)
    single = float(np.trace(Sinv @ past1[np.ix_(f, f)]) / total)

    # linear recall from exact moments, until it has fallen to a negligible part
    g = sum(wq * (Aq @ m + bq) * (uq - 0.5) for wq, uq, Aq, bq in zip(w, u, A, b))
    curve, vec = [], g.copy()
    while len(curve) < 200000:
        vec = Abar @ vec
        c = vec[f]
        curve.append(float(c @ Sinv @ c / VAR_U))
        if len(curve) > 300 and curve[-1] < 1e-7 * sum(curve):
            break
    curve = np.array(curve)
    cum = np.cumsum(curve)
    out = {"seed": seed, "design": name, "gamma": gamma, "eta": eta, "nodes": NODES, "stein_residual": res,
           "orders_computed": Kn, "order_ratio": rho, "rank": rank, "mean_order": mean_order, "single": single,
           "memory300": float(curve[:DELAYS].sum()), "memory": float(cum[-1]),
           "tau90": int(np.searchsorted(cum, 0.9 * cum[-1]) + 1), "tau50": int(np.searchsorted(cum, 0.5 * cum[-1]) + 1),
           "mean_delay": float(np.arange(1, len(curve) + 1) @ curve / cum[-1]),
           "curve300": np.round(curve[:DELAYS], 6).tolist(), "tail_share": float(tail_mass / total)}
    edges = np.unique(np.r_[np.arange(1, 65), np.round(np.geomspace(65, Kn + 1, 60)).astype(int)])
    edges = edges[edges <= Kn + 1]
    out["edges"] = edges.tolist()
    out["binned"] = [float(shares[e0 - 1:e1 - 1].sum() / total) for e0, e1 in zip(edges[:-1], edges[1:])]

    # the mixing law: slowest covariance mode of the full step T X = E[A X A^T] by two steps of inverse iteration,
    # forward from the stationary covariance and backward from the identity, and the mixing fraction s = l(K Q)/l(Q)
    if law:
        fwd = lambda C: series(stein.solve(C), lambda Y: stein.solve(K(Y)))[0]
        bwd = lambda C: series(stein.solve_adjoint(C), lambda Y: stein.solve_adjoint(Kadj(Y)))[0]
        X2 = fwd(full)
        X3 = fwd(X2)
        L1 = bwd(np.eye(n))
        L2 = bwd(L1)
        L3 = bwd(L2)
        ell = lambda X: float(np.sum(L3 * X))
        mu = ell(X2) / ell(X3)                 # 1 - slowest covariance eigenvalue (Rayleigh quotient)
        sfrac = ell(K(X3)) / ell(X3)           # mixing fraction l(K Q) / l(Q)
        mu_prev = float(np.sum(L2 * full) / np.sum(L2 * X2))       # one step less: how settled the iteration is
        out.update(t_mem=1.0 / mu, s=sfrac, law_mean=sfrac / mu, mu_settled=mu / mu_prev)
    out["seconds"] = round(time.time() - t0, 1)
    out["seconds_orders"] = round(t1 - t0, 1)
    return out


# the grid: loss rates in powers of four and amplitudes in powers of two put the collapse variables gamma eta^2
# and gamma / eta^2 on one grid of powers of four for every amplitude; the gentle lines eta = gamma^(-1/2) (opposite
# phases) and gamma^(1/2) (local loss) are on it as well. The full amplitude reaches s t_mem >> 1, where the law is
# asymptotic: up to gamma = 16384 for opposite phases and down to 1/4096 for local loss.
AMPS = (1.0, 0.5, 0.25, 0.125)
GRID = {"signed": tuple(float(4.0 ** k) for k in range(0, 8)),             # 1 ... 16384
        "separate": tuple(float(4.0 ** -k) for k in range(-1, 7))}         # 4 ... 1/4096


EXTREME = {"signed": (4096.0, 16384.0), "separate": (1 / 1024, 1 / 4096)}     # slow: fewer reservoirs
WITH_REST = "--rest" in sys.argv


def jobs(seeds, collapse_seeds):
    """First the law: full amplitude and gentle writing on all seeds (on the collapse seeds at the extreme loss
    rates); then the other amplitudes, for the collapse of the recall range, on the collapse seeds."""
    law, rest = [], []
    for name in ("signed", "separate"):
        for g in GRID[name]:
            gentle = min(1.0, g ** (-0.5 if name == "signed" else 0.5))
            use = collapse_seeds if g in EXTREME[name] else seeds
            for a in sorted({1.0, gentle}, reverse=True):
                if a == 1.0 and g == EXTREME[name][1]:
                    continue                   # the full amplitude at the very longest memory: hours per reservoir
                law += [(s, name, g, a, True) for s in use]
            rest += [(s, name, g, a, False) for a in AMPS if a != 1.0 and abs(a - gentle) > 1e-12
                     for s in collapse_seeds]
    return law + rest if WITH_REST else law


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--collapse-seeds", type=int, default=2)
    parser.add_argument("--rest", action="store_true", help="also the other amplitudes along the loss rate")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--only", default=None, help="design to run, e.g. signed")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "orders.jsonl"
    seeds = rfs.deterministic_seeds(8)
    tasks = jobs(seeds[:args.seeds], seeds[:args.collapse_seeds])
    if args.only:
        tasks = [t for t in tasks if t[1] == args.only]
    key = lambda t: json.dumps([t[0], t[1], t[2], t[3]])
    done = set()
    if path.exists():
        done = {key((r["seed"], r["design"], r["gamma"], r["eta"])) for r in map(json.loads, path.open())}
    tasks = [t for t in tasks if key(t) not in done]
    print(f"{len(tasks)} tasks left", flush=True)
    with get_context("spawn").Pool(args.workers) as pool, path.open("a") as handle:
        for i, rec in enumerate(pool.imap_unordered(case, tasks), 1):
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            law = f" law {rec['law_mean']:.2f} t_mem {rec['t_mem']:.0f} s {rec['s']:.3f}" if "law_mean" in rec else ""
            print(f"{i}/{len(tasks)} {rec['design']} {rec['gamma']:g} eta {rec['eta']:g} order {rec['mean_order']:.2f}"
                  f"{law} memory {rec['memory300']:.1f} tau90 {rec['tau90']} ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
