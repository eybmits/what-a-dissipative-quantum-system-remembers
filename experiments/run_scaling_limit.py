#!/usr/bin/env python3
"""The scaling limit of a long memory, evaluated for the five-spin reservoir (paper_v2, Sec. 3.3 and
Appendices B.4 and E.3).

A reservoir step r' = A(u) r + b(u) on the traceless coordinates of the state that expands as

    A(u) = O0 [1 + eta a1(u) + eta^2 a2(u) - eps Y] + o(eps),      b(u) = eps O0 c0 + o(eps),

with O0 orthogonal and eta^2 = nu eps, has in the limit eps -> 0 at fixed nu the stationary covariance and the
covariances of the orders (Theorem 2 of the paper)

    Sigma = nu [L + nu (S - K)]^{-1} Pi V1,     G_k = F^{k-1} G_1,   G_1 = nu (L + nu S)^{-1} Pi V1,
    F = nu (L + nu S)^{-1} K,

on the symmetric matrices that commute with O0 (projector Pi), where L X = Pi (Y X + X Y^T),
S X = -Pi (a2bar X + X a2bar^T), K X = Pi E[a1 X a1^T], V1 = E[a1 m0 m0^T a1^T], and the mean m0 is fixed by O0 and
solves Pi0 (Y - nu a2bar) m0 = Pi0 c0. Everything is built from the generator of the reservoir,

    generator(delta, eps) = Hbar + delta X + eps D(delta),       delta = eta (u - 1/2),

on coordinates whose index 0 is the trace:

    separate   the five-spin generator with independent decay, eps = gamma (992 slow covariances)
    signed     the strong-loss generator of Eq. (A2) on the 100 protected operators of the opposite-phase channel,
               eps = 1/gamma (90 slow covariances). Of the 45 features, 33 directions respond at leading order and 11
               only through the admixture of decaying operators of order 1/gamma, r = (1 - S H(u)/gamma) R c, which the
               readout of the limit includes.

Stages:
  limit   the limit on a grid of nu for four reservoirs and both channels: shares of orders one and two in the
          capacity of the 45 features, mean order, mixing number rho(F)/(1 - rho(F)) and the variance of the features
          carried by single inputs (limit.jsonl)
  exact   the order recursion of Methods for the full five-spin master equation at nu = 1/16 ... 64: two reservoirs
          at the weakest leak (gamma = 1/4096 and 4096), the first also at a leak four times larger (exact.jsonl)
  dark    the recursion for the strong-loss generator against its limit, gamma = 2^6 ... 2^18 (dark.jsonl)

Run with:  python experiments/run_scaling_limit.py limit|exact|dark [--workers 3]     (resumes; minutes to an hour)
           python experiments/run_scaling_limit.py --analyse                          (prints the numbers of the paper)
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import json
import sys
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

OUT = HERE.parent / "results" / "scaling_limit"
DT = psim.DT
VAR_U = 1.0 / 12.0                               # variance of an input, uniform on [0, 1]
FEAT = np.asarray(psim.FEAT)
POINTS = (1 / 16, 1 / 4, 1.0, 4.0, 16.0, 64.0)   # nu of the exact runs
GRID = sorted(set(np.round(np.logspace(-2, 3, 31), 6).tolist()) | set(POINTS))
WEAKEST = {"separate": 1 / 4096, "signed": 4096.0}


def seeds():
    import run_final_scaling as rfs
    return rfs.deterministic_seeds(8)[:4]


# ------------------------------------------------------------------------------------------------ the models
def hamiltonian(seed):
    """Generator at the mean input and unit input generator: H(u) = Gs + eta (u - 1/2) X1."""
    J, _ = psim.couplings_and_inputs(seed, 10)
    b = psim.blocks()
    Gs = psim.H_FIELD * b["gz"] + 1.5 * psim.H_FIELD * b["gx"]
    for key, value in J.items():
        Gs = Gs + value * b["xx"][key]
    return Gs, psim.H_FIELD * b["gx"]


class Model:
    """generator(delta, eps) on coordinates with the trace at index 0 (value r0); readout M on the traceless ones."""

    def __init__(self, Hbar, X, dis, r0, M):
        self.Hbar, self.X, self.dis, self.r0, self.M = Hbar, X, dis, r0, M
        self.n = len(Hbar) - 1

    def maps(self, eps, eta, nodes=6):
        """Step maps at Gauss-Legendre nodes of u in [0, 1]: weights, A on the traceless coordinates and b."""
        x, w = np.polynomial.legendre.leggauss(nodes)
        E = [la.expm(DT * (self.Hbar + eta * (ui - 0.5) * self.X + eps * self.dis(eta * (ui - 0.5))))
             for ui in 0.5 * (x + 1)]
        assert all(abs(e[0, 1:]).max() < 1e-9 and abs(e[0, 0] - 1) < 1e-9 for e in E), "the trace is not kept"
        return 0.5 * w, [e[1:, 1:] for e in E], [e[1:, 0] * self.r0 for e in E]


def full_model(seed, name):
    """The five-spin generator Gs + delta X1 + eps D on the 1024 Pauli coordinates."""
    Gs, X1 = hamiltonian(seed)
    D = psim.blocks()[name]
    M = np.zeros((len(FEAT), psim.DIM - 1))
    M[np.arange(len(FEAT)), FEAT - 1] = 1.0
    return Model(Gs, X1, lambda delta: D, 1.0, M)


def dark_model(seed, name="signed"):
    """Eq. (A2) for a collective channel: G(u) = L^T H(u) R - eps L^T H(u) S H(u) R on the kernel of its unit-rate
    dissipator, with the trace direction of the kernel as coordinate 0."""
    Gs, X1 = hamiltonian(seed)
    D = psim.blocks()[name]
    U, s, Vt = np.linalg.svd(D)
    R, Lk = Vt[s < 1e-9].T, U[:, s < 1e-9]
    Lk = Lk @ np.linalg.inv(R.T @ Lk)
    S = np.linalg.inv(D + R @ Lk.T) - R @ Lk.T
    SG, SX = S @ Gs @ R, S @ X1 @ R
    t = R[0] / np.linalg.norm(R[0])
    v = t - np.eye(len(t))[0]
    Q = np.eye(len(t)) - 2.0 * np.outer(v, v) / (v @ v)        # reflection that maps e_0 onto the trace direction
    F0, F1, N0, N1, N2 = (Q.T @ Z @ Q for Z in (Lk.T @ Gs @ R, Lk.T @ X1 @ R, -Lk.T @ Gs @ SG,
                                                 -Lk.T @ (Gs @ SX + X1 @ SG), -Lk.T @ X1 @ SX))
    assert abs(F0[1:, 1:] + F0[1:, 1:].T).max() < 1e-10 and max(abs(Z[0]).max() for Z in (F0, F1, N0)) < 1e-10
    model = Model(F0, F1, lambda delta: N0 + delta * N1 + delta * delta * N2, 1.0 / np.linalg.norm(R[0]),
                  (R[FEAT] @ Q)[:, 1:])
    model.M1 = (-SG[FEAT] @ Q)[:, 1:]                           # features of the admixed decaying operators
    return model


def effective_readout(M0, M1, tol=1e-9):
    """The readout that the features M0 + eps M1 are equivalent to as eps -> 0: the directions that M0 reaches and,
    for the others, their part of first order (an invertible map of the features does not change a capacity).
    Returns the readout and the number of directions that M0 reaches."""
    U, sv, _ = np.linalg.svd(M0)
    r = int(np.sum(sv > tol * sv[0]))
    return np.vstack([U[:, :r].T @ M0, U[:, r:].T @ M1]), r


# ------------------------------------------------------------------------------- the order recursion of Methods
class Stein:
    """X - A X A^T = C through the eigenvectors of A."""

    def __init__(self, A):
        lam, P = np.linalg.eig(A)
        self.P, self.Pi = P, np.linalg.inv(P)
        self.R = 1.0 / (1.0 - np.outer(lam, lam))

    def solve(self, C):
        X = (self.P @ (self.R * (self.Pi @ C @ self.Pi.T)) @ self.P.T).real
        return (X + X.T) / 2


def shares_of(blocks, Sigma_block, tail=0.0):
    """Shares of the orders in the capacity of a readout, Tr[(M Sigma M^T)^+ M G_k M^T] / rank, their mean order
    and the rank; the last order continues geometrically with the ratio `tail`."""
    Sinv = np.linalg.pinv(Sigma_block, rcond=1e-12, hermitian=True)
    sh = np.array([float(np.sum(Sinv * B)) for B in blocks])
    k = np.arange(1, len(sh) + 1)
    total = sh.sum() + sh[-1] * tail / (1 - tail)
    first = k @ sh + sh[-1] * (len(sh) * tail / (1 - tail) + tail / (1 - tail) ** 2)
    return sh / total, float(first / total), float(np.sum(Sinv * Sigma_block))


def exact(model, eps, eta, tol=1e-10, maxit=20000):
    """Stationary mean and covariance and the covariances of the orders, Eq. (A3) of the paper."""
    w, A, b = model.maps(eps, eta)
    n, M = model.n, model.M
    Abar, bbar = sum(wq * Aq for wq, Aq in zip(w, A)), sum(wq * bq for wq, bq in zip(w, b))
    m = la.solve(np.eye(n) - Abar, bbar)
    xi = [Aq @ m + bq - m for Aq, bq in zip(A, b)]
    V = sum(wq * np.outer(x, x) for wq, x in zip(w, xi))
    Dq = [Aq - Abar for Aq in A]
    stein = Stein(Abar)
    G = stein.solve(V)
    G1, Sigma, blocks, traces, tail = G.copy(), G.copy(), [M @ G @ M.T], [np.trace(G)], 0.0
    for k in range(2, maxit):
        G = stein.solve(sum(wq * D @ G @ D.T for wq, D in zip(w, Dq)))
        Sigma += G
        blocks.append(M @ G @ M.T)
        traces.append(np.trace(G))
        r = traces[-1] / traces[-2]
        if r < 1 and traces[-1] * r / (1 - r) < tol * sum(traces):
            break
        if k > 60 and abs(r - traces[-11] / traces[-12]) < 1e-4 * (1 - r):
            tail = r
            Sigma += G * r / (1 - r)
            break
    sh, mean_order, rank = shares_of(blocks, M @ Sigma @ M.T, tail)
    return {"m": m, "G1": G1, "Sigma": Sigma, "shares": sh, "mean_order": mean_order, "rank": rank,
            "orders": len(blocks)}


# ---------------------------------------------------------------------------------------------------- the limit
class Limit:
    """The operators of the limit in a real basis of the rotation O0: the vectors that O0 leaves fixed, then one
    plane for each pair of conjugate phases. A symmetric matrix that commutes with O0 is a symmetric block on the
    fixed vectors and a multiple of the identity on each plane, which gives d (d - 1) unknowns for d energy levels
    without degenerate transition frequencies."""

    def __init__(self, model, tol=1e-8):
        n1 = model.n + 1
        Z = np.zeros((n1, n1))
        E3 = la.expm(DT * np.block([[model.Hbar, model.X, Z], [Z, model.Hbar, model.X], [Z, Z, model.Hbar]]))
        O0 = E3[:n1, :n1]
        a1, a2 = O0.T @ E3[:n1, n1:2 * n1], O0.T @ E3[:n1, 2 * n1:]       # per unit delta and delta^2
        E2 = la.expm(DT * np.block([[model.Hbar, model.dis(0.0)], [Z, model.Hbar]]))
        W = O0.T @ E2[:n1, n1:]                                            # exp(dt (Hbar + eps D)) = O0 (1 + eps W)
        assert abs(O0 @ O0.T - np.eye(n1)).max() < 1e-9 and abs(a1 + a1.T).max() < 1e-9
        assert abs(a2 + a2.T - a1 @ a1).max() < 1e-9                       # the step is orthogonal at eps = 0
        a1, a2, Y, c0 = a1[1:, 1:], a2[1:, 1:], -W[1:, 1:], W[1:, 0] * model.r0
        w, Uc = np.linalg.eigh(1j * model.Hbar[1:, 1:])
        ang = (w * DT + np.pi) % (2 * np.pi) - np.pi                       # rotation angles of O0
        fixed = np.abs(ang) < tol
        ev, Bf = np.linalg.eigh((Uc[:, fixed] @ Uc[:, fixed].conj().T).real)
        cols, angles = [Bf[:, ev > 0.5]], []
        for j in np.nonzero((w > 0) & ~fixed)[0]:
            cols.append(np.sqrt(2) * np.c_[Uc[:, j].real, Uc[:, j].imag])
            angles.append(abs(ang[j]))
        B = np.hstack(cols)
        self.nf, self.npl, self.n = cols[0].shape[1], len(angles), model.n
        assert self.nf + 2 * self.npl == model.n and abs(B.T @ B - np.eye(model.n)).max() < 1e-9
        self.gap = float(np.diff(np.sort(angles)).min())                   # smallest gap between two angles
        self.a, self.a2bar, self.Y = B.T @ a1 @ B, VAR_U * (B.T @ a2 @ B), B.T @ Y @ B
        self.B, self.c0, self.Mt, self.Mfeat = B, B.T @ c0, model.M @ B, model.M @ B
        self.dim = self.nf * (self.nf + 1) // 2 + self.npl
        self.slope = float(VAR_U * np.sum(self.a ** 2) / (2 * np.trace(self.Y)))
        self._operators()

    # coordinates of a commuting matrix: the upper triangle of the fixed block, then one number for each plane
    def pack(self, S, p):
        return np.r_[S[np.triu_indices(self.nf)], p]

    def unpack(self, x):
        nt = self.nf * (self.nf + 1) // 2
        S = np.zeros((self.nf, self.nf))
        S[np.triu_indices(self.nf)] = x[:nt]
        return S + S.T - np.diag(np.diag(S)), x[nt:]

    def project(self, Xe):
        """Pi X for a matrix in the basis of the rotation."""
        d = np.diag(Xe)[self.nf:]
        return self.pack((Xe[:self.nf, :self.nf] + Xe[:self.nf, :self.nf].T) / 2, 0.5 * (d[0::2] + d[1::2]))

    def _operators(self):
        nf, dim = self.nf, self.dim
        iu = np.triu_indices(nf)
        nt = len(iu[0])

        def unit(i, j):
            E = np.zeros((nf, nf))
            E[i, j] = E[j, i] = 1.0
            return E

        def lyapunov(Zm):
            """X -> Pi (Z X + X Z^T)."""
            Op = np.zeros((dim, dim))
            for j, (i1, i2) in enumerate(zip(*iu)):
                Op[:nt, j] = (Zm[:nf, :nf] @ unit(i1, i2) + unit(i1, i2) @ Zm[:nf, :nf].T)[iu]
            d = np.diag(Zm)[nf:]
            Op[nt:, nt:] = np.diag(d[0::2] + d[1::2])
            return Op

        self.L, self.S = lyapunov(self.Y), -lyapunov(self.a2bar)
        a = self.a
        aff, afp, apf = a[:nf, :nf], a[:nf, nf:], a[nf:, :nf]
        K = np.zeros((dim, dim))
        for j, (i1, i2) in enumerate(zip(*iu)):
            K[:nt, j] = (aff @ unit(i1, i2) @ aff.T)[iu]
            col = apf[:, i1] * apf[:, i2] * (2.0 if i1 != i2 else 1.0)
            K[nt:, j] = 0.5 * (col[0::2] + col[1::2])
        for k in range(self.npl):
            K[:nt, nt + k] = (afp[:, 2 * k:2 * k + 2] @ afp[:, 2 * k:2 * k + 2].T)[iu]
        app = a[nf:, nf:] ** 2
        K[nt:, nt:] = 0.5 * (app[0::2, 0::2] + app[0::2, 1::2] + app[1::2, 0::2] + app[1::2, 1::2])
        self.K = VAR_U * K

    def with_readout(self, M_in_basis):
        other = copy.copy(self)
        other.Mt = M_in_basis
        return other

    def readout(self, x, M=None):
        """M X M^T for a commuting matrix X."""
        S, p = self.unpack(x)
        M = self.Mt if M is None else M
        Mf, Mp = M[:, :self.nf], M[:, self.nf:]
        return Mf @ S @ Mf.T + (Mp * np.repeat(p, 2)) @ Mp.T

    def embed(self, x):
        S, p = self.unpack(x)
        Xe = np.zeros((self.n, self.n))
        Xe[:self.nf, :self.nf] = S
        idx = np.arange(self.nf, self.n)
        Xe[idx, idx] = np.repeat(p, 2)
        return Xe

    def solve(self, nu, orders=2000, tol=1e-12):
        nf = self.nf
        mf = la.solve(self.Y[:nf, :nf] - nu * self.a2bar[:nf, :nf], self.c0[:nf])
        v = self.a[:, :nf] @ mf
        rhs = VAR_U * self.project(np.outer(v, v))
        Lnu = self.L + nu * self.S
        Sig = nu * la.solve(Lnu - nu * self.K, rhs)
        G = nu * la.solve(Lnu, rhs)
        F = nu * la.solve(Lnu, self.K)
        rho = float(abs(np.linalg.eigvals(F)).max())
        Gs = [G]
        for _ in range(orders):
            G = F @ G
            Gs.append(G)
            if abs(G).max() < tol * abs(Sig).max():
                break
        sh, mean_order, rank = shares_of([self.readout(g) for g in Gs], self.readout(Sig))
        return {"mean": mf, "Sigma": Sig, "G1": Gs[0], "shares": sh, "mean_order": mean_order, "rank": rank,
                "rho": rho, "decay": float(np.linalg.eigvals(Lnu - nu * self.K).real.min()),
                "signal": float(np.trace(self.readout(Gs[0], self.Mfeat))),
                "variance": float(np.trace(self.readout(Sig, self.Mfeat)))}


# --------------------------------------------------------------------------------------------------- the stages
def limit_case(task):
    seed, name = task
    t0 = time.time()
    if name == "separate":
        model = full_model(seed, name)
        lim, leading, reach = Limit(model), None, len(FEAT)
    else:
        model = dark_model(seed, name)
        leading = Limit(model)
        Meff, reach = effective_readout(model.M, model.M1)
        lim = leading.with_readout(Meff @ leading.B)
    rows = [lim.solve(nu) for nu in GRID]
    out = {"seed": seed, "design": name, "fixed": lim.nf, "planes": lim.npl, "unknowns": lim.dim, "reach": reach,
           "gap": lim.gap, "slope": lim.slope, "nu": GRID, "rank": rows[GRID.index(1.0)]["rank"],
           "decay": [r["decay"] for r in rows], "share1": [float(r["shares"][0]) for r in rows],
           "share2": [float(r["shares"][1]) for r in rows], "mean_order": [r["mean_order"] for r in rows],
           "mixing": [r["rho"] / (1 - r["rho"]) for r in rows], "signal": [r["signal"] for r in rows],
           "variance": [r["variance"] for r in rows]}
    if leading is not None:                      # the features without the admixed decaying operators
        out["share1_leading"] = [float(leading.solve(nu)["shares"][0]) for nu in POINTS]
    out["seconds"] = round(time.time() - t0, 1)
    return out


def exact_case(task):
    seed, name, loss, nu = task
    t0 = time.time()
    model = full_model(seed, name)
    eta = float(np.sqrt(nu / loss) if name == "signed" else np.sqrt(nu * loss))
    e = exact(model, loss, eta)
    M = model.M
    return {"seed": seed, "design": name, "gamma": loss, "eta": eta, "nu": nu, "share1": float(e["shares"][0]),
            "share2": float(e["shares"][1]), "mean_order": e["mean_order"], "rank": e["rank"],
            "signal": float(np.trace(M @ e["G1"] @ M.T)), "variance": float(np.trace(M @ e["Sigma"] @ M.T)),
            "orders": e["orders"], "seconds": round(time.time() - t0, 1)}


def dark_case(task):
    seed, nu, gamma = task
    model = dark_model(seed)
    lim = Limit(model)
    q = lim.solve(nu)
    e = exact(model, 1 / gamma, float(np.sqrt(nu / gamma)))
    rel = lambda X, Y: float(np.linalg.norm(X - Y) / np.linalg.norm(Y))
    B = lim.B
    return {"seed": seed, "nu": nu, "gamma": gamma, "share1": float(e["shares"][0]), "limit_share1": float(q["shares"][0]),
            "mean_order": e["mean_order"], "limit_mean_order": q["mean_order"],
            "error_mean": rel(e["m"], B[:, :lim.nf] @ q["mean"]),
            "error_covariance": rel(e["Sigma"], B @ lim.embed(q["Sigma"]) @ B.T),
            "error_first_order": rel(e["G1"], B @ lim.embed(q["G1"]) @ B.T)}


STAGES = {
    "limit": (limit_case, lambda: [(s, n) for n in ("signed", "separate") for s in seeds()],
              lambda r: (r["seed"], r["design"])),
    "exact": (exact_case, lambda: sorted([(s, n, WEAKEST[n], nu) for n in WEAKEST for nu in POINTS for s in seeds()[:2]]
                                         + [(seeds()[0], n, WEAKEST[n] * (4 if n == "separate" else 0.25), nu)
                                            for n in WEAKEST for nu in POINTS], key=lambda t: t[3]),
              lambda r: (r["seed"], r["design"], r["gamma"], r["nu"])),
    "dark": (dark_case, lambda: [(seeds()[0], nu, 2.0 ** k) for nu in (0.25, 1.0, 4.0, 16.0) for k in range(6, 19, 2)],
             lambda r: (r["seed"], r["nu"], r["gamma"])),
}


def run(stage, workers):
    case, tasks, key = STAGES[stage]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stage}.jsonl"
    done = {key(r) for r in map(json.loads, path.open())} if path.exists() else set()
    todo = [t for t in tasks() if tuple(t) not in done]
    print(f"{len(todo)} cases to run", flush=True)
    with get_context("spawn").Pool(workers) as pool, path.open("a") as handle:
        for i, rec in enumerate(pool.imap_unordered(case, todo), 1):
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            print(f"{i}/{len(todo)} " + " ".join(f"{k} {rec[k]:g}" if isinstance(rec[k], float) else f"{k} {rec[k]}"
                                               for k in list(rec)[:5]), flush=True)


# ------------------------------------------------------------------------------------------------- the numbers
def load(stage):
    return [json.loads(line) for line in (OUT / f"{stage}.jsonl").open()]


def analyse():
    lim = {(r["seed"], r["design"]): r for r in load("limit")}
    ex = load("exact")
    orders = [json.loads(line) for line in (HERE.parent / "results" / "five_spin_orders" / "orders.jsonl").open()]
    at = lambda r, field, nu: float(np.interp(np.log(nu), np.log(r["nu"]), r[field]))
    for name in ("separate", "signed"):
        print(f"\n{name}")
        rows = [lim[(s, name)] for s in seeds()]
        print(f"  slow covariances {rows[0]['unknowns']} ({rows[0]['fixed']} fixed vectors, {rows[0]['planes']} planes); "
              f"feature directions at leading order {rows[0]['reach']}, rank {rows[0]['rank']:.0f}; "
              f"decay rate of the slowest covariance at nu = 1: {min(at(r, 'decay', 1.0) for r in rows):.3f} or more")
        nu_of = lambda r: r["gamma"] * r["eta"] ** 2 if name == "signed" else r["eta"] ** 2 / r["gamma"]
        print("  nu = 1, four reservoirs: limit of the order-one share, and the recursion at its weakest leak")
        for r in rows:
            ref = [o for o in orders if o["seed"] == r["seed"] and o["design"] == name and o["eta"] < 1
                   and abs(nu_of(o) - 1) < 1e-9]
            best = max(ref, key=lambda o: o["gamma"] if name == "signed" else -o["gamma"])
            print(f"    seed {r['seed']:>10}: limit {at(r, 'share1', 1.0):.4f} (mean order {at(r, 'mean_order', 1.0):.4f}, "
                  f"linear capacity {at(r, 'share1', 1.0) * r['rank']:.2f}) | recursion {best['binned'][0]:.4f} "
                  f"(mean order {best['mean_order']:.4f}) at gamma = {best['gamma']:g}"
                  + (f" | without the admixture {r['share1_leading'][POINTS.index(1.0)]:.4f}" if "share1_leading" in r else ""))
        print(f"    mean of the four limits: share {np.mean([at(r, 'share1', 1.0) for r in rows]):.3f}, linear capacity "
              f"{np.mean([at(r, 'share1', 1.0) * r['rank'] for r in rows]):.1f}")
        for seed in seeds()[:2]:
            r = lim[(seed, name)]
            pts = sorted((e for e in ex if e["seed"] == seed and e["design"] == name
                          and abs(e["gamma"] / WEAKEST[name] - 1) < 1e-9), key=lambda e: e["nu"])
            if not pts:
                continue
            d1 = [e["share1"] - at(r, "share1", e["nu"]) for e in pts]
            dm = [e["mean_order"] - at(r, "mean_order", e["nu"]) for e in pts]
            ds = [e["signal"] / at(r, "signal", e["nu"]) - 1 for e in pts]
            print(f"  seed {seed}, nu = 1/16 ... 64 at gamma = {WEAKEST[name]:g}: largest difference of the order-one share "
                  f"{max(map(abs, d1)):.4f}, of the mean order {max(map(abs, dm)):.4f}, of the signal {100 * max(map(abs, ds)):.1f}%")
            for e in pts:
                print(f"      nu {e['nu']:<7g} share {e['share1']:.4f} (limit {at(r, 'share1', e['nu']):.4f})  mean order "
                      f"{e['mean_order']:.4f} ({at(r, 'mean_order', e['nu']):.4f})  signal {e['signal']:.4e} ({at(r, 'signal', e['nu']):.4e})")
        for r in rows:
            k = int(np.argmax(r["signal"]))
            print(f"  seed {r['seed']:>10}: single-input signal largest at nu = {r['nu'][k]:.3g}, where the order-one share is "
                  f"{r['share1'][k]:.2f} and the mixing number {r['mixing'][k]:.2f}; share at nu = 0.01: {r['share1'][0]:.4f}; "
                  f"mixing number / nu at nu = 1000: {r['mixing'][-1] / r['nu'][-1]:.5f} (slope {r['slope']:.5f})")
    print("\nstrong-loss generator against its limit (first reservoir)")
    for r in sorted(load("dark"), key=lambda r: (r["nu"], r["gamma"])):
        print(f"  nu {r['nu']:<5g} gamma 2^{int(np.log2(r['gamma'])):<2d} share {r['share1']:.4f} (limit {r['limit_share1']:.4f})  "
              f"errors: mean {r['error_mean']:.1e}, covariance {r['error_covariance']:.1e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=sorted(STAGES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args()
    analyse() if args.analyse else run(args.stage, args.workers)
