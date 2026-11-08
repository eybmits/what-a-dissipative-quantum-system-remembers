#!/usr/bin/env python3
"""Leak coefficient of one shared channel at N = 3 to 6 spins.

At strong loss the slow modes of the input-averaged step decay at the mean rate a / gamma. To first order in
1/gamma the uniform input does not turn the dark states (it has no matrix element between them), so the mean
slow rate is the trace of the averaged second-order generator R^flat E_u[H S H] R over the n - 1 slow modes,
    a = Re Tr(R^flat E_u[H(u) S H(u)] R) / (n - 1),
with H = -i[H(u), .], S the inverse of the unit-rate dissipator on the decaying operators and n the number of
protected operators. The Hamiltonian, the couplings (same seeds) and the conventions are those of the five-spin
study, so N = 5 must reproduce a = 2.945.

Run with:  python experiments/run_leak_size.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_final_scaling as rfs  # noqa: E402

OUT = HERE.parent / "results" / "review_runs" / "leak_size.json"
H_FIELD = 0.5
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
LOWER = np.array([[0, 1], [0, 0]], dtype=complex)        # sigma^- = |0><1|, as in psim


def site(op, i, n):
    mats = [np.eye(2, dtype=complex)] * n
    mats[i] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def couplings(seed, n):
    rng = np.random.default_rng(seed)
    return {(i, j): rng.uniform(-1.0, 1.0) for i in range(n) for j in range(i + 1, n)}


def commutator(H):
    eye = np.eye(H.shape[0])
    return -1j * (np.kron(eye, H) - np.kron(H.T, eye))     # column-stacking vec


def dissipator(L):
    eye = np.eye(L.shape[0])
    LL = L.conj().T @ L
    return np.kron(L.conj(), L) - 0.5 * np.kron(eye, LL) - 0.5 * np.kron(LL.T, eye)


def leak(n, seeds):
    d = 2 ** n
    xs = [site(X, i, n) for i in range(n)]
    sum_x = sum(xs)
    sum_z = sum(site(Z, i, n) for i in range(n))
    L = sum(site(LOWER, i, n) for i in range(n))
    D = dissipator(L)
    dark = sla.null_space(L)                                  # dark states, L|d> = 0
    m = dark.shape[1]
    R = np.column_stack([np.kron(dark[:, b].conj(), dark[:, a]) for b in range(m) for a in range(m)])
    left = sla.null_space(D.conj().T)                         # conserved observables
    Rflat = np.linalg.solve(left.conj().T @ R, left.conj().T)
    P = R @ Rflat
    lu = sla.lu_factor(D + P)
    H1 = commutator(H_FIELD * sum_x)

    def second(Hs):
        Y = Hs @ R
        SY = sla.lu_solve(lu, Y) - P @ Y
        return Rflat @ (Hs @ SY)

    K1 = second(H1)
    out = []
    for seed in seeds:
        J = couplings(seed, n)
        H = sum(c * xs[i] @ xs[j] for (i, j), c in J.items()) + H_FIELD * sum_z + 1.5 * H_FIELD * sum_x
        K = second(commutator(H)) + K1 / 12.0                 # E_u over u uniform on [0, 1]
        out.append(float(np.trace(K).real / (m * m - 1)))
    return m * m, out


def main() -> None:
    seeds = rfs.deterministic_seeds(8)
    record = {}
    for n in (3, 4, 5, 6):
        protected, values = leak(n, seeds)
        record[str(n)] = {"protected": protected, "a": values, "a_mean": float(np.mean(values)),
                          "a_std": float(np.std(values))}
        print(n, protected, round(float(np.mean(values)), 4), round(float(np.std(values)), 4), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
