"""Exact order moments of the seeded random-unitary reset reservoirs (Appendix A, Fig. 5).

Each family applies one of three Haar unitaries (fixed seeds) with probabilities (0.2, 0.3, 0.5) and resets
with probability p. For two reset states and two readouts the share of the memory carried by more than K
inputs is computed exactly and compared with the mixing law exp(-beta p K).

Run with:  python experiments/theory/order_law/reconstruct.py   (writes results/order_law/)
"""

import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[variable] = "1"
from pathlib import Path
import json
import numpy as np
from scipy.linalg import solve
from scipy.stats import unitary_group

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "results" / "order_law"
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = np.array([0.2, 0.3, 0.5])
PROBABILITIES = (0.03, 0.01, 0.003, 0.001)


def hermitian_basis(d):
    basis = []
    for i in range(d):
        for j in range(i + 1, d):
            z = np.zeros((d, d), complex)
            z[i, j] = z[j, i] = 2**-0.5
            basis.append(z)
            z = np.zeros((d, d), complex)
            z[i, j] = -1j * 2**-0.5
            z[j, i] = z[i, j].conjugate()
            basis.append(z)
    for j in range(1, d):
        z = np.zeros((d, d), complex)
        z[np.arange(j), np.arange(j)] = 1
        z[j, j] = -j
        basis.append(z / np.sqrt(j * (j + 1)))
    return np.array(basis)


def family(d, seed):
    basis = hermitian_basis(d)
    rng = np.random.default_rng(seed)
    n = d * d - 1
    unitaries = [unitary_group.rvs(d, random_state=rng) for _ in WEIGHTS]
    rotations = np.array(
        [
            np.einsum("aij,bji->ab", basis, U @ basis @ U.conj().T).real
            for U in unitaries
        ]
    )
    assert max(np.linalg.norm(O.T @ O - np.eye(n)) for O in rotations) < 1e-12
    A = np.einsum("u,uij->ij", WEIGHTS, rotations)
    T = sum(w * np.kron(O, O) for w, O in zip(WEIGHTS, rotations))
    beta = 2 * n / (n - np.linalg.norm(A) ** 2)
    pure_rng = np.random.default_rng([seed, 1])
    psi = pure_rng.normal(size=d) + 1j * pure_rng.normal(size=d)
    psi /= np.linalg.norm(psi)
    reset_vectors = [
        basis[:, 0, 0].real,
        np.einsum("i,aij,j->a", psi.conj(), basis, psi).real,
    ]
    scalar = np.random.default_rng([seed, 2]).normal(size=n)
    scalar /= np.linalg.norm(scalar)
    readouts = {"full": np.eye(n), "scalar": scalar[None, :]}
    eye = np.eye(n * n)
    B0 = np.kron(A, A)
    records = []
    arrays = {
        "unitaries": np.stack(unitaries),
        "A": A,
        "reset_vectors": np.stack(reset_vectors),
        "scalar_readout": scalar,
        "rotations": rotations,
    }
    assert max(abs(np.linalg.eigvals(A))) < 1
    assert (np.linalg.svd(eye - T, compute_uv=False) < 1e-10).sum() == 1
    for p in PROBABILITIES:
        r = 1 - p
        B = r * r * B0
        F = solve(eye - B, r * r * (T - B0))
        max_order = int(np.ceil(6 / (beta * p)))
        for reset, b in zip(("basis", "random"), reset_vectors):
            mean = solve(np.eye(n) - r * A, p * b)
            innovation = r * (rotations - A) @ mean
            V = np.einsum("u,ui,uj->ij", WEIGHTS, innovation, innovation)
            stationary = solve(eye - r * r * T, V.ravel())
            G1 = solve(eye - B, V.ravel())
            weighted = solve(eye - F, stationary)
            cov = stationary.reshape(n, n)
            res = np.linalg.norm(
                stationary - r * r * T @ stationary - V.ravel()
            ) / np.linalg.norm(stationary)
            reconstruction = np.linalg.norm(
                stationary - G1 - F @ stationary
            ) / np.linalg.norm(stationary)
            assert (
                res < 1e-10
                and reconstruction < 1e-10
                and np.linalg.eigvalsh(cov).min() > 0
            )
            mean0 = solve(np.eye(n) - A, b)
            v = float(
                np.einsum(
                    "u,ui,ui->",
                    WEIGHTS,
                    (rotations - A) @ mean0,
                    (rotations - A) @ mean0,
                )
            )
            assert v > 0
            phi_order = stationary.copy()
            survival = np.empty((2, max_order + 1))
            functionals = []
            means = []
            for M in readouts.values():
                q = M.shape[0]
                observed = M @ cov @ M.T
                functional = (M.T @ np.linalg.inv(observed) @ M / q).ravel()
                functionals.append(functional)
                means.append(float(functional @ weighted))
            functionals = np.array(functionals)
            for k in range(max_order + 1):
                survival[:, k] = functionals @ phi_order
                phi_order = F @ phi_order
            x = beta * p * np.arange(max_order + 1)
            reference = np.exp(-x)
            for j, key in enumerate(readouts):
                arr = survival[j]
                assert (
                    abs(arr[0] - 1) < 1e-9
                    and arr.min() > -1e-10
                    and np.diff(arr).max() < 1e-10
                )
                tag = f"{p}_{reset}_{key}"
                arrays[tag + "_x"] = x
                arrays[tag + "_survival"] = arr
                records.append(
                    dict(
                        d=d,
                        seed=seed,
                        reset=reset,
                        readout=key,
                        p=p,
                        beta=beta,
                        mean_order=means[j],
                        scaled_mean=beta * p * means[j],
                        max_survival_error=float(np.max(abs(arr - reference))),
                        relative_covariance_residual=float(res),
                        order_sum_residual=float(reconstruction),
                    )
                )
    np.savez_compressed(OUT / f"family_{seed}.npz", **arrays)
    return records


def main():
    records = []
    for d in (2, 3, 4):
        for seed in (11000 * d + 1, 11000 * d + 2):
            records.extend(family(d, seed))
    (OUT / "order_results.json").write_text(json.dumps(records, indent=2) + "\n")
    for p in PROBABILITIES:
        z = [r for r in records if r["p"] == p]
        print(
            p,
            len(z),
            "max error",
            max(r["max_survival_error"] for r in z),
            "scaled means",
            min(r["scaled_mean"] for r in z),
            max(r["scaled_mean"] for r in z),
        )


if __name__ == "__main__":
    main()
