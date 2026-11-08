#!/usr/bin/env python3
"""Runs added for the reviews of the November 2026 manuscript (five spins, exact features).

extend      memory and slow-mode rates of the four families with dark states at gamma = 128 and 256.
scaled      mixed phases with the input amplitude scaled as min(1, (gamma_c / gamma)^(1/2)),
            gamma_c = 1 (main curve) and 4 (check).
spectrum    slow-mode rates of all eight families up to gamma = 16384 (no memory run).
dephasing   one shared channel plus local dephasing sqrt(g_phi) sigma^z_i: slow-mode rates.
variance    feature variance of the shared and mixed-phase channels at fixed amplitude.

Every memory run also records the variance of the 45 features over the test inputs.
Run with:  python experiments/run_review_runs.py STAGE
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "theory" / "finite_shots_vs_gamma"))

import numpy as np

import psim
import run_dark_projection as rdp
import run_final_scaling as rfs
import run_jump_families as rjf

OUT = HERE.parent / "results" / "review_runs"
FAMILIES = ("separate", "exchange", "gainloss", "dephasing", "pair", "signed", "groups", "shared")
KERNEL = {"separate": 1, "exchange": 1, "gainloss": 1, "dephasing": 32, "pair": 36, "groups": 36,
          "signed": 100, "shared": 100}


def amplitude(gamma, gamma_c):
    return 1.0 if gamma_c is None else float(min(1.0, np.sqrt(gamma_c / gamma)))


def slow_rates(p):
    lam = np.sort(np.abs(np.linalg.eigvals(rdp.mean_step(p))))[::-1]
    return (-np.log(np.maximum(lam[1:100], 1e-300)) / psim.DT).round(7).tolist()


def memory_case(task):
    """Memory, slow-mode rates and feature variance; gamma_c scales the input amplitude."""
    seed, name, gamma, gamma_c = task
    eta = amplitude(gamma, gamma_c)
    wash = rdp.washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + rdp.TRAIN + rdp.VALIDATION + rdp.TEST)
    H0, H1 = rdp.hamiltonian(J, eta * np.ones(psim.N))
    rows, p = rdp.drive_steady(H0 + gamma * rjf.dissipator(name), H1, inputs, wash)
    var = rows[-rdp.TEST:].var(axis=0)
    return {"seed": seed, "design": name, "gamma": gamma, "gamma_c": gamma_c, "eta": eta,
            "memory": rdp.memory(rows, inputs[wash:]), "rates": slow_rates(p),
            "feature_var_mean": float(var.mean()), "feature_var_median": float(np.median(var)),
            "feature_var_max": float(var.max())}


def spectrum_case(task):
    seed, name, gamma = task
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = rdp.hamiltonian(J)
    p = psim.Propagator(H0 + gamma * rjf.dissipator(name), H1, nodes=rdp.NODES)
    return {"seed": seed, "design": name, "gamma": gamma, "rates": slow_rates(p)}


def dephasing_case(task):
    seed, gamma, g_phi = task
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = rdp.hamiltonian(J)
    D = g_phi * sum(psim.dissipator_gen([(0, psim.bit(i), 1.0)]) for i in range(psim.N))
    p = psim.Propagator(H0 + gamma * psim.blocks()["shared"] + D, H1, nodes=rdp.NODES)
    return {"seed": seed, "gamma": gamma, "dephasing": g_phi, "rates": slow_rates(p)}


def variance_case(task):
    seed, name, gamma = task
    wash = rdp.washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + rdp.TEST)
    H0, H1 = rdp.hamiltonian(J)
    rows, _ = rdp.drive_steady(H0 + gamma * rjf.dissipator(name), H1, inputs, wash)
    var = rows.var(axis=0)
    return {"seed": seed, "design": name, "gamma": gamma, "feature_var_mean": float(var.mean()),
            "feature_var_median": float(np.median(var)), "feature_var_max": float(var.max())}


def first_order_rates():
    """Mean decay rate that a unit extra dissipator adds to the 99 slow dark modes, to first order."""
    R, L, _ = rdp.kernel()
    out = {}
    extra = {"local loss": psim.blocks()["separate"],
             "local dephasing": sum(psim.dissipator_gen([(0, psim.bit(i), 1.0)]) for i in range(psim.N))}
    for key, D in extra.items():
        block = L.T @ D @ R                     # 100 x 100 on the dark operators
        lam = np.sort(-np.linalg.eigvals(block).real)
        out[key] = {"mean_rate_99": float(lam[1:].mean()), "trace_99": float(-np.trace(block).real)}
    return out


STAGES = {"extend": memory_case, "scaled": memory_case, "spectrum": spectrum_case,
          "dephasing": dephasing_case, "variance": variance_case}


def jobs(stage, seeds):
    if stage == "extend":
        # families without protected operators are at their memory floor from gamma = 64 on
        return [(s, n, g, None) for g in (128.0, 256.0) for n in ("shared", "groups", "signed", "pair")
                for s in seeds]
    if stage == "scaled":
        grid = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
        return ([(s, "signed", g, 1.0) for g in grid for s in seeds]
                + [(s, "signed", g, 4.0) for g in grid if g > 4 for s in seeds])
    if stage == "spectrum":
        return [(s, n, g) for g in (512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0) for n in FAMILIES
                for s in seeds]
    if stage == "dephasing":
        # the range over which the local-loss coefficient b was checked
        return [(s, g, d) for d in (0.001, 0.01, 0.1) for g in (32.0, 64.0, 128.0, 256.0) for s in seeds]
    return [(s, n, g) for n in ("shared", "signed") for g in (1.0, 4.0, 16.0, 64.0) for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES) + ("first_order",))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "first_order":
        record = first_order_rates()
        (OUT / "first_order.json").write_text(json.dumps(record, indent=1) + "\n")
        print(record)
        return
    tasks = jobs(args.stage, rfs.deterministic_seeds(args.seeds))
    path = OUT / f"{args.stage}.jsonl"
    done = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            done.add(tuple(r[k] for k in ("seed", "design", "gamma", "gamma_c", "dephasing") if k in r))
    key = {"extend": lambda t: t, "scaled": lambda t: t, "spectrum": lambda t: t,
           "dephasing": lambda t: (t[0], t[1], t[2]), "variance": lambda t: t}[args.stage]
    tasks = [t for t in tasks if key(t) not in done]
    with get_context("spawn").Pool(args.workers) as pool, path.open("a") as handle:
        for i, record in enumerate(pool.imap_unordered(STAGES[args.stage], tasks), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"{i}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
