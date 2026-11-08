#!/usr/bin/env python3
"""Recall against the loss rate for the remaining jump families.

Adds the exchange-assisted, dephasing and gain/loss families of the first
version to ``results/dark_projection/memory.jsonl`` (separate, shared, signed,
groups, pair). Same reservoirs, inputs, readout and budget Tr K = 80 gamma at
N = 5; exact features, no shot noise.

memory      recall to delay 300 and capacity by degree, as in run_dark_projection.
kernel      dimension of the kernel of each unit-rate dissipator.

Run with:  python experiments/run_jump_families.py STAGE
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

OUT = HERE.parent / "results" / "jump_families"
FAMILIES = ("exchange", "dephasing", "gainloss")
ALL = ("separate", "shared", "signed", "groups", "pair") + FAMILIES
_ORIGINAL = rdp.dissipator


def one_hot(i):
    return [1.0 if k == i else 0.0 for k in range(psim.N)]


def exchange_terms(i, j):
    """L = sigma^-_i sigma^+_j = (X_i X_j - i X_i Y_j + i Y_i X_j + Y_i Y_j) / 4."""
    a, c = psim.bit(i), psim.bit(j)
    return [(a | c, 0, 0.25), (a | c, c, -0.25j), (a | c, a, 0.25j), (a | c, a | c, 0.25)]


def raise_terms(i):
    """L = sigma^+_i = (X_i - i Y_i) / 2."""
    return [(psim.bit(i), 0, 0.5), (psim.bit(i), psim.bit(i), -0.5j)]


def dissipator(name):
    """Unit-rate generator with Tr K = N 2^(N-1) = 80, as for the other families."""
    if name == "exchange":
        # 40% of the budget on local loss, 60% on exchange along all ordered pairs,
        # as in the first version: Tr(s+ s-) = 16 per site, Tr = 8 per ordered pair.
        loss = sum(psim.dissipator_gen(psim.lower_terms(one_hot(i))) for i in range(psim.N))
        swap = sum(psim.dissipator_gen(exchange_terms(i, j))
                   for i in range(psim.N) for j in range(psim.N) if i != j)
        return 0.4 * loss + 0.3 * swap
    if name == "dephasing":
        # Z_i^2 = 1 gives Tr = 32 per site.
        return 0.5 * sum(psim.dissipator_gen([(0, psim.bit(i), 1.0)]) for i in range(psim.N))
    if name == "gainloss":
        # downward and upward rates in ratio 1 : 0.3, as in the first version.
        down = sum(psim.dissipator_gen(psim.lower_terms(one_hot(i))) for i in range(psim.N))
        up = sum(psim.dissipator_gen(raise_terms(i)) for i in range(psim.N))
        return (down + 0.3 * up) / 1.3
    return _ORIGINAL(name)


def memory_case(task):
    seed, name, gamma = task
    rdp.dissipator = dissipator          # route the shared code through the new families
    out = rdp.memory_case((seed, name, gamma))
    out.pop("reduced", None)
    out.pop("reduced_rates", None)
    return out


def kernel_case(name):
    D = dissipator(name)
    s = np.linalg.svd(D, compute_uv=False)
    lam = np.linalg.eigvals(D)
    return {"design": name, "kernel": int(np.sum(s < 1e-9)),
            "gap": float(np.sort(-lam.real)[np.sum(s < 1e-9)])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("memory", "kernel"))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "kernel":
        with (OUT / "kernel.jsonl").open("w") as handle:
            for name in ALL:
                record = kernel_case(name)
                handle.write(json.dumps(record) + "\n")
                print(record, flush=True)
        return
    seeds = rfs.deterministic_seeds(args.seeds)
    tasks_ = [(s, n, g) for n in FAMILIES for g in rdp.GAMMAS for s in seeds]
    path = OUT / "memory.jsonl"
    with get_context("spawn").Pool(args.workers) as pool, path.open("w") as handle:
        for i, record in enumerate(pool.imap_unordered(memory_case, tasks_), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"{i}/{len(tasks_)}", flush=True)


if __name__ == "__main__":
    main()
