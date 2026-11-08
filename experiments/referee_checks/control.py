#!/usr/bin/env python3
"""Referee checks (not part of the paper).

1. Weak local loss with gentle writing, eta = sqrt(gamma) for gamma <= 1, against opposite phases at strong loss with
   eta = min(1, gamma^(-1/2)) and the equal-phase shared channel at a fixed amplitude.
2. Finite sampling: memory from n preparations per setting and step (three settings, all spins along X, Y or Z),
   with the ridge chosen on the validation inputs, as in the first version of the study.

Same reservoir, seeds, washout and train/validation/test split as paper_v2 (run_dark_projection.py).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "theory" / "finite_shots_vs_gamma"))

import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402

import psim  # noqa: E402
import pscore  # noqa: E402
import run_dark_projection as rdp  # noqa: E402
import run_final_scaling as rfs  # noqa: E402

OUT = Path(__file__).resolve().parent / "control.jsonl"
SHOTS = (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8)
AXES = psim.WALSH                      # (3, 32) Pauli strings of one axis on every subset of spins


def eta_rule(rule, gamma):
    if rule == "fixed":
        return 1.0
    if rule == "gentle_strong":        # paper: eta = min(1, gamma^(-1/2))
        return float(min(1.0, gamma ** -0.5))
    if rule == "gentle_weak":          # control: eta = min(1, gamma^(1/2))
        return float(min(1.0, gamma ** 0.5))
    raise KeyError(rule)


def run(task):
    seed, name, gamma, rule = task
    t0 = time.time()
    eta = eta_rule(rule, gamma)
    wash = rdp.washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + rdp.TRAIN + rdp.VALIDATION + rdp.TEST)
    H0, H1 = rdp.hamiltonian(J, eta * np.ones(psim.N))
    p = psim.Propagator(H0 + gamma * rdp.dissipator(name), H1, nodes=rdp.NODES)
    r = psim.initial_vector()
    strings = np.empty((len(inputs) - wash, 3, 32))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            strings[k - wash] = r[AXES]
    post = inputs[wash:]
    P = np.einsum("tam,bm->tab", strings, psim.HAD) / 2 ** psim.N      # outcome probabilities (T, 3, 32)
    P = np.clip(P, 0.0, None)
    P /= P.sum(axis=2, keepdims=True)
    X = pscore.features_from_probs(P)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    out = {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta}
    for D in (300, 600):
        caps, _ = pscore.selected(X, post, tuple(range(1, D + 1)), tr=tr, va=va, te=te, ridges=(0.0,))
        out[f"exact_{D}"] = float(caps.sum())
        if D == 300:
            out["curve"] = caps.round(5).tolist()
            c = np.cumsum(caps)
            out["tau90"] = int(np.searchsorted(c, 0.9 * c[-1]) + 1)
    caps, ridge = pscore.selected(X, post, tuple(range(1, 301)), tr=tr, va=va, te=te)
    out["exact_ridge"], out["exact_ridge_pick"] = float(caps.sum()), float(ridge)
    var = X[-te:].var(axis=0)
    out["var_mean"], out["var_median"], out["var_min"] = float(var.mean()), float(np.median(var)), float(var.min())
    rng = np.random.default_rng(seed + 101)
    for n in SHOTS:
        Xn = pscore.sampled_features(P, n, rng)
        caps, ridge = pscore.selected(Xn, post, tuple(range(1, 301)), tr=tr, va=va, te=te)
        out[f"shots_{n}"], out[f"ridge_{n}"] = float(caps.sum()), float(ridge)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def jobs(seeds):
    tasks = []
    for g in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        tasks += [(s, "separate", g, "fixed") for s in seeds]
        if g < 1:
            tasks += [(s, "separate", g, "gentle_weak") for s in seeds]
    for g in (1.0, 4.0, 16.0, 64.0):
        tasks += [(s, "signed", g, "gentle_strong") for s in seeds]
        tasks += [(s, "shared", g, "fixed") for s in seeds]
    for g in (16.0, 64.0):
        tasks += [(s, "signed", g, "fixed") for s in seeds]
    return tasks


def main():
    seeds = rfs.deterministic_seeds(8)[:4]
    tasks = jobs(seeds)
    done = set()
    if OUT.exists():
        done = {(r["seed"], r["design"], r["gamma"], r["rule"]) for r in map(json.loads, OUT.open())}
    tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks)} tasks", flush=True)
    with get_context("spawn").Pool(int(os.environ.get("WORKERS", "8"))) as pool, OUT.open("a") as handle:
        for i, rec in enumerate(pool.imap_unordered(run, tasks), 1):
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            print(f"{i}/{len(tasks)} {rec['design']} g={rec['gamma']} {rec['rule']} exact={rec['exact_300']:.2f} "
                  f"({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
