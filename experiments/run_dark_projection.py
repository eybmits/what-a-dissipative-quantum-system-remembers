#!/usr/bin/env python3
"""Zeno projection onto the dark subspace of a shared channel.

spectrum    eigenvalues of the input-averaged step map along the strength.
rates       decay rates of the generator at the mean input along the strength: dark and bright bands.
flow        every decay rate of the generator at the mean input along the strength, shared channel, for the reservoir
            of the spectrum stage (Fig. 2d).
memory      recall to long delays for five channel patterns, with capacity by degree;
            for a shared channel also the reservoir with the bright states eliminated.
far         the eliminated model alone, far beyond the reach of the full simulation.
removal     erase the dark coordinates after every step, or a rotated copy of them.
write       response of the dark coordinates to one input, along and across the channel.
size        recall against input size along and across the channel, exact and with shots.
size_edge   larger inputs along the channel, past the edge of that grid.
local       weak local decay on top of a shared channel.
forecast    closed-loop Mackey-Glass error along the strength.

Run with:  python experiments/run_dark_projection.py STAGE
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

import _paths  # noqa: F401
import numpy as np
import scipy.linalg as sla

import pscore
import run_ipc_wide as ipc
import psim
import run_final_scaling as rfs
from qrc import tasks

OUT = HERE.parent / "results" / "dark_projection"
DELAYS = tuple(range(1, 301))
TRAIN, VALIDATION, TEST = 2000, 400, 1600
GAMMAS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
FAR = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0)
DESIGNS = ("separate", "groups", "signed", "shared", "pair")
ipc.CLASSES = (((1,), 120), ((2,), 120), ((1, 1), 40), ((3,), 60), ((2, 1), 30), ((1, 1, 1), 20),
               ((4,), 40), ((3, 1), 16), ((2, 2), 16), ((2, 1, 1), 12), ((1, 1, 1, 1), 12))
NODES = 10
SIZES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
SHOTS = (65536, 1048576, 16777216)              # 45 q shots per step in all


def washout(gamma):
    return int(max(800, 60 * gamma))


def dissipator(name):
    b = psim.blocks()
    if name in b:
        return b[name]
    if name == "groups":
        return psim.dissipator_gen(psim.lower_terms([1, 1, 1, 0, 0])) + psim.dissipator_gen(psim.lower_terms([0, 0, 0, 1, 1]))
    if name == "pair":
        total = 0.0
        for i in range(psim.N):
            for j in range(i + 1, psim.N):
                a, c = psim.bit(i), psim.bit(j)
                total = total + psim.dissipator_gen([(a | c, 0, 0.25), (a | c, a, 0.25j), (a | c, c, 0.25j),
                                                     (a | c, a | c, -0.25)])
        return total
    raise KeyError(name)


def hamiltonian(J, drive=None):
    """Static and input generators; ``drive`` turns the input and keeps the mean drive."""
    b = psim.blocks()
    G = psim.H_FIELD * b["gz"] + 1.5 * psim.H_FIELD * b["gx"]
    for key, value in J.items():
        G = G + value * b["xx"][key]
    d = np.ones(psim.N) if drive is None else np.asarray(drive)
    H1 = psim.H_FIELD * sum(di * psim.comm_gen(np.int64(psim.bit(i)), np.int64(0)) for i, di in enumerate(d))
    return G - 0.5 * H1, H1


def memory(features, post, delays=DELAYS, train=TRAIN, validation=VALIDATION, test=TEST):
    caps, _ = pscore.selected(features, post, delays, tr=train, va=validation, te=test, ridges=(0.0,))
    return caps.round(5).tolist()


def mean_step(p):
    x, w = np.polynomial.legendre.leggauss(6)
    return sum(wi * p.dense(float(0.5 * (xi + 1))) for xi, wi in zip(x, 0.5 * w))


def drive_steady(G0, G1, inputs, wash):
    p = psim.Propagator(G0, G1, nodes=NODES)
    r = psim.initial_vector()
    rows = np.empty((len(inputs) - wash, 45))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            rows[k - wash] = r[psim.FEAT]
    return rows, p


def kernel():
    D = psim.blocks()["shared"]
    u, s, vt = np.linalg.svd(D)
    R = vt[s < 1e-9].T
    L = u[:, s < 1e-9]
    L = L @ np.linalg.inv(R.T @ L)
    return R, L, np.linalg.inv(D + R @ L.T) - R @ L.T


class Reduced:
    """Dark coordinates c with r = (1 - S H(s) / gamma) R c, to second order in 1/gamma."""

    def __init__(self, H0, H1, gamma, extra=None):
        R, L, S = kernel()
        self.first = L.T @ H0 @ R
        A, B = S @ H0 @ R, S @ H1 @ R
        self.second = (-L.T @ H0 @ A, -L.T @ (H0 @ B + H1 @ A), -L.T @ H1 @ B)
        self.gamma = gamma
        self.F0, self.F1 = (R - A / gamma)[psim.FEAT], (-B / gamma)[psim.FEAT]
        self.c0 = L.T @ psim.initial_vector()
        self.extra = 0.0 if extra is None else L.T @ extra @ R
        theta = np.pi * (2 * np.arange(NODES) + 1) / (2 * NODES)
        self.x, self.bw = np.cos(theta), (-1.0) ** np.arange(NODES) * np.sin(theta)
        self.P = np.stack([self.step(float(0.5 * (x + 1))) for x in self.x])

    def generator(self, s):
        n0, n1, n2 = self.second
        return self.first + (n0 + s * n1 + s * s * n2) / self.gamma + self.extra

    def step(self, s):
        return sla.expm(psim.DT * self.generator(s))

    def rates(self):
        x, w = np.polynomial.legendre.leggauss(6)
        mean = sum(wi * self.step(float(0.5 * (xi + 1))) for xi, wi in zip(x, 0.5 * w))
        return np.sort(-np.log(np.abs(np.linalg.eigvals(mean))) / psim.DT)[1:]

    def run(self, inputs, wash):
        c = self.c0.copy()
        rows = np.empty((len(inputs) - wash, 45))
        for k, value in enumerate(inputs):
            q = self.bw / (2 * value - 1 - self.x)
            c = (q / q.sum()) @ (self.P @ c)
            if k >= wash:
                rows[k - wash] = (self.F0 + value * self.F1) @ c
        return rows


def spectrum_case(task):
    seed, name, gamma = task
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = hamiltonian(J)
    mean = mean_step(psim.Propagator(H0 + gamma * dissipator(name), H1, nodes=NODES))
    lam = np.linalg.eigvals(mean)
    lam = lam[np.argsort(-np.abs(lam))]
    return {"seed": seed, "design": name, "gamma": gamma, "re": lam.real[:130].tolist(), "im": lam.imag[:130].tolist()}


def rates_case(task):
    """Decay rates of the generator at the mean input: the slowest and the fastest mode and, when the unit-rate
    dissipator protects operators, the edges of the dark band (one mode for each protected operator besides the
    stationary state) and of the bright band above it."""
    seed, name, gamma = task
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = hamiltonian(J)
    D = dissipator(name)
    protected = int(np.sum(np.linalg.svd(D, compute_uv=False) < 1e-9))
    rates = np.sort(-np.linalg.eigvals(H0 + 0.5 * H1 + gamma * D).real)[1:]
    out = {"seed": seed, "design": name, "gamma": gamma, "protected": protected,
           "slowest": float(rates[0]), "fastest": float(rates[-1])}
    if protected > 1:
        out["dark_top"], out["bright_bottom"] = float(rates[protected - 2]), float(rates[protected - 1])
    return out


def flow_case(task):
    """Every decay rate of the generator at the mean input, sorted, without the stationary state."""
    seed, name, gamma = task
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = hamiltonian(J)
    rates = np.sort(-np.linalg.eigvals(H0 + 0.5 * H1 + gamma * dissipator(name)).real)[1:]
    return {"seed": seed, "design": name, "gamma": gamma, "rates": [float(f"{r:.6g}") for r in rates]}


def memory_case(task):
    seed, name, gamma = task
    wash = washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + TRAIN + VALIDATION + TEST)
    H0, H1 = hamiltonian(J)
    rows, p = drive_steady(H0 + gamma * dissipator(name), H1, inputs, wash)
    post = inputs[wash:]
    lam = np.abs(np.linalg.eigvals(mean_step(p)))
    out = {"seed": seed, "design": name, "gamma": gamma, "memory": memory(rows, post),
           "slow_modes": int(np.sum(lam > 0.5)) - 1,
           "rates": np.sort(-np.log(np.maximum(lam, 1e-300)) / psim.DT)[1:100].round(6).tolist()}
    capacity = ipc.evaluate(rows, 2 * inputs - 1, wash, np.random.default_rng(seed + 11))
    out["by_degree"], out["rank"] = capacity["by_degree"], capacity["rank"]
    if name == "shared" and gamma >= 1:
        model = Reduced(H0, H1, gamma)
        out["reduced"] = memory(model.run(inputs, wash), post)
        out["reduced_rates"] = model.rates()[:99].round(6).tolist()
    return out


def long_recall(features, post, reach):
    """In-sample linear recall to long delays, by FFT, with the bias of the fit removed."""
    X = features - features.mean(axis=0)
    y = post - post.mean()
    T = len(y)
    size = 2 * T
    u = np.fft.rfft(y, size)
    Y = np.fft.rfft(X, size, axis=0)
    cross = np.fft.irfft(Y * np.conj(u)[:, None], size, axis=0)[1:reach + 1]
    G = np.linalg.pinv(X.T @ X, rcond=1e-13)
    return np.einsum("ij,jk,ik->i", cross, G, cross) / float(y @ y) - X.shape[1] / T


def far_case(task):
    seed, gamma = task
    wash, length, reach = washout(gamma), 400000, 30000
    J, inputs = psim.couplings_and_inputs(seed, wash + length)
    H0, H1 = hamiltonian(J)
    model = Reduced(H0, H1, gamma)
    caps = long_recall(model.run(inputs, wash), inputs[wash:], reach)
    edges = np.unique(np.round(np.geomspace(1, reach + 1, 80)).astype(int))
    return {"seed": seed, "gamma": gamma, "edges": edges.tolist(),
            "binned": [float(caps[a - 1:b - 1].sum()) for a, b in zip(edges[:-1], edges[1:])],
            "total": float(caps.sum()), **{f"q{q}": int(np.searchsorted(np.cumsum(caps), q / 100 * caps.sum()) + 1)
                                           for q in (25, 50, 75, 90)},
            "rates": model.rates()[:99].round(8).tolist()}


def product_rotation(rng):
    b = 0.0
    for i in range(psim.N):
        angles = rng.normal(size=3)
        for a, (x, z) in zip(angles, ((psim.bit(i), 0), (psim.bit(i), psim.bit(i)), (0, psim.bit(i)))):
            b = b + a * psim.comm_gen(np.int64(x), np.int64(z))
    return sla.expm(b)


def removal_case(task):
    seed, gamma = task
    wash = washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + 800 + 200 + 800)
    H0, H1 = hamiltonian(J)
    post = inputs[wash:]
    delays = tuple(range(1, 101))
    R, L, _ = kernel()
    W = product_rotation(np.random.default_rng(seed + 5))
    p = psim.Propagator(H0 + gamma * psim.blocks()["shared"], H1, nodes=NODES)
    r, mean, count = psim.initial_vector(), 0.0, 0
    rows = np.empty((len(inputs) - wash, 45))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            rows[k - wash] = r[psim.FEAT]
            mean, count = mean + r, count + 1
    mean = mean / count
    out = {"seed": seed, "gamma": gamma, "intact": memory(rows, post, delays, 800, 200, 800)}
    for arm, (A, B) in {"removed": (R, L), "rotated": (W @ R, W @ L)}.items():
        r = psim.initial_vector()
        for k, value in enumerate(inputs):
            r = p.step(r, float(value))
            r = r - A @ (B.T @ (r - mean))
            if k >= wash:
                rows[k - wash] = r[psim.FEAT]
        out[arm] = memory(rows, post, delays, 800, 200, 800)
    return out


def write_case(task):
    seed, gamma = task
    J, _ = psim.couplings_and_inputs(seed, 10)
    R, L, _ = kernel()
    theta = 2 * np.pi * np.arange(psim.N) / psim.N
    out = {"seed": seed, "gamma": gamma}
    for arm, drive in (("along", None), ("across", np.sqrt(2.0) * np.cos(theta))):
        H0, H1 = hamiltonian(J, drive)
        G0 = H0 + gamma * psim.blocks()["shared"]
        p = psim.Propagator(G0, H1, nodes=4)
        w, v = np.linalg.eig(mean_step(p))
        steady = v[:, np.argmax(w.real)].real
        steady = steady / steady[0]
        kick = (sla.expm(psim.DT * (G0 + 0.75 * H1)) - sla.expm(psim.DT * (G0 + 0.25 * H1))) @ steady / 0.5
        out[arm + "_dark"] = float(np.linalg.norm(L.T @ kick))
        out[arm + "_feature"] = float(np.linalg.norm(kick[psim.FEAT]))
        out[arm + "_first_order"] = float(np.abs(L.T @ H1 @ R).max())
    return out


def size_case(task):
    seed, gamma, arm, size = task
    wash = washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + 1200)
    theta = 2 * np.pi * np.arange(psim.N) / psim.N
    pattern = np.ones(psim.N) if arm == "along" else np.sqrt(2.0) * np.cos(theta)
    H0, H1 = hamiltonian(J, size * pattern)
    p = psim.Propagator(H0 + gamma * psim.blocks()["shared"], H1, nodes=NODES if size <= 4 else 24)
    r = psim.initial_vector()
    rows = np.empty((1200, psim.DIM))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            rows[k - wash] = r
    post = inputs[wash:]
    out = {"seed": seed, "gamma": gamma, "arm": arm, "size": size, "exact": pscore.two_scores(rows[:, psim.FEAT], post)[0], "sampled": {}}
    P = np.clip(psim.probabilities(rows), 0.0, None)
    rng = np.random.default_rng([seed, int(gamma * 8), int(size * 16)])
    for q in SHOTS:
        out["sampled"][str(q)] = pscore.two_scores(pscore.sampled_features(P, 15 * q, rng), post)[0]
    return out


def local_case(task):
    seed, gamma, loc, recall = task
    J, inputs = psim.couplings_and_inputs(seed, 10)
    H0, H1 = hamiltonian(J)
    D = loc * psim.blocks()["separate"]
    out = {"seed": seed, "gamma": gamma, "local": loc}
    p = psim.Propagator(H0 + gamma * psim.blocks()["shared"] + D, H1, nodes=NODES)
    lam = np.sort(np.abs(np.linalg.eigvals(mean_step(p))))[::-1]
    out["rates"] = (-np.log(lam[1:100]) / psim.DT).round(7).tolist()
    out["reduced_rates"] = Reduced(H0, H1, gamma, D).rates()[:99].round(7).tolist()
    if recall:
        wash = washout(gamma)
        J, inputs = psim.couplings_and_inputs(seed, wash + TRAIN + VALIDATION + TEST)
        rows, _ = drive_steady(H0 + gamma * psim.blocks()["shared"] + D, H1, inputs, wash)
        out["memory"] = memory(rows, inputs[wash:])
    return out


def forecast_case(task):
    seed, name, gamma = task
    wash, train, horizon = 4000, 1000, 150
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = hamiltonian(J)
    series = tasks.mackey_glass_series(wash + train + horizon + 5, tau=17, sample_every=3, discard=500,
                                       rng=np.random.default_rng(seed + 3))
    series = tasks.rescale_unit(series, 0.0, 1.0)
    p = psim.Propagator(H0 + gamma * dissipator(name), H1, nodes=NODES)
    r = psim.initial_vector()
    rows = np.empty((train, 45))
    for k in range(wash + train):
        r = p.step(r, float(series[k]))
        if k >= wash:
            rows[k - wash] = r[psim.FEAT]
    X = np.column_stack([rows, np.ones(train)])[:-1]
    w = np.linalg.solve(X.T @ X + 1e-8 * np.eye(46), X.T @ series[wash + 1:wash + train])
    preds = []
    for _ in range(horizon):
        value = float(np.clip(np.append(r[psim.FEAT], 1.0) @ w, 0.0, 1.0))
        preds.append(value)
        r = p.step(r, value)
    truth = series[wash + train:wash + train + horizon]
    return {"seed": seed, "design": name, "gamma": gamma, "mse": float(np.mean((np.array(preds) - truth) ** 2)),
            "constant": float(np.mean((series[wash + train - 1] - truth) ** 2))}


STAGES = {"spectrum": spectrum_case, "rates": rates_case, "flow": flow_case, "memory": memory_case, "far": far_case,
          "removal": removal_case, "write": write_case, "size": size_case, "size_edge": size_case, "local": local_case,
          "forecast": forecast_case}


def jobs(stage, seeds):
    if stage == "spectrum":
        return [(s, n, g) for g in GAMMAS for s in seeds for n in ("shared", "separate")]
    if stage == "rates":
        return [(s, n, float(g)) for g in np.geomspace(0.1, 1000.0, 81) for s in seeds for n in ("shared", "separate")]
    if stage == "flow":             # the reservoir of the spectrum stage, as export_snapshot picks it
        return [(sorted(seeds)[0], "shared", float(g)) for g in np.geomspace(0.1, 1000.0, 81)]
    if stage == "memory":
        return [(s, n, g) for n in ("shared", "separate", "signed", "groups", "pair") for g in GAMMAS for s in seeds]
    if stage == "far":
        return [(s, g) for g in FAR for s in seeds]
    if stage == "removal":
        return [(s, g) for g in (0.1, 1.0, 16.0) for s in seeds]
    if stage == "write":
        return [(s, g) for g in (0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0) for s in seeds]
    if stage == "size":
        return [(s, g, a, z) for g in (1.0, 4.0, 16.0, 64.0) for a in ("along", "across") for z in SIZES for s in seeds]
    if stage == "size_edge":
        return [(s, g, "along", z) for g in (4.0, 16.0, 64.0) for z in (8.0, 16.0) for s in seeds]
    if stage == "local":
        rates = [(s, g, l, False) for l in (0.001, 0.01, 0.1) for g in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
                 for s in seeds]
        return [(s, g, 0.01, True) for g in (4.0, 16.0, 64.0) for s in seeds] + rates
    return [(s, n, g) for g in (0.25, 1.0, 4.0, 16.0, 64.0) for n in ("shared", "separate") for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks_ = jobs(args.stage, rfs.deterministic_seeds(args.seeds))
    path = OUT / f"{args.stage}.jsonl"
    with get_context("spawn").Pool(args.workers) as pool, path.open("w") as handle:
        for i, record in enumerate(pool.imap_unordered(STAGES[args.stage], tasks_), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"{i}/{len(tasks_)}", flush=True)


if __name__ == "__main__":
    main()
