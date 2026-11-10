#!/usr/bin/env python3
"""The input must follow the clock: five-spin runs for paper_v2.

clock         time to forget of the input-averaged step (Fig. 3a) for local loss and opposite phases at input
              amplitudes 1, 1/2, 1/4 and 1/8 along the loss rate; tests the collapse onto a^2 t_f against
              gamma/a^2 (local) and gamma a^2 (opposite phases).
sampling      memory from exact features and from n preparations per setting and step (three grouped settings,
              ridge chosen on the validation inputs) along the gentle-writing lines of local loss and opposite phases,
              with fixed-amplitude baselines at the same loss rates; also local loss at gamma = 64.
line          the gentle line of local loss at gamma = 1/2 and 1/4, for Fig. 2c.
pattern_clock, pattern_sampling
              the clock and the sampling test with the input on spin 1 only, instead of on all spins.
budget        memory at a fixed budget of preparations times replayed steps: memory at preparations per setting and
              step from 10^5 to 10^8.5 and the range tau_90 that holds 90% of the exact recall, along the gentle lines
              and the full-amplitude lines of local loss and opposite phases and for the equal-phase shared channel.
polarization  all eight shapes at a matched clock and input amplitude: feature variance and memory against
              the stationary state's response to the input, -i[H_in, rho*], on the slow modes.
budget_low    the lines compared in the text at 10^3 to 10^4.5 preparations per setting and step (own random stream).
replay        the budget points of the capacity table with the replay that the budget charges: every output step replays its
              last tau_90 inputs from |00000>, and the features come from that state; memory at 10^9 preparations
              times replayed steps, next to the same estimate from the state with the full history.
washout       exact memory of local loss with gentle writing at the weakest loss rates, after the usual washout and
              after five times it (same recorded inputs).
budget_replay the budget points with the features of that replay: memory at budgets of 10^8, 10^9 and 10^10
              preparations times replayed steps, n = B / (3 tau_90) preparations per setting and step, when every
              output step replays its last tau_90 inputs from |00000> (Fig. 4a and the capacity table).
replay_shots, pattern_replay_shots
              memory against preparations per setting and step, 10^3 to 10^8.5, with the features of that replay, at
              the crossings of Sec. 3.4 (local loss at 1/32, opposite phases at 64; gentle and full amplitude) and
              at strong loss with the full amplitude (equal phases and local loss at 64), with the input on all spins
              and on spin 1 only.

Run with:  python experiments/run_clock.py STAGE [--seeds 4] [--workers 7]   (resumes; four reservoirs by default)
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

import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402

import pscore  # noqa: E402
import psim  # noqa: E402
import run_dark_projection as rdp  # noqa: E402
import run_final_scaling as rfs  # noqa: E402
import run_jump_families as rjf  # noqa: E402
import run_review_runs as rrr  # noqa: E402

OUT = HERE.parent / "results" / "clock"
AMPLITUDES = (1.0, 0.5, 0.25, 0.125)
CLOCK_GAMMAS = {"separate": tuple(float(2.0 ** k) for k in range(-8, 7)),      # beyond, t_f is below one step
                "signed": tuple(float(2.0 ** k) for k in range(-2, 13))}
SHOTS = (10 ** 6, 10 ** 7, 10 ** 8)
TARGET = 32.0                      # matched leak time (steps) for the polarization stage
T0 = 4.0                           # leak time at which the gentle amplitude reaches one


def dissipator(name):
    return rjf.dissipator(name)


def drive(eta, pattern="uniform"):
    """Input pattern: the transverse field of every spin, or of spin 1 only."""
    d = np.ones(psim.N) if pattern == "uniform" else np.eye(psim.N)[0]
    return eta * d


def steps(name, gamma, eta, seed, pattern="uniform"):
    J, _ = psim.couplings_and_inputs(seed, 10)
    H0, H1 = rdp.hamiltonian(J, drive(eta, pattern))
    return H0, H1, H0 + gamma * dissipator(name)


# --------------------------------------------------------------------------------------------- run A: the clock
def clock_case(task, pattern="uniform"):
    seed, name, gamma, amp = task
    H0, H1, G = steps(name, gamma, amp, seed, pattern)
    p = psim.Propagator(G, H1, nodes=rdp.NODES)
    lam = np.sort(np.abs(np.linalg.eigvals(rdp.mean_step(p))))[::-1]
    rates = -np.log(np.maximum(lam[1:], 1e-300)) / psim.DT
    slow = rrr.KERNEL[name] - 1
    kappa = float(np.mean(rates[:slow])) if slow > 0 else float(rates[0])
    return {"seed": seed, "design": name, "gamma": gamma, "amplitude": amp, "forget": 1.0 / (psim.DT * kappa),
            "slowest": float(rates[0])}


# ------------------------------------------------------------------------------------------ run B: sampling
def simulate(seed, name, gamma, eta, pattern="uniform", extra=0, full=False):
    """Outcome distributions of the three settings along a run, after the washout; `extra` steps with inputs of
    their own run before it (a longer washout with the same recorded inputs); full=True also returns the
    propagator, the whole input stream and the washout."""
    wash = rdp.washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + rdp.TRAIN + rdp.VALIDATION + rdp.TEST)
    H0, H1 = rdp.hamiltonian(J, drive(eta, pattern))
    p = psim.Propagator(H0 + gamma * dissipator(name), H1, nodes=rdp.NODES)
    r = psim.initial_vector()
    for value in np.random.default_rng(seed + 999).uniform(0.0, 1.0, extra):
        r = p.step(r, float(value))
    strings = np.empty((len(inputs) - wash, 3, 32))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            strings[k - wash] = r[psim.WALSH]
    P = np.clip(np.einsum("tam,bm->tab", strings, psim.HAD) / 2 ** psim.N, 0.0, None)
    P /= P.sum(axis=2, keepdims=True)
    if full:
        return P, inputs[wash:], p, inputs, wash
    return P, inputs[wash:]


def scores(P, post, seed, shots=SHOTS):
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    delays = tuple(range(1, 301))
    X = pscore.features_from_probs(P)
    caps, _ = pscore.selected(X, post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    var = X[-te:].var(axis=0)
    out = {"exact": float(caps.sum()), "curve": caps.round(5).tolist(), "var_mean": float(var.mean()),
           "var_median": float(np.median(var))}
    rng = np.random.default_rng(seed + 303)
    for n in shots:
        caps, _ = pscore.selected(pscore.sampled_features(P, n, rng), post, delays, tr=tr, va=va, te=te)
        out[f"shots_{n}"] = float(caps.sum())
        out[f"curve_{n}"] = caps.round(5).tolist()
    return out


def amplitude(rule, gamma):
    if rule == "fixed":
        return 1.0
    if rule == "gentle_strong":
        return float(min(1.0, gamma ** -0.5))
    if rule == "gentle_weak":
        return float(min(1.0, gamma ** 0.5))
    raise KeyError(rule)


def sampling_case(task, pattern="uniform"):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    P, post = simulate(seed, name, gamma, eta, pattern)
    return {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, **scores(P, post, seed)}


def pattern_clock_case(task):
    return clock_case(task, pattern="single")


def pattern_sampling_case(task):
    return sampling_case(task, pattern="single")


# ----------------------------------------------------------------------------------------- fixed budget
BUDGET_SHOTS = tuple(int(round(10 ** e)) for e in np.arange(5.0, 8.75, 0.5))


def budget_case(task):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    P, post = simulate(seed, name, gamma, eta)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    delays = tuple(range(1, 301))
    caps, _ = pscore.selected(pscore.features_from_probs(P), post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    cum = np.cumsum(caps)
    out = {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, "exact": float(caps.sum()),
           "tau90": int(np.searchsorted(cum, 0.9 * cum[-1]) + 1), "shots": list(BUDGET_SHOTS), "memory": []}
    rng = np.random.default_rng(seed + 404)
    for n in BUDGET_SHOTS:
        caps, _ = pscore.selected(pscore.sampled_features(P, n, rng), post, delays, tr=tr, va=va, te=te)
        out["memory"].append(float(caps.sum()))
    return out


LOW_SHOTS = tuple(int(round(10 ** e)) for e in np.arange(3.0, 5.0, 0.5))


def budget_low_case(task):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    P, post = simulate(seed, name, gamma, eta)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    delays = tuple(range(1, 301))
    rng = np.random.default_rng(seed + 505)
    out = {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, "shots": list(LOW_SHOTS), "memory": []}
    for n in LOW_SHOTS:
        caps, _ = pscore.selected(pscore.sampled_features(P, n, rng), post, delays, tr=tr, va=va, te=te)
        out["memory"].append(float(caps.sum()))
    return out


# -------------------------------------------------------------------------------------- run C: polarization
# ------------------------------------------------------------------------------ replay and washout checks
REPLAY_BUDGET = 1e9
REPLAY_BUDGETS = (1e8, 1e9, 1e10)
REPLAY_POINTS = (("separate", "fixed", 0.0625), ("signed", "fixed", 64.0), ("shared", "fixed", 1.0),
                 ("separate", "gentle_weak", 0.015625), ("signed", "gentle_strong", 64.0))
WASHOUT_POINTS = (("separate", "gentle_weak", 0.015625), ("separate", "gentle_weak", 0.03125))


def node_weights(p, s):
    """Barycentric weights of the Chebyshev nodes of a propagator for many inputs at once, (nodes, inputs)."""
    x = (2 * np.asarray(s, float) - p.hi - p.lo) / (p.hi - p.lo)
    q = p.w[:, None] / (x[None, :] - p.x[:, None])
    return q / q.sum(axis=0, keepdims=True)


def replayed(p, inputs, first, count, length):
    """Outcome distributions of the three settings for the output steps i = 0 .. count-1 when each replays the
    `length` inputs up to and including inputs[first + i] from |00000>, all output steps at once."""
    R = np.repeat(psim.initial_vector()[:, None], count, axis=1)
    for j in range(length):
        W = node_weights(p, inputs[first - length + 1 + j: first - length + 1 + j + count])
        out = np.zeros_like(R)
        for k in range(p.K):
            out += W[k] * (p.P[k] @ R)
        R = out
    strings = R.T[:, psim.WALSH]
    P = np.clip(np.einsum("tam,bm->tab", strings, psim.HAD) / 2 ** psim.N, 0.0, None)
    return P / P.sum(axis=2, keepdims=True)


def replay_case(task):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    P, post, p, inputs, wash = simulate(seed, name, gamma, eta, full=True)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    delays = tuple(range(1, 301))
    X = pscore.features_from_probs(P)
    caps, _ = pscore.selected(X, post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    cum = np.cumsum(caps)
    tau90 = int(np.searchsorted(cum, 0.9 * cum[-1]) + 1)
    n = int(round(REPLAY_BUDGET / (3 * tau90)))
    Q = replayed(p, inputs, wash, len(post), tau90)
    Xq = pscore.features_from_probs(Q)
    caps_q, _ = pscore.selected(Xq, post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    full_state, _ = pscore.selected(pscore.sampled_features(P, n, np.random.default_rng(seed + 606)), post, delays,
                                    tr=tr, va=va, te=te)
    replay_state, _ = pscore.selected(pscore.sampled_features(Q, n, np.random.default_rng(seed + 606)), post, delays,
                                      tr=tr, va=va, te=te)
    offset = float(np.mean((Xq - X)[-te:] ** 2) / np.mean(X[-te:].var(axis=0)))
    return {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, "tau90": tau90, "shots": n,
            "exact": float(caps.sum()), "replay_exact": float(caps_q.sum()), "full_state": float(full_state.sum()),
            "replay": float(replay_state.sum()), "offset": offset}


def budget_replay_case(task):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    P, post, p, inputs, wash = simulate(seed, name, gamma, eta, full=True)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    delays = tuple(range(1, 301))
    caps, _ = pscore.selected(pscore.features_from_probs(P), post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    cum = np.cumsum(caps)
    tau90 = int(np.searchsorted(cum, 0.9 * cum[-1]) + 1)
    Q = replayed(p, inputs, wash, len(post), tau90)
    caps_q, _ = pscore.selected(pscore.features_from_probs(Q), post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    shots = [int(round(b / (3 * tau90))) for b in REPLAY_BUDGETS]
    out = {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, "exact": float(caps.sum()),
           "replay_exact": float(caps_q.sum()), "tau90": tau90, "budgets": list(REPLAY_BUDGETS), "shots": shots,
           "memory": []}
    rng = np.random.default_rng(seed + 404)
    for n in shots:
        caps, _ = pscore.selected(pscore.sampled_features(Q, n, rng), post, delays, tr=tr, va=va, te=te)
        out["memory"].append(float(caps.sum()))
    return out


REPLAY_SHOTS = tuple(int(round(10 ** e)) for e in np.arange(3.0, 8.75, 0.5))
REPLAY_LINES = (("separate", 0.03125, "gentle_weak"), ("separate", 0.03125, "fixed"),
                ("signed", 64.0, "gentle_strong"), ("signed", 64.0, "fixed"),
                ("shared", 64.0, "fixed"), ("separate", 64.0, "fixed"))


def replay_shots_case(task, pattern="uniform"):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    P, post, p, inputs, wash = simulate(seed, name, gamma, eta, pattern, full=True)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    delays = tuple(range(1, 301))
    caps, _ = pscore.selected(pscore.features_from_probs(P), post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    cum = np.cumsum(caps)
    tau90 = int(np.searchsorted(cum, 0.9 * cum[-1]) + 1)
    Q = replayed(p, inputs, wash, len(post), tau90)
    Xq = pscore.features_from_probs(Q)
    caps_q, _ = pscore.selected(Xq, post, delays, tr=tr, va=va, te=te, ridges=(0.0,))
    out = {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, "tau90": tau90,
           "exact": float(caps.sum()), "replay_exact": float(caps_q.sum()),
           "signal_replay": float(Xq[-te:].var(axis=0).mean()), "shots": list(REPLAY_SHOTS), "memory": []}
    rng = np.random.default_rng(seed + 707)
    for n in REPLAY_SHOTS:
        caps, _ = pscore.selected(pscore.sampled_features(Q, n, rng), post, delays, tr=tr, va=va, te=te)
        out["memory"].append(float(caps.sum()))
    return out


def pattern_replay_shots_case(task):
    return replay_shots_case(task, pattern="single")


def washout_case(task):
    seed, name, gamma, rule = task
    eta = amplitude(rule, gamma)
    tr, va, te = rdp.TRAIN, rdp.VALIDATION, rdp.TEST
    out = {"seed": seed, "design": name, "gamma": gamma, "rule": rule, "eta": eta, "washout": rdp.washout(gamma)}
    for label, extra in (("exact", 0), ("exact_long", 4 * rdp.washout(gamma))):
        P, post = simulate(seed, name, gamma, eta, extra=extra)
        caps, _ = pscore.selected(pscore.features_from_probs(P), post, tuple(range(1, 301)), tr=tr, va=va, te=te,
                                  ridges=(0.0,))
        out[label] = float(caps.sum())
    return out


def generator_modes(name, gamma, seed):
    """Rates, right and left eigenvectors of the generator at the mean input (stationary state first)."""
    _, _, G = steps(name, gamma, 1.0, seed)
    _, H1 = rdp.hamiltonian(psim.couplings_and_inputs(seed, 10)[0])
    lam, Rv = np.linalg.eig(G + 0.5 * H1)
    order = np.argsort(-lam.real)
    lam, Rv = lam[order], Rv[:, order]
    Lv = np.linalg.inv(Rv)
    return -lam.real, Rv, Lv


def leak_time(name, gamma, seed):
    _, H1, G = steps(name, gamma, 1.0, seed)
    rates = np.sort(-np.linalg.eigvals(G + 0.5 * H1).real)[1:]
    slow = rrr.KERNEL[name] - 1
    kappa = float(np.mean(rates[:slow])) if slow > 0 else float(rates[0])
    return 1.0 / (psim.DT * kappa)


def matched_gamma(name, seed, target=TARGET):
    """Loss rate at which the leak time equals the target: weak loss for shapes whose clock slows with weaker
    loss, strong loss for shapes whose clock slows with stronger loss."""
    strong = rrr.KERNEL[name] > 1
    lo, hi = (8.0, 4096.0) if strong else (1.0 / 4096, 1.0)
    for _ in range(24):
        mid = np.sqrt(lo * hi)
        t = leak_time(name, mid, seed)
        if (t < target) == strong:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def polarization_case(task):
    seed, name = task
    gamma = matched_gamma(name, seed)
    rates, Rv, Lv = generator_modes(name, gamma, seed)
    ident = psim.pid(np.int64(0), np.int64(0))
    stationary = np.real(Rv[:, 0] / Rv[ident, 0])           # normalized so that the identity coordinate is one
    _, H1 = rdp.hamiltonian(psim.couplings_and_inputs(seed, 10)[0])
    write = H1 @ stationary                                 # -i[H_in, rho*] per unit input, Pauli coordinates
    coeff = Lv @ write
    slow = rrr.KERNEL[name] - 1 if rrr.KERNEL[name] > 1 else int(np.sum(rates[1:] <= 4 * rates[1]))
    on_slow = np.real(Rv[:, 1:slow + 1] @ coeff[1:slow + 1])
    eta = float(min(1.0, np.sqrt(T0 / TARGET)))
    P, post = simulate(seed, name, gamma, eta)
    return {"seed": seed, "design": name, "gamma": gamma, "eta": eta, "slow_modes": slow,
            "leak_time": leak_time(name, gamma, seed), "write_slow": float(np.linalg.norm(on_slow)),
            "write_all": float(np.linalg.norm(write)), "polarization": float(np.linalg.norm(stationary[1:])),
            **scores(P, post, seed, shots=(10 ** 6,))}


# ------------------------------------------------------------------------------------------------------ jobs
STAGES = {"clock": clock_case, "sampling": sampling_case, "line": sampling_case, "budget": budget_case,
          "pattern_clock": pattern_clock_case, "pattern_sampling": pattern_sampling_case,
          "polarization": polarization_case, "budget_low": budget_low_case, "replay": replay_case,
          "washout": washout_case, "budget_replay": budget_replay_case, "replay_shots": replay_shots_case,
          "pattern_replay_shots": pattern_replay_shots_case}


def jobs(stage, seeds):
    if stage == "clock":
        return [(s, n, g, a) for n in ("separate", "signed") for a in AMPLITUDES for g in CLOCK_GAMMAS[n]
                for s in seeds]
    if stage == "sampling":
        tasks = []
        for g in (0.125, 0.0625, 0.03125):
            tasks += [(s, "separate", g, r) for r in ("gentle_weak", "fixed") for s in seeds]
        for g in (16.0, 64.0, 256.0):
            tasks += [(s, "signed", g, r) for r in ("gentle_strong", "fixed") for s in seeds]
        tasks += [(s, "separate", 64.0, "fixed") for s in seeds]
        return tasks
    if stage == "pattern_clock":    # a coarser grid than the clock stage
        grid = {"separate": tuple(float(2.0 ** k) for k in range(-8, 7, 2)),
                "signed": tuple(float(2.0 ** k) for k in range(-2, 13, 2))}
        return [(s, n, g, a) for n in ("separate", "signed") for a in AMPLITUDES for g in grid[n] for s in seeds]
    if stage == "pattern_sampling":
        return ([(s, "signed", 64.0, r) for r in ("gentle_strong", "fixed") for s in seeds]
                + [(s, "separate", 0.03125, r) for r in ("gentle_weak", "fixed") for s in seeds])
    if stage == "budget_low":       # the lines that the text compares at few preparations
        return [(s, n, g, r) for n, g in (("signed", 64.0), ("separate", 0.03125), ("separate", 0.0625))
                for r in ("gentle_strong" if n == "signed" else "gentle_weak", "fixed") for s in seeds]
    if stage in ("budget", "budget_replay"):   # the most expensive first, so that no long run starts last
        tasks = []
        for g in (256.0, 128.0, 64.0, 32.0, 16.0, 8.0, 4.0, 2.0):
            tasks += [(s, "signed", g, "gentle_strong") for s in seeds]
            if g in (256.0, 64.0, 16.0, 4.0):
                tasks += [(s, "signed", g, "fixed") for s in seeds]
            if g in (64.0, 16.0, 4.0):
                tasks += [(s, "shared", g, "fixed") for s in seeds]
        tasks += [(s, n, 1.0, "fixed") for n in ("signed", "shared", "separate") for s in seeds]
        for g in (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            tasks += [(s, "separate", g, r) for r in ("gentle_weak", "fixed") for s in seeds]
        return tasks
    if stage == "replay":
        return [(s, n, g, r) for n, r, g in REPLAY_POINTS for s in seeds]
    if stage == "washout":
        return [(s, n, g, r) for n, r, g in WASHOUT_POINTS for s in seeds]
    if stage in ("replay_shots", "pattern_replay_shots"):
        return [(s, n, g, r) for n, g, r in REPLAY_LINES for s in seeds]
    if stage == "line":             # the rest of the gentle line of local loss, for Fig. 2c
        return [(s, "separate", g, "gentle_weak") for g in (0.5, 0.25) for s in seeds]
    return [(s, n) for n in rrr.FAMILIES for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.stage}.jsonl"
    tasks = jobs(args.stage, rfs.deterministic_seeds(8)[:args.seeds])
    key = lambda t: json.dumps(list(t))
    fields = {"clock": ("seed", "design", "gamma", "amplitude"), "pattern_clock": ("seed", "design", "gamma", "amplitude"),
              "polarization": ("seed", "design")}
    names = fields.get(args.stage, ("seed", "design", "gamma", "rule"))
    done = set()
    if path.exists():                # resume: keep finished records
        done = {key(tuple(r[f] for f in names)) for r in map(json.loads, path.open())}
    tasks = [t for t in tasks if key(t) not in done]
    print(f"{len(tasks)} tasks left", flush=True)
    with get_context("spawn").Pool(args.workers) as pool, path.open("a") as handle:
        for i, record in enumerate(pool.imap_unordered(STAGES[args.stage], tasks), 1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"{i}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
