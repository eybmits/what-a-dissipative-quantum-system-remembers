#!/usr/bin/env python3
"""Reduce the raw records to the numbers the figures show, in paper_v2/data/snapshot.json.

Sources (exact expectation values, except for the finite-sampling test of the first version):
  results/dark_projection/{memory,spectrum,rates,removal,write}.jsonl   five-spin reservoir, five shapes
  results/jump_families/{memory,kernel}.jsonl                     exchange, dephasing, gain/loss; kernels
  results/spin_orders/orders.jsonl                                three-spin reservoir, order shares
  results/spin_orders/coherence.json                              three-spin recovery with/without dephasing
  results/order_law/                                              random-unitary reset reservoirs
  results/review_runs/{extend,scaled,spectrum,dephasing,variance}.jsonl   runs added for the reviews
  results/clock/*.jsonl                                           clock, sampling, budget and replay runs (run_clock.py)
  results/sizes/memory.jsonl                                      memory against loss at three, four and six spins
  results/gentle_limit/{limit,exact,dark}.jsonl                   gentle-writing limit and the recursion (run_gentle_limit.py)
  results/measurement_full_v3/measurement_full_aggregate.json     finite sampling of the first version
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "paper_v2" / "data" / "snapshot.json"
DT = 0.5
FAMILIES = ("separate", "exchange", "gainloss", "dephasing", "pair", "signed", "groups", "shared")


def jsonl(path):
    return [json.loads(line) for line in path.open()]


def half_delay(curve):
    c = np.cumsum(curve)
    return int(np.searchsorted(c, 0.5 * c[-1])) + 1


def memory_block(kernel):
    rows = jsonl(RES / "dark_projection" / "memory.jsonl")
    extra = RES / "jump_families" / "memory.jsonl"
    if extra.exists():
        rows += jsonl(extra)
    base = sorted({r["gamma"] for r in rows})
    extend = RES / "review_runs" / "extend.jsonl"
    if extend.exists():
        rows += jsonl(extend)
    out = {}
    for fam in FAMILIES:
        sel = [r for r in rows if r["design"] == fam]
        if not sel:
            continue
        gammas = sorted({r["gamma"] for r in sel})
        seeds = sorted({r["seed"] for r in sel})
        by = {(r["seed"], r["gamma"]): r for r in sel}
        slow = max(kernel[fam] - 1, 1)
        rec = {"gamma": gammas, "seeds": seeds, "recall": [], "half": [], "forget": [], "degree": [], "complete": True}
        for s in seeds:
            R, H, F, Dg = [], [], [], []
            for g in gammas:
                r = by.get((s, g))
                if r is None:
                    rec["complete"] = False
                    R.append(None); H.append(None); F.append(None); Dg.append(None)
                    continue
                curve = np.array(r["memory"])
                R.append(float(curve.sum()))
                H.append(half_delay(curve))
                F.append(float(1.0 / (DT * np.mean(np.array(r["rates"])[:slow]))))
                Dg.append([float(r["by_degree"].get(str(k), 0.0)) for k in (1, 2, 3, 4)] if "by_degree" in r else None)
            rec["recall"].append(R); rec["half"].append(H); rec["forget"].append(F); rec["degree"].append(Dg)
        if fam in ("separate", "shared"):
            curves = np.array([[by[(s, g)]["memory"] for g in base] for s in seeds])
            rec["curve_mean"] = curves.mean(axis=0).round(5).tolist()
            rec["curve_gamma"] = base
            # per-reservoir recall at two loss rates, for the bands
            rec["curve_rows"] = {f"{g:g}": curves[:, base.index(g)].round(4).tolist() for g in (0.25, 64.0)}
        if fam == "shared":
            red = {}
            for g in (4.0, 64.0):
                full = np.mean([by[(s, g)]["memory"] for s in seeds], axis=0)
                model = np.mean([by[(s, g)]["reduced"] for s in seeds], axis=0)
                red[f"{g:g}"] = {"full": full.round(5).tolist(), "reduced": model.round(5).tolist(),
                                 "full_rows": np.round([by[(s, g)]["memory"] for s in seeds], 4).tolist(),
                                 "reduced_rows": np.round([by[(s, g)]["reduced"] for s in seeds], 4).tolist()}
            rec["reduced"] = red
        out[fam] = rec
    return out


def spectrum_block():
    rows = jsonl(RES / "dark_projection" / "spectrum.jsonl")
    seed = sorted({r["seed"] for r in rows})[0]
    keep = {}
    for r in rows:
        if r["seed"] != seed:
            continue
        if (r["design"], r["gamma"]) in (("shared", 1.0), ("shared", 8.0), ("shared", 64.0), ("separate", 8.0)):
            keep[f"{r['design']}_{r['gamma']:g}"] = {"re": r["re"], "im": r["im"]}
    return {"seed": seed, "maps": keep}


def rates_block(seed):
    """Decay rates of the generator at the mean input along the loss, for the reservoir of the spectrum block."""
    rows = [r for r in jsonl(RES / "dark_projection" / "rates.jsonl") if r["seed"] == seed]
    out = {"seed": seed}
    for fam, keys in (("shared", ("slowest", "dark_top", "bright_bottom", "fastest")), ("separate", ("slowest",))):
        sel = sorted((r for r in rows if r["design"] == fam), key=lambda r: r["gamma"])
        out[fam] = {"gamma": [round(r["gamma"], 6) for r in sel], "protected": sel[0]["protected"],
                    **{k: [float(f"{r[k]:.6g}") for r in sel] for k in keys}}
    flow = RES / "dark_projection" / "flow.jsonl"           # every decay rate, shared channel, same reservoir
    if flow.exists():
        sel = sorted((r for r in jsonl(flow) if r["seed"] == seed), key=lambda r: r["gamma"])
        out["flow"] = {"gamma": [round(r["gamma"], 6) for r in sel],
                       "rates": [[float(f"{x:.4g}") for x in r["rates"]] for r in sel]}
    return out


def clock_block():
    """Run A: time to forget along the loss at four input amplitudes, local loss and opposite phases."""
    path = RES / "clock" / "clock.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    out = {}
    for fam in sorted({r["design"] for r in rows}):
        out[fam] = {}
        for amp in sorted({r["amplitude"] for r in rows}, reverse=True):
            sel = [r for r in rows if r["design"] == fam and r["amplitude"] == amp]
            gammas = sorted({r["gamma"] for r in sel})
            seeds = sorted({r["seed"] for r in sel})
            by = {(r["seed"], r["gamma"]): r["forget"] for r in sel}
            out[fam][f"{amp:g}"] = {"gamma": gammas,
                                    "forget": [[float(f"{by[(s, g)]:.6g}") if (s, g) in by else None for g in gammas] for s in seeds]}
    return out


def sampling_block():
    """Run B: memory from exact features and from finite preparations per setting and step."""
    path = RES / "clock" / "sampling.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    if (RES / "clock" / "line.jsonl").exists():
        rows += jsonl(RES / "clock" / "line.jsonl")
    keys = sorted({(r["design"], r["rule"], r["gamma"]) for r in rows})
    out = []
    for design, rule, gamma in keys:
        sel = sorted((r for r in rows if (r["design"], r["rule"], r["gamma"]) == (design, rule, gamma)),
                     key=lambda r: r["seed"])
        rec = {"design": design, "rule": rule, "gamma": gamma, "eta": sel[0]["eta"], "seeds": len(sel)}
        for k in ("exact", "var_mean", "shots_1000000", "shots_10000000", "shots_100000000"):
            rec[k] = [r[k] for r in sel]
        rec["curve"] = np.mean([r["curve"] for r in sel], axis=0).round(5).tolist()
        rec["curve_1e8"] = np.mean([r["curve_100000000"] for r in sel], axis=0).round(5).tolist()
        out.append(rec)
    return out


def budget_block():
    """Memory at preparations per setting and step from 10^5 to 10^8.5, with the range tau_90 of the exact recall."""
    path = RES / "clock" / "budget.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    out = []
    for key in sorted({(r["design"], r["rule"], r["gamma"]) for r in rows}):
        sel = sorted((r for r in rows if (r["design"], r["rule"], r["gamma"]) == key), key=lambda r: r["seed"])
        out.append({"design": key[0], "rule": key[1], "gamma": key[2], "eta": sel[0]["eta"], "seeds": len(sel),
                    "shots": sel[0]["shots"], "exact": [r["exact"] for r in sel], "tau90": [r["tau90"] for r in sel],
                    "memory": [[round(m, 4) for m in r["memory"]] for r in sel]})
    return out


def budget_low_block():
    """Memory at 10^3 to 10^4.5 preparations per setting and step along the same lines (own random stream)."""
    path = RES / "clock" / "budget_low.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    out = []
    for key in sorted({(r["design"], r["rule"], r["gamma"]) for r in rows}):
        sel = sorted((r for r in rows if (r["design"], r["rule"], r["gamma"]) == key), key=lambda r: r["seed"])
        out.append({"design": key[0], "rule": key[1], "gamma": key[2], "seeds": [r["seed"] for r in sel],
                    "shots": sel[0]["shots"], "memory": [[round(m, 4) for m in r["memory"]] for r in sel]})
    return out


def _points(rows, fields, extra=()):
    """Records grouped by (design, rule, gamma), seeds in order: the listed fields as lists over the seeds."""
    out = []
    for key in sorted({(r["design"], r["rule"], r["gamma"]) for r in rows}):
        sel = sorted((r for r in rows if (r["design"], r["rule"], r["gamma"]) == key), key=lambda r: r["seed"])
        rec = {"design": key[0], "rule": key[1], "gamma": key[2], "eta": sel[0]["eta"], "seeds": len(sel)}
        rec.update({f: sel[0][f] for f in extra})
        rec.update({f: [r[f] for r in sel] for f in fields})
        out.append(rec)
    return out


def replay_block():
    """Budget points at 10^9: memory from the state with the full history and from the replay of the last
    tau_90 inputs from |00000>, at the same preparations per setting and step."""
    path = RES / "clock" / "replay.jsonl"
    return _points(jsonl(path), ("tau90", "shots", "exact", "replay_exact", "full_state", "replay")) \
        if path.exists() else None


def washout_block():
    """Exact memory after the usual washout and after five times it."""
    path = RES / "clock" / "washout.jsonl"
    return _points(jsonl(path), ("exact", "exact_long"), extra=("washout",)) if path.exists() else None


def budget_replay_block():
    """Memory at budgets of 10^8, 10^9 and 10^10 preparations times replayed steps, features from the replay."""
    path = RES / "clock" / "budget_replay.jsonl"
    return _points(jsonl(path), ("tau90", "shots", "exact", "replay_exact", "memory"), extra=("budgets",)) \
        if path.exists() else None


def replay_shots_block():
    """Memory against preparations per setting and step, features from the replay, at the crossings of Sec. 3.4;
    input on all spins ("uniform") and on spin 1 only ("single")."""
    out = {}
    for label, name in (("uniform", "replay_shots"), ("single", "pattern_replay_shots")):
        path = RES / "clock" / f"{name}.jsonl"
        if path.exists():
            out[label] = _points(jsonl(path), ("tau90", "exact", "replay_exact", "signal_replay", "memory"),
                                 extra=("shots",))
    return out or None


def sizes_block():
    """Memory against the loss rate at three, four and six spins (run_sizes.py): capacity of every reservoir along
    each line, and the mean recall summed over the last twenty delays, which shows where the memory reaches past the
    delays summed."""
    path = RES / "sizes" / "memory.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    out = {}
    for n in sorted({r["spins"] for r in rows}):
        lines = []
        for design, rule in sorted({(r["design"], r["rule"]) for r in rows if r["spins"] == n}):
            sel = [r for r in rows if (r["spins"], r["design"], r["rule"]) == (n, design, rule)]
            gammas = sorted({r["gamma"] for r in sel})
            seeds = sorted({r["seed"] for r in sel})
            by = {(r["seed"], r["gamma"]): r for r in sel}
            lines.append({"design": design, "rule": rule, "gamma": gammas, "seeds": len(seeds),
                          "capacity": [[round(by[(s, g)]["capacity"], 4) for g in gammas] for s in seeds],
                          "tail": [round(float(np.mean([by[(s, g)]["tail"] for s in seeds])), 4) for g in gammas]})
        out[str(n)] = {"features": sel[0]["features"], "lines": lines}
    return out


def five_orders_block():
    """Order of the five-spin memory from exact moments (run_five_spin_orders.py), per design, loss rate and
    amplitude, one entry per reservoir."""
    path = RES / "five_spin_orders" / "orders.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    keys = ("seed", "mean_order", "single", "memory300", "memory", "tau90", "order_ratio", "t_mem", "s", "law_mean",
            "mu_settled")
    out = []
    for key in sorted({(r["design"], r["gamma"], r["eta"]) for r in rows}):
        sel = sorted((r for r in rows if (r["design"], r["gamma"], r["eta"]) == key), key=lambda r: r["seed"])
        entry = {"design": key[0], "gamma": key[1], "eta": key[2]}
        for k in keys:
            if all(k in r for r in sel):
                entry[k] = [r[k] for r in sel]
        entry["edges"] = [r["edges"] for r in sel]
        entry["binned"] = [[round(b, 6) for b in r["binned"]] for r in sel]
        out.append(entry)
    return out


def gentle_block():
    """The gentle-writing limit (run_gentle_limit.py): the limit on its grid of nu for every reservoir and channel,
    the recursion of the full five-spin reservoir at single values of nu, and the recursion of the strong-loss
    generator against its limit. `main` is the reservoir that was also run at the larger leak."""
    path = RES / "gentle_limit"
    if not all((path / f"{name}.jsonl").exists() for name in ("limit", "exact", "dark")):
        return None
    fields = ("seed", "design", "unknowns", "reach", "rank", "gap", "slope", "nu", "share1", "share2", "mean_order",
              "mixing", "signal", "decay", "share1_leading")
    limit = sorted(({k: r[k] for k in fields if k in r} for r in jsonl(path / "limit.jsonl")),
                   key=lambda r: (r["design"], r["seed"]))
    fields = ("seed", "design", "gamma", "eta", "nu", "share1", "share2", "mean_order", "rank", "signal")
    exact = sorted(({k: r[k] for k in fields} for r in jsonl(path / "exact.jsonl")),
                   key=lambda r: (r["design"], r["seed"], r["gamma"], r["nu"]))
    count = defaultdict(int)
    for r in exact:
        count[r["seed"]] += 1
    dark = sorted(jsonl(path / "dark.jsonl"), key=lambda r: (r["nu"], r["gamma"]))
    return {"main": max(count, key=count.get), "limit": limit, "exact": exact, "dark": dark}


def pattern_block():
    """The clock and the sampling test with the input on spin 1 only."""
    out = {}
    path = RES / "clock" / "pattern_clock.jsonl"
    if path.exists():
        rows = jsonl(path)
        out["clock"] = {}
        for fam in sorted({r["design"] for r in rows}):
            out["clock"][fam] = {}
            for amp in sorted({r["amplitude"] for r in rows}, reverse=True):
                sel = [r for r in rows if r["design"] == fam and r["amplitude"] == amp]
                gammas = sorted({r["gamma"] for r in sel})
                seeds = sorted({r["seed"] for r in sel})
                by = {(r["seed"], r["gamma"]): r["forget"] for r in sel}
                out["clock"][fam][f"{amp:g}"] = {"gamma": gammas, "forget": [[by.get((s, g)) for g in gammas]
                                                                             for s in seeds]}
    path = RES / "clock" / "pattern_sampling.jsonl"
    if path.exists():
        rows = jsonl(path)
        out["sampling"] = [{"design": d, "rule": r_, "gamma": g,
                            **{k: [r[k] for r in rows if (r["design"], r["rule"], r["gamma"]) == (d, r_, g)]
                               for k in ("exact", "shots_1000000", "var_mean")}}
                           for d, r_, g in sorted({(r["design"], r["rule"], r["gamma"]) for r in rows})]
    return out or None


def polarization_block():
    """Run C: all eight shapes at a matched clock and input amplitude."""
    path = RES / "clock" / "polarization.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    out = {}
    for fam in FAMILIES:
        sel = sorted((r for r in rows if r["design"] == fam), key=lambda r: r["seed"])
        if sel:
            out[fam] = {k: [r[k] for r in sel] for k in ("gamma", "eta", "write_slow", "write_all", "polarization",
                                                           "var_mean", "exact", "shots_1000000", "leak_time")}
    return out


def prefactor_block():
    """Three spins at gamma = 4096: share of single past inputs and signal size against the prefactor."""
    path = RES / "spin_orders" / "prefactor.jsonl"
    if not path.exists():
        return None
    rows = jsonl(path)
    seeds = [8675309, 424242, 97531, 123457]          # H1-H4, as in experiments/run_spin_orders.py
    out = {"order": seeds}
    for s in seeds:
        sel = sorted((r for r in rows if r["seed"] == s), key=lambda r: r["prefactor"])
        out[str(s)] = {k: [r[k] for r in sel] for k in ("prefactor", "single", "size")}
    return out


def removal_block():
    rows = jsonl(RES / "dark_projection" / "removal.jsonl")
    out = {}
    for g in sorted({r["gamma"] for r in rows}):
        sel = [r for r in rows if r["gamma"] == g]
        out[f"{g:g}"] = {k: [float(np.sum(r[k][10:])) for r in sel] for k in ("intact", "rotated", "removed")}
        out[f"{g:g}"]["curves"] = {k: [list(np.round(r[k], 5)) for r in sel] for k in ("intact", "rotated", "removed")}
    return out


def write_block():
    rows = jsonl(RES / "dark_projection" / "write.jsonl")
    gammas = sorted({r["gamma"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["gamma"]): r for r in rows}
    return {"gamma": gammas,
            "along": [[by[(s, g)]["along_dark"] for g in gammas] for s in seeds],
            "across": [[by[(s, g)]["across_dark"] for g in gammas] for s in seeds]}


def spin_block():
    rows = jsonl(RES / "spin_orders" / "orders.jsonl")
    extra_path = RES / "spin_orders" / "orders_extra.jsonl"
    extra = jsonl(extra_path) if extra_path.exists() else []
    out = {}
    for drive in ("strong", "gentle"):
        sel = [r for r in rows if r["drive"] == drive]
        gammas = sorted({r["gamma"] for r in sel})
        seeds = sorted({r["seed"] for r in sel})
        by = {(r["seed"], r["gamma"]): r for r in sel}
        more = [r for r in extra if r["drive"] == drive]
        line_by = dict(by)
        line_by.update({(r["seed"], r["gamma"]): r for r in more})
        line_gammas = sorted({g for _, g in line_by})
        rec = {"gamma": gammas, "line_gamma": line_gammas, "seeds": seeds}
        for key in ("size", "single", "mean_order", "range"):
            rec[key] = [[line_by[(s, g)][key] for g in line_gammas] for s in seeds]
        rec["heat_mean_order"] = [[by[(s, g)]["mean_order"] for g in gammas] for s in seeds]
        # order shares of the first reservoir on a common logarithmic grid of orders
        grid = np.unique(np.round(np.geomspace(1, 4096, 49)).astype(int))
        heat = []
        for g in gammas:
            r = by[(seeds[0], g)]
            edges, binned = np.array(r["edges"]), np.array(r["binned"])
            per_order = np.zeros(int(edges[-1]))
            for e0, e1, value in zip(edges[:-1], edges[1:], binned):
                per_order[e0 - 1:e1 - 1] = value / (e1 - e0)
            cum = np.r_[0.0, np.cumsum(per_order)]
            # share in [grid_i, grid_{i+1}) divided by the log-width of the bin
            lo = np.minimum(grid[:-1] - 1, len(per_order))
            hi = np.minimum(grid[1:] - 1, len(per_order))
            share = cum[hi] - cum[lo]
            heat.append((share / np.log(grid[1:] / grid[:-1])).tolist())
        rec["order_grid"] = grid.tolist()
        rec["order_density"] = heat
        out[drive] = rec
    coh = json.loads((RES / "spin_orders" / "coherence.json").read_text())["rows"]
    gammas = sorted({r["gamma"] for r in coh})
    seeds = sorted({r["seed"] for r in coh})
    by = {(r["seed"], r["gamma"]): r for r in coh}
    out["coherence"] = {"gamma": gammas, "seeds": seeds,
                        "kept": [[by[(s, g)]["recovered"]["signal_variance"] for g in gammas] for s in seeds],
                        "erased": [[by[(s, g)]["dephased"]["signal_variance"] for g in gammas] for s in seeds]}
    return out


def law_block():
    base = RES / "order_law"
    out = {"one": {}, "beta": {}, "families": []}
    rates = json.loads((base / "order_results.json").read_text())
    with np.load(base / "family_22001.npz") as fam:
        for p in (0.03, 0.01, 0.003):
            out["one"][f"{p:g}"] = fam[f"{p}_basis_scalar_survival"][:400].round(6).tolist()
            out["beta"][f"{p:g}"] = [r["beta"] for r in rates if r["seed"] == 22001 and r["reset"] == "basis"
                                     and r["readout"] == "scalar" and abs(r["p"] - p) < 1e-12][0]
    for f in sorted(base.glob("family_*.npz")):
        with np.load(f) as fam:
            for reset in ("basis", "random"):
                for readout in ("full", "scalar"):
                    key = f"0.001_{reset}_{readout}"
                    x, y = fam[key + "_x"], fam[key + "_survival"]
                    sel = x <= 4.2
                    out["families"].append({"x": x[sel][::3].round(5).tolist(), "y": y[sel][::3].round(6).tolist()})
    return out


def mean_rate(r, n):
    return float(np.mean(np.array(r["rates"])[:n]))


def review_block(kernel):
    """Runs added for the reviews: scaled input, slow modes up to large loss, local dephasing, signal size."""
    base = RES / "review_runs"
    out = {}
    if (base / "scaled.jsonl").exists():
        rows = jsonl(base / "scaled.jsonl")
        out["scaled"] = {}
        for gc in sorted({r["gamma_c"] for r in rows}):
            sel = [r for r in rows if r["gamma_c"] == gc]
            gammas = sorted({r["gamma"] for r in sel})
            seeds = sorted({r["seed"] for r in sel})
            by = {(r["seed"], r["gamma"]): r for r in sel}
            get = lambda f: [[f(by[(s, g)]) if (s, g) in by else None for g in gammas] for s in seeds]
            out["scaled"][f"{gc:g}"] = {
                "gamma": gammas, "eta": [by[(seeds[0], g)]["eta"] for g in gammas],
                "recall": get(lambda r: float(np.sum(r["memory"]))),
                "forget": get(lambda r: 1.0 / (DT * mean_rate(r, 99))),
                "var_mean": get(lambda r: r["feature_var_mean"]), "var_max": get(lambda r: r["feature_var_max"])}
    if (base / "spectrum.jsonl").exists():
        rows = jsonl(base / "spectrum.jsonl")
        out["far"] = {}
        for fam in FAMILIES:
            sel = [r for r in rows if r["design"] == fam]
            if not sel:
                continue
            slow = max(kernel[fam] - 1, 1)
            gammas = sorted({r["gamma"] for r in sel})
            seeds = sorted({r["seed"] for r in sel})
            by = {(r["seed"], r["gamma"]): r for r in sel}
            out["far"][fam] = {"gamma": gammas,
                               "forget": [[1.0 / (DT * mean_rate(by[(s, g)], slow)) if (s, g) in by else None
                                           for g in gammas] for s in seeds],
                               "gamma_rate": [[g * mean_rate(by[(s, g)], slow) if (s, g) in by else None
                                               for g in gammas] for s in seeds]}
    if (base / "dephasing.jsonl").exists():
        rows = jsonl(base / "dephasing.jsonl")
        out["dephasing"] = [{"seed": r["seed"], "gamma": r["gamma"], "dephasing": r["dephasing"],
                             "rate": mean_rate(r, 99)} for r in rows]
    if (base / "variance.jsonl").exists():
        rows = list({(r["seed"], r["design"], r["gamma"]): r for r in jsonl(base / "variance.jsonl")}.values())
        out["variance"] = {}
        for fam in ("shared", "signed"):
            sel = [r for r in rows if r["design"] == fam]
            gammas = sorted({r["gamma"] for r in sel})
            out["variance"][fam] = {"gamma": gammas,
                                    "var_mean": [float(np.mean([r["feature_var_mean"] for r in sel if r["gamma"] == g]))
                                                 for g in gammas],
                                    "var_max": [float(np.mean([r["feature_var_max"] for r in sel if r["gamma"] == g]))
                                                for g in gammas]}
    return out


def v1_block():
    """Finite-sampling test of the first version: memory minus local loss against preparations per step."""
    agg = json.loads((RES / "measurement_full_v3" / "measurement_full_aggregate.json").read_text())
    names = {"B3_collective": "shared", "B5_pair": "pair", "B4_loss_exchange": "exchange", "B2_thermal": "gainloss"}
    out = {}
    for model in ("independent", "grouped"):
        rec = {}
        for channel, fam in names.items():
            sel = [r for r in agg["paired_vs_local"] if r["measurement_model"] == model and r["channel"] == channel]
            finite = sorted((r for r in sel if not r["is_exact"]), key=lambda r: r["total_shots_per_time_step"])
            exact = [r for r in sel if r["is_exact"]][0]
            rec[fam] = {"budget": [r["total_shots_per_time_step"] for r in finite],
                        "mean": [r["mean"] for r in finite], "lo": [r["ci95_low"] for r in finite],
                        "hi": [r["ci95_high"] for r in finite],
                        "exact": [exact["mean"], exact["ci95_low"], exact["ci95_high"]]}
        out[model] = rec
    return out


def main():
    kernel = {r["design"]: r["kernel"] for r in jsonl(RES / "jump_families" / "kernel.jsonl")}
    spectrum = spectrum_block()
    snap = {"kernel": kernel, "memory": memory_block(kernel), "spectrum": spectrum,
            "rates": rates_block(spectrum["seed"]), "removal": removal_block(), "write": write_block(),
            "spin": spin_block(), "law": law_block(), "v1": v1_block(), "review": review_block(kernel)}
    for key, block in (("clock", clock_block()), ("sampling", sampling_block()), ("budget", budget_block()),
                       ("pattern", pattern_block()), ("polarization", polarization_block()),
                       ("prefactor", prefactor_block()), ("budget_low", budget_low_block()),
                       ("five_orders", five_orders_block()), ("replay", replay_block()),
                       ("washout", washout_block()), ("budget_replay", budget_replay_block()),
                       ("replay_shots", replay_shots_block()), ("sizes", sizes_block()),
                       ("gentle_limit", gentle_block())):
        if block is not None:
            snap[key] = block
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, separators=(",", ":")) + "\n")
    done = {f: snap["memory"][f]["complete"] for f in snap["memory"]}
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB); complete: {done}")


if __name__ == "__main__":
    main()
