import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import json
import numpy as np
from multiprocessing import get_context
import control as c
import pscore, psim, run_dark_projection as rdp, run_final_scaling as rfs

TR, VA, TE = 16000, 2000, 4000

def case(task):
    seed, name, gamma, rule = task
    eta = c.eta_rule(rule, gamma)
    wash = rdp.washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + TR + VA + TE)
    H0, H1 = rdp.hamiltonian(J, eta * np.ones(psim.N))
    p = psim.Propagator(H0 + gamma * rdp.dissipator(name), H1, nodes=rdp.NODES)
    r = psim.initial_vector()
    strings = np.empty((len(inputs) - wash, 3, 32))
    for k, value in enumerate(inputs):
        r = p.step(r, float(value))
        if k >= wash:
            strings[k - wash] = r[c.AXES]
    post = inputs[wash:]
    P = np.clip(np.einsum("tam,bm->tab", strings, psim.HAD) / 2 ** psim.N, 0, None)
    P /= P.sum(axis=2, keepdims=True)
    out = {"design": name, "gamma": gamma, "rule": rule}
    caps, _ = pscore.selected(pscore.features_from_probs(P), post, tuple(range(1, 301)), tr=TR, va=VA, te=TE, ridges=(0.0,))
    out["exact"] = float(caps.sum())
    rng = np.random.default_rng(seed + 202)
    for n in (10 ** 6, 10 ** 8):
        caps, _ = pscore.selected(pscore.sampled_features(P, n, rng), post, tuple(range(1, 301)), tr=TR, va=VA, te=TE)
        out[f"shots_{n}"] = float(caps.sum())
    return out

if __name__ == "__main__":
    seed = rfs.deterministic_seeds(8)[0]
    tasks = [(seed, "shared", 64.0, "fixed"), (seed, "signed", 64.0, "gentle_strong"), (seed, "separate", 0.03125, "gentle_weak")]
    with get_context("spawn").Pool(3) as pool:
        for rec in pool.imap_unordered(case, tasks):
            print(json.dumps(rec), flush=True)
