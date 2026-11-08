import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import control as c
import pscore, psim, run_dark_projection as rdp, run_final_scaling as rfs

def features(seed, name, gamma, rule):
    eta = c.eta_rule(rule, gamma)
    wash = rdp.washout(gamma)
    J, inputs = psim.couplings_and_inputs(seed, wash + rdp.TRAIN + rdp.VALIDATION + rdp.TEST)
    H0, H1 = rdp.hamiltonian(J, eta * np.ones(psim.N))
    rows, _ = rdp.drive_steady(H0 + gamma * rdp.dissipator(name), H1, inputs, wash)
    return rows, inputs[wash:]

seed = rfs.deterministic_seeds(8)[0]
for name, gamma, rule in (("shared", 64.0, "fixed"), ("signed", 64.0, "gentle_strong"), ("separate", 0.03125, "gentle_weak")):
    X, post = features(seed, name, gamma, rule)
    Xc = X - X[:rdp.TRAIN].mean(axis=0)
    w, V = np.linalg.eigh(np.cov(Xc[:rdp.TRAIN].T))
    order = np.argsort(w)[::-1]; w, V = w[order], V[:, order]
    P = Xc @ V
    line = []
    for k in (5, 10, 20, 30, 45):
        caps, _ = pscore.selected(P[:, :k], post, tuple(range(1, 301)), tr=rdp.TRAIN, va=rdp.VALIDATION, te=rdp.TEST, ridges=(0.0,))
        line.append(f"top{k}: short {caps[:10].sum():5.2f} long {caps[10:].sum():5.2f}")
    print(f"{name} g={gamma} {rule}: variance of PC1/PC10/PC20/PC45 = {w[0]:.1e}/{w[9]:.1e}/{w[19]:.1e}/{w[44]:.1e}")
    print("   " + " | ".join(line), flush=True)
