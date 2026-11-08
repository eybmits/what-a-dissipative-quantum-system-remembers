import os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, scipy.linalg as la
from multiprocessing import get_context
import run_spin_orders as so

def point(task):
    seed, gamma, c = task
    model = so.SpinModel(seed)
    (Am, bm), (Ap, bp) = model.maps(gamma, c / np.sqrt(gamma))
    A, D = (Ap + Am) / 2, (Ap - Am) / 2
    mean = la.solve(np.eye(len(A)) - A, (bp + bm) / 2)
    v = D @ mean + (bp - bm) / 2
    stein = so.Stein(A)
    sigma = so.solve_full(np.outer(v, v), A, D, stein)
    a = model.x1
    size = float(a @ sigma @ a)
    G = stein.solve(np.outer(v, v))
    single = float(a @ (A @ G @ A.T) @ a) / size
    return seed, c, size, single

if __name__ == "__main__":
    gamma = 4096.0
    cs = (0.5, 1, 2, 4, 8, 16, 32, 64)
    tasks = [(s, gamma, c) for s in so.SEEDS for c in cs]
    with get_context("spawn").Pool(4) as pool:
        res = pool.map(point, tasks)
    for k, s in enumerate(so.SEEDS):
        rows = [r for r in res if r[0] == s]
        smax = max(r[2] for r in rows)
        print(f"H{k+1}: " + "  ".join(f"c={r[1]:g}: share {r[3]:.2f}, size {r[2]/smax:.2f}" for r in rows))
