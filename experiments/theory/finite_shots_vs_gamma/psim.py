"""Independent simulator for the 5-qubit reservoir.  Shares no code with the repo.

State: the 1024 real numbers r_P = Tr(P rho), P a Pauli string.
Generator: real matrix G with dr/dt = G r, built by multiplying Pauli strings
symbolically (no density matrices, no kron, no Liouvillian in the matrix basis).
Propagator: dense expm(G(s) dt) at Chebyshev nodes in s, interpolated in s
(the map s -> exp(G(s) dt) is entire, so this converges fast).
The repo instead uses complex sparse Liouvillians and expm_multiply on vec(rho).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.linalg as sla

N = 5
MASK = 2 ** N - 1
DIM = 4 ** N
POP = np.array([bin(i).count("1") for i in range(2 ** N)], dtype=np.int64)
IK = np.array([1, 1j, -1, -1j])
ALLX = np.arange(DIM) >> N
ALLZ = np.arange(DIM) & MASK
H_FIELD = 0.5
DT = 0.5


def pid(x, z):
    return (np.asarray(x) << N) | np.asarray(z)


def pmul(x1, z1, x2, z2):
    """P1 P2 = i^k P3 for Hermitian Paulis P(x,z) = i^{|x&z|} X^x Z^z."""
    x3, z3 = x1 ^ x2, z1 ^ z2
    k = (POP[x1 & z1] + POP[x2 & z2] + 2 * POP[z1 & x2] - POP[x3 & z3]) % 4
    return k, x3, z3


def add(G, x, z, coef):
    np.add.at(G, (pid(x, z), np.arange(DIM)), coef)


def comm_gen(xa, za):
    """Matrix of Q -> -i [P_a, Q] in the Pauli basis (real)."""
    G = np.zeros((DIM, DIM), dtype=complex)
    k1, x1, z1 = pmul(xa, za, ALLX, ALLZ)
    k2, x2, z2 = pmul(ALLX, ALLZ, xa, za)
    add(G, x1, z1, -1j * IK[k1])
    add(G, x2, z2, +1j * IK[k2])
    assert abs(G.imag).max() < 1e-13
    return G.real


def dissipator_gen(terms):
    """Matrix of Q -> L Q L^+ - {L^+ L, Q}/2 for L = sum l_a P_a, terms = [(x,z,l)]."""
    G = np.zeros((DIM, DIM), dtype=complex)
    for xa, za, la in terms:
        for xb, zb, lb in terms:
            c = la * np.conj(lb)
            # P_a Q P_b
            k1, x1, z1 = pmul(xa, za, ALLX, ALLZ)
            k2, x2, z2 = pmul(x1, z1, xb, zb)
            add(G, x2, z2, c * IK[(k1 + k2) % 4])
            # P_b P_a = i^k P_c
            k, xc, zc = pmul(np.int64(xb), np.int64(zb), np.int64(xa), np.int64(za))
            k3, x3, z3 = pmul(xc, zc, ALLX, ALLZ)
            add(G, x3, z3, -0.5 * c * IK[(k + k3) % 4])
            k4, x4, z4 = pmul(ALLX, ALLZ, xc, zc)
            add(G, x4, z4, -0.5 * c * IK[(k + k4) % 4])
    assert abs(G.imag).max() < 1e-13
    return G.real


def bit(site):
    """Internal map: site i <-> bit i."""
    return 1 << site


def lower_terms(coeffs):
    """L = sum_i c_i sigma^-_i with sigma^- = |0><1| = (X + iY)/2, Y = i X Z here."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        terms.append((bit(i), 0, 0.5 * c))             # X_i
        terms.append((bit(i), bit(i), 0.5j * c))       # Y_i
    return terms


_CACHE = {}


def blocks():
    if "xx" in _CACHE:
        return _CACHE
    xx = {}
    for i in range(N):
        for j in range(i + 1, N):
            xx[(i, j)] = comm_gen(np.int64(bit(i) | bit(j)), np.int64(0))
    gz = sum(comm_gen(np.int64(0), np.int64(bit(i))) for i in range(N))
    gx = sum(comm_gen(np.int64(bit(i)), np.int64(0)) for i in range(N))
    shared = dissipator_gen(lower_terms([1.0] * N))
    separate = sum(dissipator_gen(lower_terms([1.0 if k == i else 0.0 for k in range(N)]))
                   for i in range(N))
    signed = dissipator_gen(lower_terms([1.0, 1.0, 1.0, -1.0, -1.0]))
    _CACHE.update(xx=xx, gz=gz, gx=gx, shared=shared, separate=separate, signed=signed)
    return _CACHE


def couplings_and_inputs(seed, n_steps):
    """Same random numbers as the study: 10 couplings, then the input stream."""
    rng = np.random.default_rng(seed)
    J = {}
    for i in range(N):
        for j in range(i + 1, N):
            J[(i, j)] = rng.uniform(-1.0, 1.0)
    s = rng.uniform(0.0, 1.0, size=n_steps)
    return J, s


def generators(design, gamma, J, drive=1.0, h=H_FIELD):
    b = blocks()
    G0 = gamma * b[design] + h * b["gz"] + h * b["gx"]
    for key, value in J.items():
        G0 = G0 + value * b["xx"][key]
    G1 = drive * h * b["gx"]
    return G0, G1


class Propagator:
    """r -> exp((G0 + s G1) dt) r for s in [lo, hi], by Chebyshev interpolation in s."""

    def __init__(self, G0, G1, dt=DT, nodes=14, lo=0.0, hi=1.0):
        self.K = nodes
        self.lo, self.hi = lo, hi
        theta = np.pi * (2 * np.arange(nodes) + 1) / (2 * nodes)
        self.theta = theta
        self.x = np.cos(theta)
        sn = 0.5 * (hi + lo) + 0.5 * (hi - lo) * self.x
        self.P = np.stack([sla.expm((G0 + v * G1) * dt) for v in sn])
        # barycentric weights for Chebyshev points of the first kind
        self.w = (-1.0) ** np.arange(nodes) * np.sin(theta)

    def weights(self, s):
        x = (2 * s - self.hi - self.lo) / (self.hi - self.lo)
        d = x - self.x
        hit = np.abs(d) < 1e-14
        if hit.any():
            out = np.zeros(self.K)
            out[np.argmax(hit)] = 1.0
            return out
        q = self.w / d
        return q / q.sum()

    def step(self, r, s):
        return self.weights(s) @ (self.P @ r)

    def dense(self, s):
        return np.tensordot(self.weights(s), self.P, axes=(0, 0))


def initial_vector():
    """|00000><00000|: r_P = 1 on all strings of I and Z, 0 otherwise."""
    r = np.zeros(DIM)
    r[pid(np.zeros(2 ** N, dtype=np.int64), np.arange(2 ** N))] = 1.0
    return r


def mixed_vector():
    r = np.zeros(DIM)
    r[0] = 1.0
    return r


# ---- read-out indices -------------------------------------------------------

def single_axis_string(axis, subset_mask):
    """Index of the Pauli string with `axis` on the sites in subset_mask."""
    if axis == "x":
        return int(pid(subset_mask, 0))
    if axis == "y":
        return int(pid(subset_mask, subset_mask))
    return int(pid(0, subset_mask))


def feature_index():
    """The 45 features in the study's order: X_i, Y_i, Z_i, then X_iX_j, Y_iY_j, Z_iZ_j."""
    out = []
    for a in "xyz":
        for i in range(N):
            out.append(single_axis_string(a, bit(i)))
    for a in "xyz":
        for i in range(N):
            for j in range(i + 1, N):
                out.append(single_axis_string(a, bit(i) | bit(j)))
    return np.array(out)


FEAT = feature_index()
# all same-axis strings, one block per axis: index [axis, subset]
WALSH = np.array([[single_axis_string(a, m) for m in range(2 ** N)] for a in "xyz"])
# Walsh matrix: row bitstring b (bit i = site i), column subset m, entry (-1)^{b.m}
HAD = np.array([[(-1) ** int(POP[b & m]) for m in range(2 ** N)] for b in range(2 ** N)], dtype=float)


def probabilities(r_rows):
    """Outcome probabilities (rows, 3, 32) from Pauli components; bit i = site i, bit 0 = +1."""
    out = np.empty((r_rows.shape[0], 3, 2 ** N))
    for a in range(3):
        out[:, a, :] = r_rows[:, WALSH[a]] @ HAD.T / 2 ** N
    return out


def run(design, gamma, seed, n_steps, wash, nodes=14, every=None, drive=1.0, r0=None,
        keep_all=True, h=H_FIELD, J=None, s=None, prop=None):
    """Return rows of the full Pauli vector after step index >= wash, and the inputs."""
    if J is None:
        J, s = couplings_and_inputs(seed, n_steps)
    G0, G1 = generators(design, gamma, J, drive=drive, h=h)
    p = prop or Propagator(G0, G1, nodes=nodes)
    r = initial_vector() if r0 is None else r0.copy()
    rows = np.empty((n_steps - wash, DIM))
    for k in range(n_steps):
        r = p.step(r, s[k])
        if k >= wash:
            rows[k - wash] = r
    return rows, s[wash:], p
