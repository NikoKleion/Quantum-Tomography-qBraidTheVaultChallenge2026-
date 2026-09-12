"""Variational state tomography for HEA vaults: fit an ansatz V to probe data, attack
with V^dagger. Also the numpy statevector and probe helpers (qubit i = bit i).

Usage: python engine/hea_tomography.py   # runs on the mock HEA vaults
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit

from toolkit import counts_to_bits


# ---------------------------------------------------------------------------
# fast vectorized statevector engine (n <= ~12).  index k: qubit i = bit i.
# ---------------------------------------------------------------------------
def _apply_1q(psi, q, U):
    k = np.arange(psi.size)
    k0 = k & ~(1 << q)
    k1 = k0 | (1 << q)
    a0, a1 = psi[k0], psi[k1]
    qbit = (k >> q) & 1
    return np.where(qbit == 0, U[0, 0] * a0 + U[0, 1] * a1,
                    U[1, 0] * a0 + U[1, 1] * a1)


def _apply_cx(psi, c, t):
    k = np.arange(psi.size)
    ctrl = ((k >> c) & 1).astype(bool)
    src = k.copy()
    src[ctrl] = k[ctrl] ^ (1 << t)
    return psi[src]


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(p):
    return np.array([[np.exp(-1j * p / 2), 0], [0, np.exp(1j * p / 2)]], dtype=complex)


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)


def fast_state(params, n, layers):
    """|V(params)> as a flat statevector, same layout as ansatz()."""
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    p = iter(params)
    for _ in range(layers):
        for q in range(n):
            psi = _apply_1q(psi, q, _ry(next(p)))
            psi = _apply_1q(psi, q, _rz(next(p)))
        for q in range(n - 1):
            psi = _apply_cx(psi, q, q + 1)
    for q in range(n):
        psi = _apply_1q(psi, q, _ry(next(p)))
        psi = _apply_1q(psi, q, _rz(next(p)))
    return psi


def rotate_probs(psi, basis):
    """Probabilities of psi measured in a Pauli basis string, one letter per qubit."""
    out = psi
    for q, b in enumerate(basis):
        if b == "X":
            out = _apply_1q(out, q, _H)
        elif b == "Y":
            out = _apply_1q(_apply_1q(out, q, _SDG), q, _H)
    return np.abs(out) ** 2


# ---------------------------------------------------------------------------
# probe data collection
# ---------------------------------------------------------------------------
def measurement_bases(n: int, num_random: int, seed: int = 0):
    """Z, X, Y on all qubits, then num_random random Pauli product bases."""
    bases = ["Z" * n, "X" * n, "Y" * n]
    rng = np.random.default_rng(seed)
    while len(bases) < 3 + num_random:
        bases.append("".join(rng.choice(list("XYZ"), size=n)))
    return bases


def _per_qubit_basis_circuit(basis: str) -> QuantumCircuit:
    n = len(basis)
    qc = QuantumCircuit(n)
    for q, b in enumerate(basis):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
    return qc


def observed_prob(counts: dict, n: int) -> np.ndarray:
    """Normalized probability vector p[k], k = sum_i qubit_i * 2^i."""
    bits, w = counts_to_bits(counts, n)          # bits[:,i] = qubit i
    p = np.zeros(1 << n)
    idx = (bits * (1 << np.arange(n))).sum(axis=1)
    for a, weight in zip(idx, w):
        p[a] += weight
    s = p.sum()
    return p / s if s else p


def collect(vault, bases, shots=200):
    """One probe per basis. Returns list of (basis_string, observed_prob)."""
    n = vault.num_qubits
    data = []
    for b in bases:
        counts = vault.probe(_per_qubit_basis_circuit(b), shots=shots)
        data.append((b, observed_prob(counts, n)))
    return data


# ---------------------------------------------------------------------------
# ansatz + fitting
# ---------------------------------------------------------------------------
def ansatz(params: np.ndarray, n: int, layers: int) -> QuantumCircuit:
    """RY/RZ layers alternating with CX chains, then a final RY/RZ layer."""
    qc = QuantumCircuit(n)
    p = iter(params)
    for _ in range(layers):
        for q in range(n):
            qc.ry(next(p), q)
            qc.rz(next(p), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):                            # final single qubit layer
        qc.ry(next(p), q)
        qc.rz(next(p), q)
    return qc


def num_params(n: int, layers: int) -> int:
    return 2 * n * (layers + 1)


def warm_start(prev: np.ndarray, n: int, prev_layers: int, rng) -> np.ndarray:
    """Init for L+1 layers: insert a small random block before the final layer."""
    blk = 2 * n
    body = prev[:prev_layers * blk]
    final = prev[prev_layers * blk:]
    new_block = rng.uniform(-0.3, 0.3, blk)
    return np.concatenate([body, new_block, final])


def fit_ansatz(data, n: int, layers: int, restarts: int = 3, seed: int = 0,
               maxiter: int = 600, warm=None):
    """L2 fit of V(phi)|0> to the probe distributions. Returns (params, resid)."""
    npar = num_params(n, layers)
    rng = np.random.default_rng(seed)

    def loss(params):
        psi = fast_state(params, n, layers)
        tot = 0.0
        for basis, p_obs in data:
            tot += np.sum((rotate_probs(psi, basis) - p_obs) ** 2)
        return tot

    inits = ([warm] if warm is not None else []) + \
            [rng.uniform(0, 2 * np.pi, npar) for _ in range(restarts)]
    best, best_loss = None, np.inf
    for x0 in inits:
        res = minimize(loss, x0, method="Powell",
                       options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-4})
        if res.fun < best_loss:
            best, best_loss = res.x, res.fun
    return best, best_loss


def disentangler(params: np.ndarray, n: int, layers: int) -> QuantumCircuit:
    """A = V(phi)^dagger : maps the fitted state back to |0>."""
    return ansatz(params, n, layers).inverse()


# ---------------------------------------------------------------------------
# full HEA crack: fit increasing depth, return Pareto candidates
# ---------------------------------------------------------------------------
def hea_tomography_candidates(vault, max_layers=2, num_random=9, seed=0,
                              reuse_data=None):
    """Fit L=0..max_layers, warm started. reuse_data: [(basis, prob)] already probed."""
    n = vault.num_qubits
    if reuse_data:
        have = {b for b, _ in reuse_data}
        extra = [b for b in measurement_bases(n, num_random, seed) if b not in have]
        data = list(reuse_data) + collect(vault, extra[:num_random])
    else:
        data = collect(vault, measurement_bases(n, num_random, seed))
    rng = np.random.default_rng(seed + 7)
    cands, prev = [], None
    for L in range(0, max_layers + 1):
        restarts = 3 if L == 0 else 1             # deeper layers are warm started
        warm = warm_start(prev, n, L - 1, rng) if prev is not None else None
        params, resid = fit_ansatz(data, n, L, restarts=restarts, seed=seed,
                                   maxiter=500, warm=warm)
        cands.append((f"tomo_L{L}", disentangler(params, n, L), resid))
        prev = params
    return cands


if __name__ == "__main__":
    from mock_platform import build_challenge, count_2q
    print("HEA variational tomography vs. hidden vaults:")
    for v in build_challenge():
        if v.family != "hea":
            continue
        cands = hea_tomography_candidates(v, max_layers=2, num_random=9)
        line = [f"vault {v.index} n={v.num_qubits} c={v._c} probes={v.probes_used}"]
        best = None
        for label, A, resid in cands:
            if v.attacks_used >= v.MAX_ATTACKS:
                break
            res = v.attack(A)
            d = count_2q(A)
            line.append(f"{label}(d={d} R={res['raw']:.2f} s={res['score']:.2f})")
            if best is None or res["score"] > best:
                best = res["score"]
        line.append(f"-> best {best:.3f}")
        print("  " + " ".join(line))
