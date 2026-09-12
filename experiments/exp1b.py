"""Tomography fidelity vs number of probe bases (200 shots each) for two HEA
targets, plus the best score reached from 6 and 10 bases.

Usage: python experiments/exp1b.py <n>
"""
import sys, time
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from mock_platform import make_hea_vault, count_2q
from state_tomography import raw_tomography
from hea_tomography import measurement_bases, rotate_probs
from fine_pareto import _fit_to_state, fine_pareto
from crack import minimize_delta

def true_R(psi_true, A):
    return float(abs(Statevector(psi_true).evolve(A).data[0]) ** 2)

def hea_targets(n):
    U1 = make_hea_vault(9, n, seed=1000 + n, layers=4)
    rng = np.random.default_rng(500 + n)
    U2 = QuantumCircuit(n)
    for L in range(2):
        for q in range(n):
            U2.ry(rng.uniform(0, 2*np.pi), q); U2.rz(rng.uniform(0, 2*np.pi), q)
        for q in range(n - 1):
            U2.cx(q, q + 1)
    for q in range(n):
        U2.ry(rng.uniform(0, 2*np.pi), q); U2.rz(rng.uniform(0, 2*np.pi), q)
    return [("mockHEA", U1), ("hardHEA", U2)]

def sampled_data(psi_true, n, n_bases, seed, shots=200):
    bases = measurement_bases(n, max(0, n_bases - 3), seed=seed)[:n_bases]
    rng = np.random.default_rng(seed + 77)
    out = []
    for b in bases:
        p = rotate_probs(psi_true, b); p = np.clip(p.real, 0, None); p /= p.sum()
        out.append((b, rng.multinomial(shots, p) / shots))
    return out

def score_from_recon(psi_est, psi_true, n, c_true):
    best = (0.0, 0.0, 0)
    rng = np.random.default_rng(5)
    L = 2
    x0 = rng.uniform(0, 2*np.pi, 2*n*(L+1))
    params, ov = _fit_to_state(psi_est, n, L, [True]*(L*(n-1)), x0, maxiter=300)
    for d, fid, A in fine_pareto(psi_est, params, n, L):
        A2 = minimize_delta(A); d2 = count_2q(A2)
        R = true_R(psi_true, A2)
        s = R * 4*c_true/(4*c_true + d2)
        if s > best[0]: best = (s, R, d2)
    return best

n = int(sys.argv[1])
for name, U in hea_targets(n):
    psi_true = Statevector(U).data; c_true = count_2q(U)
    fids = {}
    for nb in (4, 6, 8, 10, 14):
        data = sampled_data(psi_true, n, nb, seed=nb*7 + n)
        psi_est, resid = raw_tomography(data, n, restarts=3, seed=2)
        fids[nb] = (abs(np.vdot(psi_true, psi_est))**2, psi_est)
    line = f"{name} n={n} c={c_true} fid: " + " ".join(
        f"{nb}b={fids[nb][0]:.3f}" for nb in (4, 6, 8, 10, 14))
    s6 = score_from_recon(fids[6][1], psi_true, n, c_true)
    s12 = score_from_recon(fids[10][1], psi_true, n, c_true)
    line += f" | score@6b={s6[0]:.3f}(R{s6[1]:.2f},d{s6[2]}) score@10b={s12[0]:.3f}(R{s12[1]:.2f},d{s12[2]})"
    print(line, flush=True)
