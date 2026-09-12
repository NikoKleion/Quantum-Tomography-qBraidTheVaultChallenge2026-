"""Adaptive oracle loop at n=5 with a noiseless oracle on two simulated HEA
targets: 20 product queries, then fit and attack until 45 queries.

Usage: python experiments/exp_adaptive_n5.py
"""
import sys
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from live import make_queries, build_rows, product_attack, row_of_circuit, recover_rank1
from fine_pareto import _fit_to_state, fine_pareto
from crack import minimize_delta, count_2q

N = 5
BUDGET = 45


def targets():
    from mock_platform import make_hea_vault
    U1 = make_hea_vault(11, N, seed=2026, layers=4)
    rng = np.random.default_rng(77)
    U2 = QuantumCircuit(N)
    for L in range(2):
        for q in range(N):
            U2.ry(rng.uniform(0, 2 * np.pi), q); U2.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(N - 1):
            U2.cx(q, q + 1)
    for q in range(N):
        U2.ry(rng.uniform(0, 2 * np.pi), q); U2.rz(rng.uniform(0, 2 * np.pi), q)
    return [("mockHEA", U1), ("hardHEA", U2)]


def true_R(psi_true, A):
    return float(abs(Statevector(psi_true).evolve(A).data[0]) ** 2)


for name, U in targets():
    psi_true = Statevector(U).data
    queries = make_queries(N, 20, seed=42)
    rows = list(build_rows(N, queries))
    qk = [true_R(psi_true, product_attack(N, a)) for a in queries]
    used = len(qk)
    bestR, trace = 0.0, []
    psi_prev = None
    while used < BUDGET:
        psi_est, resid = recover_rank1(np.array(rows), np.array(qk), 1 << N,
                                       restarts=30, seed=used, psi0=psi_prev)
        psi_prev = psi_est
        fid = abs(np.vdot(psi_true, psi_est)) ** 2
        rng = np.random.default_rng(used)
        bestA, bestPred = None, -1
        for L in (1, 2):
            x0 = rng.uniform(0, 2 * np.pi, 2 * N * (L + 1))
            params, ov = _fit_to_state(psi_est, N, L, [True] * (L * (N - 1)),
                                       x0, maxiter=300)
            for d, f, A in fine_pareto(psi_est, params, N, L):
                if f > bestPred:
                    bestPred, bestA = f, minimize_delta(A)
        R = true_R(psi_true, bestA)
        bestR = max(bestR, R)
        rows.append(row_of_circuit(bestA, N)); qk.append(R)
        used += 1
        trace.append((used, round(fid, 3), round(R, 3), round(bestR, 3)))
    print(f"{name}: bestR@{BUDGET}={bestR:.3f}")
    print("   (queries, recon_fid, gotR, bestR):", trace[::3], flush=True)
