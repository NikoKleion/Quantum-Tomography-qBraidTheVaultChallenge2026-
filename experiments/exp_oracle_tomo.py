"""Recovers |psi> from exact attack scores q_k = |<0|A_k|psi>|^2 with product
rotation attacks and no probes, on simulated HEA targets at n=3 and 4.

Usage: python experiments/exp_oracle_tomo.py [n_queries]
"""
import sys
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from mock_platform import make_hea_vault, count_2q
from fine_pareto import _fit_to_state, fine_pareto
from hea_tomography import fast_state
from crack import minimize_delta

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 17


def true_R(psi_true, A):
    return float(abs(Statevector(psi_true).evolve(A).data[0]) ** 2)


def hea_targets(n):
    U1 = make_hea_vault(9, n, seed=1000 + n, layers=4)
    rng = np.random.default_rng(500 + n)
    U2 = QuantumCircuit(n)
    for L in range(2):
        for q in range(n):
            U2.ry(rng.uniform(0, 2 * np.pi), q); U2.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(n - 1):
            U2.cx(q, q + 1)
    for q in range(n):
        U2.ry(rng.uniform(0, 2 * np.pi), q); U2.rz(rng.uniform(0, 2 * np.pi), q)
    return [("mockHEA", U1), ("hardHEA", U2)]


def product_attack(n, angles):
    """Rz(a) then Ry(b) on each qubit, with angles[q] = (a, b)."""
    A = QuantumCircuit(n)
    for q in range(n):
        a, b = angles[q]
        A.rz(a, q); A.ry(b, q)
    return A


def one_qubit_u(a, b):
    """Matrix of Ry(b) @ Rz(a) (qiskit order rz then ry on the wire)."""
    rz = np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]])
    ry = np.array([[np.cos(b / 2), -np.sin(b / 2)], [np.sin(b / 2), np.cos(b / 2)]])
    return ry @ rz


def bras(n, query_angles):
    """<0|A_k as a vector: first row of kron(U_{n-1},...,U_0)."""
    out = []
    for angles in query_angles:
        row = np.array([1.0 + 0j])
        for q in range(n - 1, -1, -1):
            U = one_qubit_u(*angles[q])
            row = np.kron(U[0, :], row) if False else np.kron(row, U[0, :])
        # kron order: qiskit index bit q = qubit q (LSB first)
        out.append(row)
    return out


def build_rows(n, query_angles):
    rows = []
    for angles in query_angles:
        M = np.array([1.0 + 0j])
        for q in range(n - 1, -1, -1):          # kron MSB..LSB: qubit n-1 first
            M = np.kron(M, one_qubit_u(*angles[q])[0, :])
        rows.append(M)
    return np.array(rows)


def recover_rank1(rows, q, dim, restarts=10, seed=0):
    """min_psi sum_k (|rows_k . psi|^2 - q_k)^2, |psi|=1."""
    rng = np.random.default_rng(seed)
    best, bl = None, np.inf

    def loss_grad(x):
        psi = x[:dim] + 1j * x[dim:]
        nrm2 = float(np.vdot(psi, psi).real) or 1e-12
        amps = rows @ psi
        pred = np.abs(amps) ** 2 / nrm2
        diff = pred - q
        L = float(np.sum(diff ** 2))
        gpsi = (2.0 / nrm2) * (rows.conj().T @ (diff * amps))
        gpsi -= (2.0 * float(np.sum(diff * np.abs(amps) ** 2)) / nrm2 ** 2) * psi
        g = np.concatenate([2 * gpsi.real, 2 * gpsi.imag])
        return L, g

    for r in range(restarts):
        x0 = rng.standard_normal(2 * dim)
        res = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 800})
        if res.fun < bl:
            bl, best = res.fun, res.x
    psi = best[:dim] + 1j * best[dim:]
    return psi / np.linalg.norm(psi), bl


def best_score(psi_est, psi_true, n, c_true):
    best = (0.0, 0.0, 0)
    rng = np.random.default_rng(5)
    for L in (1, 2):
        x0 = rng.uniform(0, 2 * np.pi, 2 * n * (L + 1))
        params, ov = _fit_to_state(psi_est, n, L, [True] * (L * (n - 1)), x0,
                                   maxiter=400)
        for d, fid, A in fine_pareto(psi_est, params, n, L):
            A2 = minimize_delta(A); d2 = count_2q(A2)
            R = true_R(psi_true, A2)
            s = R * 4 * c_true / (4 * c_true + d2)
            if s > best[0]:
                best = (s, R, d2)
    return best


for n in (3, 4):
    rng = np.random.default_rng(42)
    # query set: identity + random product rotations
    queries = [[(0.0, 0.0)] * n]
    while len(queries) < NQ:
        queries.append([(rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
                        for _ in range(n)])
    rows = build_rows(n, queries)
    for name, U in hea_targets(n):
        psi_true = Statevector(U).data
        c_true = count_2q(U)
        # check the row convention against qiskit on query 1
        qk = np.array([true_R(psi_true, product_attack(n, a)) for a in queries])
        pred0 = abs(rows[1] @ psi_true) ** 2
        assert abs(pred0 - qk[1]) < 1e-9, f"convention mismatch {pred0} vs {qk[1]}"
        psi_est, resid = recover_rank1(rows, qk, 1 << n, restarts=12, seed=1)
        fid = abs(np.vdot(psi_true, psi_est)) ** 2
        s, R, d = best_score(psi_est, psi_true, n, c_true)
        print(f"n={n} {name}: {NQ} oracle queries -> recover fid={fid:.3f} "
              f"resid={resid:.2e} | final attack score={s:.3f} (R={R:.3f}, d={d})",
              flush=True)
