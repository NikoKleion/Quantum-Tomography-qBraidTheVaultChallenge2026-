"""Simulated state recovery at n=4 and 5: (A) staircase fit to 17 exact oracle
queries, (B) 3 probe bases plus 14 oracle queries.

Usage: python experiments/exp_oracle2.py
"""
import sys
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from mock_platform import make_hea_vault, count_2q
from fine_pareto import _fit_to_state, fine_pareto
from hea_tomography import fast_state, rotate_probs, measurement_bases
from crack import minimize_delta
from exp_oracle_tomo import (hea_targets, build_rows, product_attack, true_R,
                             recover_rank1, best_score)

NQ = 17


def oracle_prior_fit(rows, qk, n, L, restarts=6, seed=0, maxiter=500):
    """Fit staircase ansatz params to exact oracle values."""
    npar = 2 * n * (L + 1)
    rng = np.random.default_rng(seed)

    def loss(p):
        psi = fast_state(p, n, L)
        pred = np.abs(rows @ psi) ** 2
        return float(np.sum((pred - qk) ** 2))

    best, bl = None, np.inf
    for r in range(restarts):
        x0 = rng.uniform(0, 2 * np.pi, npar)
        res = minimize(loss, x0, method="Powell",
                       options={"maxiter": maxiter, "xtol": 1e-5, "ftol": 1e-7})
        if res.fun < bl:
            bl, best = res.fun, res.x
    return best, bl


def hybrid_recover(rows, qk, probe_data, n, lam=60.0, restarts=10, seed=0):
    """Rank 1 recovery from noisy probes plus exact oracle rows (weight lam)."""
    dim = 1 << n
    rng = np.random.default_rng(seed)
    from state_tomography import _rotate

    def loss_grad(x):
        psi = x[:dim] + 1j * x[dim:]
        nrm2 = float(np.vdot(psi, psi).real) or 1e-12
        L = 0.0
        g = np.zeros(dim, dtype=complex)
        for basis, pobs in probe_data:
            phi = _rotate(psi, basis)
            amp2 = np.abs(phi) ** 2
            probs = amp2 / nrm2
            diff = probs - pobs
            L += float(np.sum(diff ** 2))
            gphi = (2.0 / nrm2) * diff * phi
            g += _rotate(gphi, basis, dagger=True)
            g -= (2.0 * float(np.sum(diff * amp2)) / nrm2 ** 2) * psi
        amps = rows @ psi
        pred = np.abs(amps) ** 2 / nrm2
        diff = pred - qk
        L += lam * float(np.sum(diff ** 2))
        g += lam * (2.0 / nrm2) * (rows.conj().T @ (diff * amps))
        g -= lam * (2.0 * float(np.sum(diff * np.abs(amps) ** 2)) / nrm2 ** 2) * psi
        return L, np.concatenate([2 * g.real, 2 * g.imag])

    best, bl = None, np.inf
    for r in range(restarts):
        x0 = rng.standard_normal(2 * dim)
        res = minimize(loss_grad, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 900})
        if res.fun < bl:
            bl, best = res.fun, res.x
    psi = best[:dim] + 1j * best[dim:]
    return psi / np.linalg.norm(psi), bl


print("A) oracle only, staircase prior (17 exact queries, 0 probes)")
for n in (4, 5):
    rng = np.random.default_rng(42)
    queries = [[(0.0, 0.0)] * n]
    while len(queries) < NQ:
        queries.append([(rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
                        for _ in range(n)])
    rows = build_rows(n, queries)
    for name, U in hea_targets(n):
        psi_true = Statevector(U).data
        c_true = count_2q(U)
        qk = np.array([true_R(psi_true, product_attack(n, a)) for a in queries])
        line = f"  n={n} {name}:"
        for L in (1, 2):
            p, resid = oracle_prior_fit(rows, qk, n, L, restarts=3)
            psi_est = fast_state(p, n, L)
            psi_est = psi_est / np.linalg.norm(psi_est)
            fid = abs(np.vdot(psi_true, psi_est)) ** 2
            line += f" L{L}: fid={fid:.3f} resid={resid:.1e}"
        # score from the L=2 fit
        s, R, d = best_score(psi_est, psi_true, n, c_true)
        print(line + f" | score(L2 recon)={s:.3f}", flush=True)

print("B) hybrid: 3 probes (Z,X,Y @200 shots) + 14 exact oracle queries")
for n in (4, 5):
    rng = np.random.default_rng(42)
    queries = [[(0.0, 0.0)] * n]
    while len(queries) < 14:
        queries.append([(rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
                        for _ in range(n)])
    rows = build_rows(n, queries)
    for name, U in hea_targets(n):
        psi_true = Statevector(U).data
        c_true = count_2q(U)
        qk = np.array([true_R(psi_true, product_attack(n, a)) for a in queries])
        srng = np.random.default_rng(7)
        pdata = []
        for b in ("Z" * n, "X" * n, "Y" * n):
            p = rotate_probs(psi_true, b)
            p = np.clip(p.real, 0, None); p /= p.sum()
            pdata.append((b, srng.multinomial(200, p) / 200))
        psi_est, resid = hybrid_recover(rows, qk, pdata, n, restarts=8)
        fid = abs(np.vdot(psi_true, psi_est)) ** 2
        s, R, d = best_score(psi_est, psi_true, n, c_true)
        print(f"  n={n} {name}: fid={fid:.3f} | score={s:.3f} (R={R:.3f}, d={d})",
              flush=True)
