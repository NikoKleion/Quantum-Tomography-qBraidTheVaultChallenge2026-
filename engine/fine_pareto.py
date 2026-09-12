"""Delta vs overlap frontier: greedily drop CX gates from a fitted ansatz, refitting the
rotations to the reconstructed state after each drop.

Usage: python engine/fine_pareto.py   # runs on compressible mock MPS vaults
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit

from hea_tomography import (_apply_1q, _apply_cx, _ry, _rz, num_params,
                            fit_ansatz, measurement_bases, collect)


# ---------------------------------------------------------------------------
# masked ansatz (some CX gates removed)
# ---------------------------------------------------------------------------
def masked_state(params, n, layers, keep):
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    p = iter(params)
    m = iter(keep)
    for _ in range(layers):
        for q in range(n):
            psi = _apply_1q(psi, q, _ry(next(p)))
            psi = _apply_1q(psi, q, _rz(next(p)))
        for q in range(n - 1):
            if next(m):
                psi = _apply_cx(psi, q, q + 1)
    for q in range(n):
        psi = _apply_1q(psi, q, _ry(next(p)))
        psi = _apply_1q(psi, q, _rz(next(p)))
    return psi


def masked_circuit(params, n, layers, keep) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    p = iter(params)
    m = iter(keep)
    for _ in range(layers):
        for q in range(n):
            qc.ry(next(p), q)
            qc.rz(next(p), q)
        for q in range(n - 1):
            if next(m):
                qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(next(p), q)
        qc.rz(next(p), q)
    return qc


def _overlap(psi_target, params, n, layers, keep):
    return abs(np.vdot(psi_target, masked_state(params, n, layers, keep))) ** 2


def _fit_to_state(psi_target, n, layers, keep, warm, maxiter=150):
    def loss(x):
        return 1.0 - _overlap(psi_target, x, n, layers, keep)
    res = minimize(loss, warm, method="Powell",
                   options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5})
    return res.x, 1.0 - res.fun


# ---------------------------------------------------------------------------
# greedy Pareto over delta  (O(gates) refits)
# ---------------------------------------------------------------------------
def fine_pareto(psi_target, full_params, n, layers):
    """Returns [(delta, overlap, disentangler)] from all CX kept down to none."""
    ncx = layers * (n - 1)
    keep = [True] * ncx
    params = full_params
    pts = [(sum(keep), 1.0, masked_circuit(params, n, layers, keep).inverse())]
    while any(keep):
        best_idx, best_ov = None, -1.0
        for idx in [i for i, k in enumerate(keep) if k]:
            tk = list(keep)
            tk[idx] = False
            ov = _overlap(psi_target, params, n, layers, tk)   # no refit
            if ov > best_ov:
                best_ov, best_idx = ov, idx
        keep[best_idx] = False
        params, ov = _fit_to_state(psi_target, n, layers, keep, params)  # one refit
        pts.append((sum(keep), ov,
                    masked_circuit(params, n, layers, keep).inverse()))
    return pts


def fine_pareto_candidates(vault, max_layers=2, num_random=9, seed=0,
                           reuse_data=None):
    """Probe, fit and run fine_pareto. Returns (candidates, resid, psi_est)."""
    n = vault.num_qubits
    if reuse_data:
        have = {b for b, _ in reuse_data}
        extra = [b for b in measurement_bases(n, num_random, seed) if b not in have]
        data = list(reuse_data) + collect(vault, extra[:num_random])
    else:
        data = collect(vault, measurement_bases(n, num_random, seed))
    params, resid = fit_ansatz(data, n, max_layers, restarts=3, seed=seed,
                               maxiter=500)
    psi_est = masked_state(params, n, max_layers, [True] * (max_layers * (n - 1)))
    pts = fine_pareto(psi_est, params, n, max_layers)
    # label by delta and predicted overlap
    out = []
    for delta, ov, A in pts:
        out.append((f"fp_d{delta}_o{ov:.2f}", A, delta, ov))
    return out, resid, psi_est


if __name__ == "__main__":
    # check on compressible MPS vaults
    import numpy as np
    from qiskit import QuantumCircuit as QC
    from mock_platform import Vault, count_2q

    def compressible_mps(n, seed):
        rng = np.random.default_rng(seed)
        qc = QC(n)
        for q in range(n):
            qc.ry(rng.uniform(0.3, 1.2), q)
        strong = rng.integers(0, n - 1)              # one strong bond, rest weak
        for q in range(n - 1):
            cpl = 1.1 if q == strong else 0.06       # weak bonds ~ droppable
            qc.cx(q, q + 1); qc.ry(cpl, q + 1); qc.cx(q, q + 1)
        return qc

    for n in (4, 5):
        v = Vault(99, compressible_mps(n, 7 + n), "mps", seed=7 + n)
        cands, resid, _psi = fine_pareto_candidates(v, max_layers=2, num_random=9)
        print(f"\ncompressible MPS n={n} c={v._c} fit_resid={resid:.3f}")
        best = None
        for label, A, delta, ov in cands:
            if v.attacks_used >= v.MAX_ATTACKS:
                break
            res = v.attack(A)
            mark = ""
            if best is None or res["score"] > best:
                best = res["score"]; mark = " <-"
            print(f"  d={delta:2d} pred_overlap={ov:.3f} "
                  f"R={res['raw']:.3f} score={res['score']:.3f}{mark}")
        print(f"  best {best:.3f}   (full inverse ceiling ~0.80)")
