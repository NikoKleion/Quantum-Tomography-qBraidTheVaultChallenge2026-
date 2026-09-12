"""Adaptive oracle loop at n=4 with 0 probes and 19 attacks on simulated HEA
targets: 12 product queries, then fit, attack the best candidate, add its row.

Usage: python experiments/exp_adaptive.py
"""
import sys
import numpy as np
sys.path.insert(0, "engine")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from mock_platform import count_2q
from fine_pareto import _fit_to_state, fine_pareto
from crack import minimize_delta
from exp_oracle_tomo import (hea_targets, build_rows, product_attack, true_R,
                             recover_rank1)

BUDGET = 19


def row_of_circuit(A, n):
    """<0...0| A as a row vector, for any circuit."""
    U = Operator(A).data
    return U[0, :]


for n in (4,):
    for name, U in hea_targets(n):
        psi_true = Statevector(U).data
        c_true = count_2q(U)
        rng = np.random.default_rng(42)
        queries = [[(0.0, 0.0)] * n]
        while len(queries) < 12:
            queries.append([(rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi))
                            for _ in range(n)])
        rows = list(build_rows(n, queries))
        qk = [true_R(psi_true, product_attack(n, a)) for a in queries]
        used = len(qk)
        best_score_live = 0.0
        best_R = 0.0
        while used < BUDGET:
            psi_est, resid = recover_rank1(np.array(rows), np.array(qk), 1 << n,
                                           restarts=8, seed=used)
            # candidate pool from this recon; pick best predicted score
            rng2 = np.random.default_rng(used)
            bestA, bestPred = None, -1
            for L in (1, 2):
                x0 = rng2.uniform(0, 2 * np.pi, 2 * n * (L + 1))
                params, ov = _fit_to_state(psi_est, n, L,
                                           [True] * (L * (n - 1)), x0, maxiter=300)
                for d, fid, A in fine_pareto(psi_est, params, n, L):
                    A2 = minimize_delta(A)
                    pred = fid * 4 * c_true / (4 * c_true + count_2q(A2))
                    if pred > bestPred:
                        bestPred, bestA = pred, A2
            # attack it; the exact R also becomes a new row
            R = true_R(psi_true, bestA)
            s = R * 4 * c_true / (4 * c_true + count_2q(bestA))
            best_score_live = max(best_score_live, s)
            best_R = max(best_R, R)
            rows.append(row_of_circuit(bestA, n))
            qk.append(R)
            used += 1
            fid_true = abs(np.vdot(psi_true, psi_est)) ** 2
            print(f"  {name} n={n}: attack#{used} recon_fid={fid_true:.3f} "
                  f"pred={bestPred:.3f} gotR={R:.3f} score={s:.3f} "
                  f"best={best_score_live:.3f}", flush=True)
        print(f"=> {name} n={n}: final best score {best_score_live:.3f} "
              f"({BUDGET} attacks, 0 probes)", flush=True)
