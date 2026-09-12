"""Simulate the oracle tomography loop offline on two HEA targets for a given width
and budget, and print the score for three seeds.

Usage: python engine/rehearse_offline.py <n> <probes> <attacks> [reserve]
"""
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import live
from mock_platform import make_hea_vault, count_2q
from hea_tomography import measurement_bases, rotate_probs


def targets(n):
    U1 = make_hea_vault(9, n, seed=1000 + n, layers=4)
    rng = np.random.default_rng(500 + n)
    U2 = QuantumCircuit(n)
    for _ in range(2):
        for q in range(n):
            U2.ry(rng.uniform(0, 2 * np.pi), q); U2.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(n - 1):
            U2.cx(q, q + 1)
    for q in range(n):
        U2.ry(rng.uniform(0, 2 * np.pi), q); U2.rz(rng.uniform(0, 2 * np.pi), q)
    return [("mockHEA", U1), ("hardHEA", U2)]


def run_pipeline(psi_true, c_true, n, n_probes, n_attacks, reserve=2, seed=0,
                 verbose=False):
    """Return (best score, best R) after n_probes probes and up to n_attacks attacks."""
    # attack R is exact, like the server's rawScore
    def oracle(A):
        return float(abs(Statevector(psi_true).evolve(A).data[0]) ** 2)

    rng = np.random.default_rng(seed)
    # probes sampled with 200 shots, like the server
    probe_data = []
    if n_probes:
        for b in measurement_bases(n, max(0, n_probes - 3), seed=seed)[:n_probes]:
            p = rotate_probs(psi_true, b)
            probe_data.append((b, rng.multinomial(200, p) / 200.0))

    budget = n_attacks
    n_queries = max(0, min(budget - reserve - 3, 17))
    queries = live.make_queries(n, n_queries, seed=42) if n_queries else []
    rows = list(live.build_rows(n, queries)) if n_queries else []
    qk = [oracle(live.product_attack(n, a)) for a in queries]
    used = len(qk)
    best_s, best_R = 0.0, 0.0

    while used < budget:
        if probe_data:
            psi_est, _ = live.recover_joint(np.array(rows) if rows else np.zeros((0, 1 << n), complex),
                                            np.array(qk) if qk else np.zeros(0),
                                            probe_data, n, restarts=8, seed=used)
        else:
            psi_est, _ = live.recover_rank1(np.array(rows), np.array(qk), 1 << n,
                                            restarts=8, seed=used)
        pool = live.candidates_from_psi(psi_est, n, seed=used)
        ranked = live.rank_by_score(pool, c_true)
        if not ranked:
            break
        pred, A, d, rhat = ranked[0]
        R = oracle(A)
        s = R * (1.0 if d == 0 else 4 * c_true / (4 * c_true + d))
        best_s, best_R = max(best_s, s), max(best_R, R)
        rows.append(live.row_of_circuit(A, n)); qk.append(R)
        used += 1
        if verbose:
            fid = abs(np.vdot(psi_true, psi_est)) ** 2
            print(f"    attack#{used} fid={fid:.3f} pred={rhat:.3f} R={R:.3f} "
                  f"d={d} s={s:.3f} best={best_s:.3f}", flush=True)
        if best_R > 0.995:
            break
    return best_s, best_R


if __name__ == "__main__":
    n = int(sys.argv[1]); P = int(sys.argv[2]); A = int(sys.argv[3])
    res = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    print(f"REHEARSAL n={n}, probes={P}, attacks={A}, reserve={res}")
    for name, U in targets(n):
        psi = Statevector(U).data; c = count_2q(U)
        scores = []
        for seed in (0, 1, 2):
            s, R = run_pipeline(psi, c, n, P, A, res, seed=seed)
            scores.append(s)
        print(f"  {name} (c={c}): scores={[f'{x:.3f}' for x in scores]} "
              f"median={np.median(scores):.3f}", flush=True)
