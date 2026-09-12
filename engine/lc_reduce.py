"""Cut the CZ count of a graph state inverse with local complementation (LC).

LC at vertex a complements the edges among N(a). On the state:
    |tau_a(G)> = U_a |G>,   U_a = exp(i pi/4 X_a) * prod_{b in N(a)} exp(-i pi/4 Z_b)
             = RX(-pi/2)_a * prod_b RZ(pi/2)_b     (up to global phase).
If LC steps take G to G_min, A0 = H^n . CZ(E(G_min)) . U_LC maps |G> to |0>,
where U_LC is the product of the U_a in order. U_LC has only single qubit gates,
so delta = |E(G_min)|.

Usage: python engine/lc_reduce.py   (self test)
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def edges_to_adj(n: int, edges) -> np.ndarray:
    A = np.zeros((n, n), dtype=int)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    return A


def adj_to_edges(A: np.ndarray):
    n = A.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]


def local_complement(A: np.ndarray, v: int) -> np.ndarray:
    """Return graph after LC at v: complement edges among neighbours of v."""
    A = A.copy()
    nb = np.where(A[v] == 1)[0]
    for i in nb:
        for j in nb:
            if i < j:
                A[i, j] ^= 1
                A[j, i] = A[i, j]
    return A


def greedy_min_edges(A: np.ndarray, rounds: int = 200, seed: int = 0):
    """Greedy + randomized LC search minimizing edge count. Returns (A_min, seq)."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    def n_edges(M):
        return int(M.sum() // 2)

    best_A, best_seq = A.copy(), []
    cur_A, cur_seq = A.copy(), []
    for _ in range(rounds):
        # pick the LC that removes the most edges
        deltas = []
        for v in range(n):
            M = local_complement(cur_A, v)
            deltas.append((n_edges(M) - n_edges(cur_A), v))
        deltas.sort()
        if deltas[0][0] < 0:
            v = deltas[0][1]
        else:                                    # local min: random kick to escape
            v = int(rng.integers(n))
        cur_A = local_complement(cur_A, v)
        cur_seq = cur_seq + [v]
        if n_edges(cur_A) < n_edges(best_A):
            best_A, best_seq = cur_A.copy(), list(cur_seq)
    return best_A, best_seq


def lc_clifford_circuit(n: int, A_start: np.ndarray, seq) -> QuantumCircuit:
    """Build U_LC = product of U_a for the LC sequence (replaying neighbourhoods)."""
    qc = QuantumCircuit(n)
    A = A_start.copy()
    for v in seq:
        nb = np.where(A[v] == 1)[0]
        qc.rx(-np.pi / 2, v)                     # exp(i pi/4 X_v)
        for b in nb:
            qc.rz(np.pi / 2, int(b))             # exp(-i pi/4 Z_b)
        A = local_complement(A, v)
    return qc


def lc_inverter(n: int, edges, seed: int = 0):
    """Return (A0 circuit, delta) inverting the graph state with minimal CZ."""
    A = edges_to_adj(n, edges)
    A_min, seq = greedy_min_edges(A, seed=seed)
    if int(A_min.sum() // 2) >= len(edges):
        return None, len(edges)                  # no reduction found
    qc = lc_clifford_circuit(n, A, seq)          # U_LC first
    for (i, j) in adj_to_edges(A_min):           # then CZ on reduced graph
        qc.cz(i, j)
    qc.h(range(n))                               # |+>^n to |0>^n
    return qc, int(A_min.sum() // 2)


if __name__ == "__main__":
    from qiskit.quantum_info import Statevector

    def graph_state(n, edges):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for i, j in edges:
            qc.cz(i, j)
        return qc

    # complete graph K_n is LC equivalent to a star with n-1 edges
    for n in (4, 5, 6):
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        A0, delta = lc_inverter(n, edges)
        psi = Statevector(graph_state(n, edges)).evolve(A0)
        p0 = float(np.abs(psi.data[0]) ** 2)
        print(f"K_{n}: |E|={len(edges)} -> LC delta={delta}  P(0)={p0:.6f} "
              f"{'PASS' if p0 > 0.999 else 'FAIL'}")

    # regression case: graph learned on live vault 6
    edges = [(0, 4), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
    A0, delta = lc_inverter(6, edges)
    psi = Statevector(graph_state(6, edges)).evolve(A0)
    p0 = float(np.abs(psi.data[0]) ** 2)
    print(f"V6 graph: |E|=6 -> LC delta={delta}  P(0)={p0:.6f} "
          f"{'PASS' if p0 > 0.999 else 'FAIL'}")
