"""Learn graph state edges from probes and build the inverse circuit.

|G> = prod CZ_ij . H^n |0> has stabilisers K_i = X_i prod_{j in N(i)} Z_j, so
measuring qubit i in X and the rest in Z gives x_i = sum_{j in N(i)} z_j mod 2,
a GF(2) system per qubit. The inverse is A = H^n . prod CZ_ij.

Usage: python engine/graph_solver.py   (edge recovery check on the mock vaults)
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from toolkit import counts_to_bits


# ---------------------------------------------------------------------------
# GF(2) least squares
# ---------------------------------------------------------------------------
def _gf2_solve(Z: np.ndarray, x: np.ndarray, tries: int = 40, rng=None):
    """Return (a, fraction of rows satisfied) for noisy Z a = x (mod 2)."""
    rng = rng or np.random.default_rng(0)
    m, k = Z.shape
    best_a, best_agree = np.zeros(k, dtype=int), -1
    for _ in range(tries):
        idx = rng.choice(m, size=min(k, m), replace=False)
        A = Z[idx].copy() % 2
        b = x[idx].copy() % 2
        # forward elimination over GF(2)
        rows = A.shape[0]
        piv_cols, r = [], 0
        Aug = np.hstack([A, b[:, None]]) % 2
        for c in range(k):
            pr = None
            for rr in range(r, rows):
                if Aug[rr, c]:
                    pr = rr
                    break
            if pr is None:
                continue
            Aug[[r, pr]] = Aug[[pr, r]]
            for rr in range(rows):
                if rr != r and Aug[rr, c]:
                    Aug[rr] ^= Aug[r]
            piv_cols.append(c)
            r += 1
            if r == rows:
                break
        a = np.zeros(k, dtype=int)
        for i, c in enumerate(piv_cols):
            a[c] = Aug[i, -1]
        agree = int((((Z @ a) % 2) == x).sum())
        if agree > best_agree:
            best_agree, best_a = agree, a
    return best_a, (best_agree / m if m else 0.0)


def learn_edge_row(vault, i: int, shots: int = 200):
    """Probe X on qubit i, Z elsewhere; return neighbour indicator + confidence."""
    n = vault.num_qubits
    qc = QuantumCircuit(n)
    qc.h(i)                                   # measure qubit i in X
    counts = vault.probe(qc, shots=shots)
    bits, w = counts_to_bits(counts, n)
    # counts (mock) or probabilities (real): normalize, then scale to shots
    p = w / w.sum() if w.sum() else w
    reps = np.rint(p * shots).astype(int)
    reps = np.maximum(reps, (p > 0.5 / shots).astype(int))   # keep observed outcomes
    samples = np.repeat(bits, reps, axis=0)
    if samples.shape[0] == 0:                                # degenerate fallback
        samples = bits
    x = samples[:, i]
    others = [j for j in range(n) if j != i]
    Z = samples[:, others]
    a, conf = _gf2_solve(Z, x, rng=np.random.default_rng(i + 1))
    row = np.zeros(n, dtype=int)
    for idx, j in enumerate(others):
        row[j] = a[idx]
    return row, conf


def learn_graph(vault, shots: int = 200):
    """Recover the edge set. Uses n probes (X on each qubit, Z elsewhere)."""
    n = vault.num_qubits
    rows = np.zeros((n, n), dtype=int)
    confs = np.zeros(n)
    for i in range(n):
        rows[i], confs[i] = learn_edge_row(vault, i, shots=shots)
    # symmetrise: keep (i, j) if either row found it
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i, j] or rows[j, i]:
                edges.append((i, j))
    return edges, confs


def stabiliser_inverter(n: int, edges) -> QuantumCircuit:
    """CZ on every edge, then H on every qubit: maps |G> back to |0>."""
    qc = QuantumCircuit(n)
    for (i, j) in edges:
        qc.cz(i, j)
    qc.h(range(n))
    return qc


if __name__ == "__main__":
    from mock_platform import build_challenge
    print("Graph learning check (recovered edges vs. hidden truth):")
    for v in build_challenge():
        if v.family != "graph":
            continue
        true_edges = [(a, b) for (a, b) in _true_edges(v)] if False else None
        edges, confs = learn_graph(v, shots=200)
        # peek at ground truth for validation only
        from mock_platform import count_2q
        truth = set()
        for inst in v._U.data:
            if inst.operation.name == "cz":
                q = tuple(sorted(v._U.find_bit(b).index for b in inst.qubits))
                truth.add(q)
        found = set(edges)
        tp = len(found & truth)
        print(f"vault {v.index}: n={v.num_qubits} truth={len(truth)} "
              f"found={len(found)} correct={tp} "
              f"missing={sorted(truth-found)} extra={sorted(found-truth)} "
              f"minconf={confs.min():.2f}")
