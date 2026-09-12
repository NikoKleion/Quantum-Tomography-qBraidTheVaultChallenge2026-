"""CX staircase for large MPS vaults: CX on the nearest neighbour bonds with the largest
connected correlation, then single qubit realignment. Pareto over the number of CX.

Usage: python engine/mps_large.py   # runs on mock bond 2 MPS vaults, n=10 and 12
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from toolkit import basis_circuit, expectation_z, correlation_zz, product_inverter


def _bonds_by_entanglement(vault, n):
    """3 probes. Returns (Bloch vectors, summed connected NN correlation per bond)."""
    r = np.zeros((n, 3))
    conn_sum = np.zeros(n - 1)
    for k, b in enumerate("XYZ"):
        counts = vault.probe(basis_circuit(n, b))
        z = expectation_z(counts, n)
        r[:, k] = z
        cc = correlation_zz(counts, n) - np.outer(z, z)
        for i in range(n - 1):
            conn_sum[i] += abs(cc[i, i + 1])
    return r, conn_sum


def build_staircase(bloch, bonds, n):
    """Product inverter, then CX on each bond in the given order."""
    qc = product_inverter(bloch)                             # single qubit align first
    for (i, j) in bonds:
        qc.cx(i, j)
    return qc


def mps_large_candidates(vault, ks=None):
    """Returns [(label, circuit)] for increasing numbers of NN CX gates."""
    from crack import align_after, remaining
    n = vault.num_qubits
    bloch, conn = _bonds_by_entanglement(vault, n)           # 3 probes
    order = [i for i in np.argsort(-conn) if conn[i] > 0.02]  # entangled bonds only
    cands = [("mps_d0", product_inverter(bloch))]
    if ks is None:
        ks = [k for k in (4, 8, len(order)) if 0 < k <= len(order)]
        ks = sorted(set(ks))
    for k in ks:
        if remaining(vault) < 3:
            break
        # the k strongest bonds, applied in chain order
        chosen = sorted(int(i) for i in order[:k])
        bonds = [(i, i + 1) for i in chosen]
        base = build_staircase(bloch, bonds, n)
        cands.append((f"mps_d{len(bonds)}", align_after(vault, base)))  # 3 probes
    return cands


if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    from qiskit import transpile
    from mock_platform import Vault, count_2q

    def bond2_mps(n, seed):
        r = np.random.default_rng(seed)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(r.uniform(0.3, 1.0), q)
        for q in range(n - 1):
            qc.cx(q, q + 1); qc.ry(r.uniform(0.3, 0.8), q + 1); qc.cx(q, q + 1)
        return qc

    for n in (10, 12):
        v = Vault(99, bond2_mps(n, n), "mps", seed=1)
        print(f"--- bond 2 MPS n={n}, c={v._c} ---")
        best = 0
        for label, A in mps_large_candidates(v):
            res = v.attack(A)
            best = max(best, res["score"])
            print(f"  {label}: R={res['raw']:.3f} d={res['delta']} score={res['score']:.3f}")
        print(f"  best {best:.3f}")
