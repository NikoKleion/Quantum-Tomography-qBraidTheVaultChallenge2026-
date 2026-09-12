"""Histogram helpers, single qubit Bloch estimates, the product inverter
(delta 0), and vault family detection.

Counts keys are in Qiskit order (qubit n-1 leftmost). counts_to_bits flips them
so bits[:, i] is qubit i.

Usage: python engine/toolkit.py   (product inverter on the mock vaults)
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


# ---------------------------------------------------------------------------
# histogram helpers
# ---------------------------------------------------------------------------
def counts_to_bits(counts: dict[str, int], n: int):
    """Return (bits, weights), one row per outcome, with bits[:, i] = qubit i."""
    rows, w = [], []
    for s, c in counts.items():
        be = s.replace(" ", "")
        le = be[::-1]                       # le[i] = qubit i
        rows.append([int(ch) for ch in le])
        w.append(c)
    return np.array(rows), np.array(w, dtype=float)


def expectation_z(counts: dict[str, int], n: int) -> np.ndarray:
    """<Z_i> for each qubit i, in the basis the probe was taken in."""
    bits, w = counts_to_bits(counts, n)
    p1 = (bits * w[:, None]).sum(0) / w.sum()
    return 1.0 - 2.0 * p1                    # <Z> = P0 - P1


def correlation_zz(counts: dict[str, int], n: int) -> np.ndarray:
    """<Z_i Z_j> matrix in the basis the probe was taken in."""
    bits, w = counts_to_bits(counts, n)
    signs = 1.0 - 2.0 * bits                 # +1 for 0, -1 for 1
    W = w.sum()
    return (signs * w[:, None]).T @ signs / W


# ---------------------------------------------------------------------------
# basis rotation circuits (single qubit gates only)
# ---------------------------------------------------------------------------
def basis_circuit(n: int, basis: str) -> QuantumCircuit:
    """Rotate every qubit so that measuring in Z reads the chosen Pauli."""
    qc = QuantumCircuit(n)
    if basis == "Z":
        return qc
    for q in range(n):
        if basis == "X":
            qc.h(q)
        elif basis == "Y":
            qc.sdg(q)
            qc.h(q)
    return qc


# ---------------------------------------------------------------------------
# single qubit tomography
# ---------------------------------------------------------------------------
def estimate_bloch(vault, shots: int = 200) -> np.ndarray:
    """Return Bloch vectors r[n,3] = (<X>,<Y>,<Z>) using 3 probes."""
    n = vault.num_qubits
    r = np.zeros((n, 3))
    for k, basis in enumerate("XYZ"):
        counts = vault.probe(basis_circuit(n, basis), shots=shots)
        r[:, k] = expectation_z(counts, n)
    return r


# ---------------------------------------------------------------------------
# product inverter (delta 0)
# ---------------------------------------------------------------------------
def product_inverter(bloch: np.ndarray) -> QuantumCircuit:
    """Rotate each qubit's Bloch vector to +z (|0>). No two qubit gates."""
    n = bloch.shape[0]
    qc = QuantumCircuit(n)
    for i in range(n):
        rx, ry, rz = bloch[i]
        norm = np.linalg.norm(bloch[i])
        if norm < 1e-9:
            continue                         # maximally mixed: skip
        theta = np.arccos(np.clip(rz / norm, -1, 1))
        phi = np.arctan2(ry, rx)
        qc.rz(-phi, i)
        qc.ry(-theta, i)
    return qc


# ---------------------------------------------------------------------------
# family detection from cheap probes
# ---------------------------------------------------------------------------
def detect_family(vault, bloch: np.ndarray) -> str:
    """Classify a vault from single qubit purities."""
    purity = np.linalg.norm(bloch, axis=1)   # |r|: 1 = pure, 0 = max mixed
    mean_pure = purity.mean()
    if mean_pure > 0.8:
        return "mps"                         # nearly a product state
    if mean_pure < 0.35:
        return "graph"                       # marginals near maximally mixed
    return "hea"


if __name__ == "__main__":
    from mock_platform import build_challenge

    print(f"{'vault':>5} {'fam':>5} {'n':>3} {'detect':>7} "
          f"{'meanPure':>9} {'R(Δ=0)':>8}")
    for v in build_challenge():
        bloch = estimate_bloch(v, shots=400)
        fam = detect_family(v, bloch)
        A = product_inverter(bloch)
        res = v.attack(A, shots=2000)
        print(f"{v.index:5d} {v.family:>5} {v.num_qubits:3d} {fam:>7} "
              f"{np.linalg.norm(bloch, axis=1).mean():9.3f} {res['raw']:8.3f}")
