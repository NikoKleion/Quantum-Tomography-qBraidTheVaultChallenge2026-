"""Disentangle a known statevector to |0..0> with SVD two qubit gates on nearest
neighbour bonds. Skipping weak bonds trades fidelity for fewer CX.

Usage: python engine/svd_disentangle.py   # checks on synthetic MPS states
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


# ---------------------------------------------------------------------------
# statevector two qubit apply and matricization  (index k: qubit i = bit i)
# ---------------------------------------------------------------------------
def apply_2q(psi, n, qa, qb, G):
    T = psi.reshape([2] * n)
    axa, axb = n - 1 - qa, n - 1 - qb
    T = np.moveaxis(T, [axa, axb], [0, 1])
    rest = T.shape[2:]
    T = (G @ T.reshape(4, -1)).reshape((2, 2) + rest)
    T = np.moveaxis(T, [0, 1], [axa, axb])
    return T.reshape(-1)


def apply_1q(psi, n, q, U):
    T = psi.reshape([2] * n)
    ax = n - 1 - q
    T = np.moveaxis(T, ax, 0)
    rest = T.shape[1:]
    T = (U @ T.reshape(2, -1)).reshape((2,) + rest)
    T = np.moveaxis(T, 0, ax)
    return T.reshape(-1)


def _two_qubit_matricize(psi, n, k):
    T = psi.reshape([2] * n)
    T = np.moveaxis(T, [n - 1 - k, n - 1 - (k + 1)], [0, 1])
    return T.reshape(4, -1)


def bond_entanglements(psi, n):
    """Entanglement (1 - largest Schmidt weight^2) across each cut {0..k}|{k+1..n-1}."""
    ent = []
    for k in range(n - 1):
        # axis j holds qubit n-1-j, so the leading n-1-k axes are qubits k+1..n-1
        T = psi.reshape([2] * n).reshape(2 ** (n - 1 - k), -1)
        s = np.linalg.svd(T, compute_uv=False)
        s = s / np.linalg.norm(s)
        ent.append(1.0 - s[0] ** 2)
    return np.array(ent)


# ---------------------------------------------------------------------------
# disentangler for a chosen set of bonds
# ---------------------------------------------------------------------------
def build_disentangler(psi, n, keep_bonds):
    """Returns (A, final state, |<0|A|psi>|^2) using only the kept bonds."""
    state = psi.astype(complex).copy()
    qc = QuantumCircuit(n)
    for k in range(n - 1):
        if k in keep_bonds:
            U, _, _ = np.linalg.svd(_two_qubit_matricize(state, n, k))
            G = U.conj().T                        # 4x4, maps the column space to qk=0
            state = apply_2q(state, n, k, k + 1, G)
            # qiskit takes qubits LSB first and G has qubit k as MSB, so pass [k+1, k]
            qc.append(UnitaryGate(G, label=f"D{k}"), [k + 1, k])
        # skipped bond: qubit k stays entangled
    # align the last qubit to |0>
    rho_last = _single_qubit_rho(state, n, n - 1)
    Ulast = _align_to_zero(rho_last)
    state = apply_1q(state, n, n - 1, Ulast)
    qc.append(UnitaryGate(Ulast, label="F"), [n - 1])
    # single qubit cleanup for qubits left misaligned by skipped bonds
    for q in range(n - 1):
        rho = _single_qubit_rho(state, n, q)
        Uq = _align_to_zero(rho)
        if not np.allclose(Uq, np.eye(2), atol=1e-6):
            state = apply_1q(state, n, q, Uq)
            qc.append(UnitaryGate(Uq, label=f"c{q}"), [q])
    fidelity = float(np.abs(state[0]) ** 2)
    return qc, state, fidelity


def _single_qubit_rho(psi, n, q):
    T = psi.reshape([2] * n)
    T = np.moveaxis(T, n - 1 - q, 0).reshape(2, -1)
    return T @ T.conj().T


def _align_to_zero(rho):
    """Single qubit unitary that rotates the top eigenvector of rho to |0>."""
    w, v = np.linalg.eigh(rho)
    top = v[:, np.argmax(w)]
    U = np.array([[np.conj(top[0]), np.conj(top[1])],
                  [-top[1], top[0]]], dtype=complex)
    return U


# ---------------------------------------------------------------------------
# Pareto: keep the m most entangled bonds, m = 0..n-1
# ---------------------------------------------------------------------------
def svd_pareto(psi, n):
    """Returns [(fidelity, kept_bonds)], dropping the weakest bond at each step."""
    ent = bond_entanglements(psi, n)
    order = list(np.argsort(ent))                 # weakest first
    keep = set(range(n - 1))
    pts = []
    _, _, fid = build_disentangler(psi, n, keep)
    pts.append((fid, set(keep)))
    for weak in order:                            # drop weakest bonds one by one
        keep = keep - {int(weak)}
        _, _, fid = build_disentangler(psi, n, keep)
        pts.append((fid, set(keep)))
    return pts


def _block_ent(psi, n, qubits):
    """Entanglement (1 - top Schmidt weight^2) of a qubit block vs the rest."""
    T = psi.reshape([2] * n)
    axes = [n - 1 - q for q in qubits]
    T = np.moveaxis(T, axes, list(range(len(qubits))))
    s = np.linalg.svd(T.reshape(2 ** len(qubits), -1), compute_uv=False)
    s = s / np.linalg.norm(s)
    return 1.0 - s[0] ** 2


def _disentangle_gate(psi, n, a, b):
    """4x4 gate on (a,b) sending qubit a toward |0> (exact if rank({a,b}|rest)<=2)."""
    T = psi.reshape([2] * n)
    T = np.moveaxis(T, [n - 1 - a, n - 1 - b], [0, 1])
    U, _, _ = np.linalg.svd(T.reshape(4, -1))
    return U.conj().T


def general_pareto(psi, n, eps=1e-3):
    """Greedy any pair disentangler. Returns [(fidelity, circuit)] per gate prefix."""
    state = psi.astype(complex).copy()
    seq = []                                             # (a, b, G)
    done = set()
    while len(done) < n - 1:
        ent = {q: _block_ent(state, n, (q,)) for q in range(n) if q not in done}
        a = max(ent, key=ent.get)
        if ent[a] < eps:
            break
        b = min((q for q in range(n) if q != a),
                key=lambda q: _block_ent(state, n, tuple(sorted((a, q)))))
        G = _disentangle_gate(state, n, a, b)
        state = apply_2q(state, n, a, b, G)
        seq.append((a, b, G))
        done.add(a)
    # prefix Pareto: first k gates plus single qubit cleanup
    pts = []
    for k in range(len(seq) + 1):
        qc = QuantumCircuit(n)
        st = psi.astype(complex).copy()
        for (a, b, G) in seq[:k]:
            qc.append(UnitaryGate(G, label=f"g{a}{b}"), [b, a])  # a is MSB in G
            st = apply_2q(st, n, a, b, G)
        for q in range(n):                               # single qubit cleanup
            Uq = _align_to_zero(_single_qubit_rho(st, n, q))
            if not np.allclose(Uq, np.eye(2), atol=1e-6):
                qc.append(UnitaryGate(Uq, label=f"c{q}"), [q])
                st = apply_1q(st, n, q, Uq)
        pts.append((float(np.abs(st[0]) ** 2), qc))
    return pts


if __name__ == "__main__":
    from qiskit import transpile

    def bond2_mps(n, seed):
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(rng.uniform(0.3, 1.2), q)
        for q in range(n - 1):
            qc.cx(q, q + 1); qc.ry(rng.uniform(0.4, 1.0), q + 1); qc.cx(q, q + 1)
        from qiskit.quantum_info import Statevector
        return Statevector(qc).data

    print("Exactness on bond 2 MPS (full keep should give fidelity 1.0):")
    for n in (4, 5, 6, 7):
        psi = bond2_mps(n, n)
        A, st, fid = build_disentangler(psi, n, set(range(n - 1)))
        d = sum(v for g, v in transpile(A, basis_gates=["u3", "cx"],
                optimization_level=3).count_ops().items() if g == "cx")
        print(f"  n={n}: full keep fidelity={fid:.4f}  delta(CX)={d}")

    print("\nCompressible MPS (one strong bond) Pareto:")
    def compressible(n, seed):
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(rng.uniform(0.3, 1.2), q)
        strong = rng.integers(0, n - 1)
        for q in range(n - 1):
            cpl = 1.1 if q == strong else 0.05
            qc.cx(q, q + 1); qc.ry(cpl, q + 1); qc.cx(q, q + 1)
        from qiskit.quantum_info import Statevector
        return Statevector(qc).data
    psi = compressible(6, 5)
    for fid, keep in svd_pareto(psi, 6):
        A, _, _ = build_disentangler(psi, 6, keep)
        d = sum(v for g, v in transpile(A, basis_gates=["u3", "cx"],
                optimization_level=3).count_ops().items() if g == "cx")
        print(f"  keep={sorted(keep)} delta={d:2d} fidelity={fid:.4f}")
