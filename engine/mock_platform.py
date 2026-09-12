"""Offline mock of the Vault Challenge: MPS, perturbed graph state, and HEA vaults
with the same probe()/attack() interface as real_backend.RealVault.

    score = R * 4c / (4c + delta)

    R     = P(|0...0>) of (A . U)|0>
    c     = two qubit gate count of the hidden vault circuit U
    delta = two qubit gate count of the attacker circuit A

Usage: python engine/mock_platform.py   (one empty probe per mock vault)
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

_SIM = AerSimulator()

# gates that count as "entangling" for cost purposes (named, for reference)
TWO_QUBIT_GATES = {"cx", "cz", "cy", "ch", "crx", "cry", "crz", "cp", "cu",
                   "swap", "iswap", "rxx", "ryy", "rzz", "rzx", "ecr"}


def count_2q(circ: QuantumCircuit) -> int:
    """Count instructions on 2+ qubits, except barrier, measure, delay, snapshot."""
    total = 0
    for inst in circ.data:
        name = inst.operation.name
        if name in ("barrier", "measure", "delay", "snapshot"):
            continue
        if len(inst.qubits) >= 2:
            total += 1
    return total


class Vault:
    """A sealed vault. The hidden circuit U is private."""

    # real challenge limits (scored vaults)
    MAX_PROBES = 20
    MAX_ATTACKS = 20

    def __init__(self, index: int, U: QuantumCircuit, family: str, seed: int = 0):
        self.index = index
        self._U = U
        self.num_qubits = U.num_qubits
        self.family = family              # bookkeeping and evaluation only
        self._c = count_2q(U)
        self._rng = np.random.default_rng(seed)
        self.probes_used = 0
        self.attacks_used = 0
        self.best_raw = 0.0
        self.best_score = 0.0

    # --- internal: build U then A, measure all qubits ------------------------
    def _run(self, A: QuantumCircuit | None, shots: int) -> dict[str, int]:
        n = self.num_qubits
        full = QuantumCircuit(n, n)
        full.compose(self._U, inplace=True)
        if A is not None:
            if A.num_qubits != n:
                raise ValueError(f"attacker circuit has {A.num_qubits} qubits, "
                                 f"vault has {n}")
            full.compose(A, inplace=True)
        full.measure(range(n), range(n))
        result = _SIM.run(full, shots=shots, seed_simulator=int(
            self._rng.integers(1 << 31))).result()
        return result.get_counts()

    @staticmethod
    def _p_zero(counts: dict[str, int], shots: int) -> float:
        zero = "0" * len(next(iter(counts)))
        return counts.get(zero, 0) / shots

    # --- public interface (same as real_backend.RealVault) -------------------
    def probe(self, A: QuantumCircuit | None = None, shots: int = 200) -> dict[str, int]:
        """Unscored. Returns the measurement histogram of (A.U)|0>."""
        if self.probes_used >= self.MAX_PROBES:
            raise RuntimeError(f"vault {self.index}: probe budget exhausted "
                               f"({self.MAX_PROBES})")
        self.probes_used += 1
        return self._run(A, shots)

    def attack(self, A: QuantumCircuit | None = None, shots: int = 2000) -> dict:
        """Scored. Returns raw R, score, delta, counts, and c_hint (the true c)."""
        if self.attacks_used >= self.MAX_ATTACKS:
            raise RuntimeError(f"vault {self.index}: attack budget exhausted "
                               f"({self.MAX_ATTACKS})")
        self.attacks_used += 1
        counts = self._run(A, shots)
        R = self._p_zero(counts, shots)
        delta = count_2q(A) if A is not None else 0
        factor = 4 * self._c / (4 * self._c + delta) if self._c else 1.0
        score = R * factor
        self.best_raw = max(self.best_raw, R)
        self.best_score = max(self.best_score, score)
        return {"raw": R, "score": score, "delta": delta,
                "counts": counts, "c_hint": self._c}


# ---------------------------------------------------------------------------
# Vault generators
# ---------------------------------------------------------------------------
def make_mps_vault(index: int, n: int, seed: int, coupling: float = 0.9) -> QuantumCircuit:
    """Bond dimension 2 MPS: RY layer, CX RY CX staircase, then RZ layer."""
    rng = np.random.default_rng(seed)
    cpl = coupling * (1.0 + 0.25 * index)        # later MPS vaults entangle more
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(rng.uniform(0.3, 1.2), q)
    for q in range(n - 1):                        # one entangling block per bond
        qc.cx(q, q + 1)
        qc.ry(rng.uniform(-cpl, cpl), q + 1)
        qc.cx(q, q + 1)
    for q in range(n):
        qc.rz(rng.uniform(0, 0.6), q)
    return qc


def make_graph_vault(index: int, n: int, seed: int, edge_p: float = 0.4,
                     perturb: float = 0.18) -> QuantumCircuit:
    """Perturbed graph state: H^n, CZ over a random graph, small 1q rotations."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < edge_p:
                qc.cz(i, j)
                edges.append((i, j))
    if not edges:                                # guarantee at least one edge
        qc.cz(0, 1)
    for q in range(n):                           # small single qubit perturbation
        qc.rz(rng.uniform(-perturb, perturb), q)
        qc.rx(rng.uniform(-perturb, perturb), q)
    return qc


def make_hea_vault(index: int, n: int, seed: int, layers: int = 4) -> QuantumCircuit:
    """HEA: many CX layers (large c), small rotation angles (low entanglement)."""
    rng = np.random.default_rng(seed)
    amp = 0.35                                    # small angles, low entanglement
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-amp, amp), q)
            qc.rz(rng.uniform(-amp, amp), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(rng.uniform(0, np.pi), q)           # final single qubit layer
        qc.rz(rng.uniform(0, np.pi), q)
    return qc


def build_challenge(seed: int = 2026) -> list[Vault]:
    """12 scored vaults, 4 per family, spanning small qubit counts."""
    vaults = []
    specs = [
        ("mps", [4, 6, 8, 10]),
        ("graph", [4, 5, 6, 8]),
        ("hea", [3, 4, 5, 6]),
    ]
    idx = 1
    for fam, sizes in specs:
        for n in sizes:
            s = seed + idx * 101
            if fam == "mps":
                U = make_mps_vault(idx, n, s)
            elif fam == "graph":
                U = make_graph_vault(idx, n, s)
            else:
                U = make_hea_vault(idx, n, s)
            vaults.append(Vault(idx, U, fam, seed=s))
            idx += 1
    return vaults


if __name__ == "__main__":
    for v in build_challenge():
        empty_hist = v.probe(None)
        top = max(empty_hist.items(), key=lambda kv: kv[1])
        print(f"vault {v.index:2d}  {v.family:5s}  n={v.num_qubits:2d}  "
              f"c(2q)={v._c:2d}  empty-probe top={top[0]} ({top[1]}/200)")
