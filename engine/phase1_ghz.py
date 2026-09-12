"""Check bit order and histogram format on practice vault 0 (3 qubit GHZ, unscored).

Usage: python engine/phase1_ghz.py
"""

from qiskit import QuantumCircuit
from connect import _find_key
from real_backend import _call_with_retry, normalize_hist


def main():
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1

    client = VaultClient(QbraidSessionV1(api_key=_find_key()))

    # 1) empty probe on vault 0: expect about half 000 and half 111
    empty = QuantumCircuit(3)
    hist = _call_with_retry(client.probe, 0, empty)
    print("empty probe v0 (raw decimal keys):", hist)
    print("empty probe v0 (binary):", normalize_hist(hist, 3))

    # 2) known GHZ inverse: expect rawScore 1.0 and score 0.8
    inv = QuantumCircuit(3)
    inv.cx(0, 2)
    inv.cx(0, 1)
    inv.h(0)
    res = _call_with_retry(client.attack, 0, inv)
    print("\nGHZ inverse attack v0:", res)
    raw = float(res.get("rawScore", 0))
    print(f"\n{'PASS' if abs(raw-1.0) < 0.03 else 'CHECK'}: rawScore={raw:.3f} "
          f"(expect 1.0). costFactor={res.get('costFactor')}, score={res.get('score')}")
    if abs(raw - 1.0) >= 0.03:
        print("  bit order convention differs; fix it before attacking scored vaults.")


if __name__ == "__main__":
    main()
